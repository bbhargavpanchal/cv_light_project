"""Head-anchored day/night toggle switch.

Sits just above the head -- position supplied by main.py every frame as
a plain (x, y) point derived from FaceTracker's eye-midpoint, so this
module has no MediaPipe or face knowledge of its own, same separation
as light_control.py not knowing about HandTracker. It flips day<->night
only when a hand pinches within reach of the switch -- not a gesture
anywhere on screen, the same "must actually be near the thing" rule the
light's own grab already follows so the two gestures don't trip over
each other. A held pinch flips it once, on the moment it starts (a
rising-edge check), not continuously while held.

The night amount is a plain exponential ramp toward 0 or 1, not a
Spring -- a fade should ease smoothly toward its target, not overshoot
and bounce back the way the light's pick-up "pop" deliberately does.
Because each step is a convex blend of the current value and a target
already in [0, 1], the result can never leave that range, so nothing
downstream needs to clamp it.
"""

import cv2
import numpy as np

from utils import EMASmoother, distance
from hand_tracker import THUMB_TIP, INDEX_TIP, is_pinching

TRACK_W, TRACK_H = 46, 22  # pill-switch footprint, px -- smaller and more "switch"-like than the old plain circle
THUMB_R = TRACK_H // 2 - 2
GRAB_RADIUS = 50           # extra reach beyond the icon's own edge that still counts as "near", px -- tune here first
TRANSITION_RATE = 0.06     # per-frame ramp toward the target -- smaller = slower fade
MAX_DIM = 0.45             # night never dims past this fraction -- "little bit darker", not pitch black
TINT_COLOR = (40, 20, 10)  # BGR, deep navy -- the colour the frame fades toward at night

# Fixed relative (x, y) offsets, from the track's left end, for the
# little night-sky dots -- precomputed once rather than re-randomised
# every frame, which would just flicker.
_STAR_OFFSETS = [(8, -4), (14, 5), (20, -6), (26, 3)]

# Rendering polish: the switch is drawn onto a small canvas at SS times
# its final size (gradient shading, glossy thumb, soft shadow, all
# included) and then shrunk with INTER_AREA -- noticeably smoother
# edges on something this small than drawing shapes natively at 1x,
# even with cv2's own anti-aliasing. PATCH_PAD gives the blurred
# shadow room to fall off before the patch itself runs out of canvas.
SS = 4
PATCH_PAD = 12
SHADOW_OFFSET = 3      # px, at 1x -- how far below the track the shadow sits
SHADOW_BLUR_SIGMA = 1.6  # px, at 1x
SHADOW_STRENGTH = 0.35   # how much the shadow darkens the frame under it, 0..1


class DayNightToggle:
    def __init__(self, smoothing_alpha: float = 0.3):
        self.position = None
        self.is_night = False
        self._amount = 0.0
        self._was_triggering = False
        self._smoother = EMASmoother(alpha=smoothing_alpha)

    @property
    def amount(self):
        """Current transition value: 0.0 = full day, 1.0 = full night."""
        return self._amount

    def update(self, head_anchor, hands):
        """Call once per frame, after hands are detected. `head_anchor`
        is a plain (x, y) point, or None on a frame where no face was
        detected (the switch just holds at its last smoothed spot).
        `hands` is the same per-frame Hand list everything else uses."""
        if head_anchor is not None:
            self.position = self._smoother.update(head_anchor)

        triggering = False
        if self.position is not None:
            for hand in hands:
                if not is_pinching(hand):
                    continue
                thumb = hand.point(THUMB_TIP)
                index = hand.point(INDEX_TIP)
                pinch_point = ((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2)
                if distance(pinch_point, self.position) <= GRAB_RADIUS + TRACK_W // 2:
                    triggering = True
                    break

        if triggering and not self._was_triggering:
            self.is_night = not self.is_night  # flip on the rising edge only
        self._was_triggering = triggering

        target = 1.0 if self.is_night else 0.0
        self._amount += (target - self._amount) * TRANSITION_RATE

    def _build_patch(self, t):
        """Render the switch at transition value `t` onto a small
        supersampled canvas and shrink it back down. Returns (rgb,
        opaque_mask, shadow_mask) at 1x scale -- rgb/opaque_mask are the
        crisp switch artwork to paste onto the frame; shadow_mask (0..255)
        says how much to darken the frame *underneath* it first, for a
        soft drop shadow.
        """
        w1, h1 = TRACK_W + PATCH_PAD * 2, TRACK_H + PATCH_PAD * 2
        w, h = w1 * SS, h1 * SS
        cx, cy = w // 2, h // 2
        tw, th = TRACK_W * SS, TRACK_H * SS
        x0, y0 = cx - tw // 2, cy - th // 2
        x1, y1 = cx + tw // 2, cy + th // 2

        # --- soft drop shadow: a pill shape offset down, heavily blurred ---
        shadow = np.zeros((h, w), dtype=np.uint8)
        off = SHADOW_OFFSET * SS
        # rectangle spans only the STRAIGHT MIDDLE section, between the
        # two cap centres -- not the full x0..x1 width. The end circles
        # (radius th/2, centred on those same cap points) are what
        # actually bulge past the rectangle to round the ends; centring
        # them at x0/x1 themselves (as an earlier version did) makes
        # them exactly tangent to the rectangle's own corners instead of
        # covering them, which is a circle drawn for nothing -- the
        # corners stay square. This was the real cause of the "sharp
        # edges" look, more than anti-aliasing.
        cv2.rectangle(shadow, (x0 + th // 2, y0 + off), (x1 - th // 2, y1 + off), 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(shadow, (x0 + th // 2, cy + off), th // 2, 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(shadow, (x1 - th // 2, cy + off), th // 2, 255, -1, lineType=cv2.LINE_AA)
        thumb_x = int(x0 + th // 2 + (tw - th) * t)
        cv2.circle(shadow, (thumb_x, cy + off), int(THUMB_R * SS * 1.1), 255, -1, lineType=cv2.LINE_AA)  # a little extra under the thumb
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=SHADOW_BLUR_SIGMA * SS)

        # --- track: a vertical gradient (lighter near the top, darker
        # near the bottom) so it reads as a rounded groove rather than a
        # flat rectangle -- fill the WHOLE row band first, then use the
        # pill-shaped mask below to keep only the parts that count, so
        # the rounded end-caps pick up the same gradient seamlessly ---
        day_track, night_track = (235, 206, 135), (70, 35, 20)
        base = tuple(day_track[i] * (1 - t) + night_track[i] * t for i in range(3))
        top = tuple(min(255.0, c * 1.18) for c in base)
        bottom = tuple(c * 0.82 for c in base)

        rows = np.arange(h)
        rt = np.clip((rows - y0) / max(1, (y1 - y0 - 1)), 0, 1)
        row_colors = (np.outer(1 - rt, top) + np.outer(rt, bottom)).astype(np.uint8)  # (h, 3)
        rgb = np.repeat(row_colors[:, None, :], w, axis=1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x0 + th // 2, y0), (x1 - th // 2, y1), 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(mask, (x0 + th // 2, cy), th // 2, 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(mask, (x1 - th // 2, cy), th // 2, 255, -1, lineType=cv2.LINE_AA)

        # --- small ring markers at each end of the track -- a fixed
        # detail (not part of the day/night crossfade) that gives the
        # track's two edges a bit more definition: warm at the sun end,
        # cool at the moon end, each one covered by the thumb once it
        # slides over that side, the way a real switch's end markers would be ---
        end_r = max(2, int(th * 0.3))
        end_thickness = max(1, SS)
        cv2.circle(rgb, (x0 + th // 2, cy), end_r, (200, 225, 255), end_thickness, lineType=cv2.LINE_AA)
        cv2.circle(rgb, (x1 - th // 2, cy), end_r, (255, 235, 210), end_thickness, lineType=cv2.LINE_AA)

        if t >= 0.5:
            for dx, dy in _STAR_OFFSETS:
                px, py = x0 + dx * SS, cy + dy * SS
                cv2.circle(rgb, (px, py), max(1, SS // 3), (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(mask, (px, py), max(1, SS // 3), 255, -1, lineType=cv2.LINE_AA)

        # --- thumb: glossy (a few nested circles, brighter toward the
        # top-left) rather than a flat disc, then the sun/moon glyph on top ---
        thumb_r = THUMB_R * SS
        for rr, col in ((thumb_r, (225, 225, 225)), (int(thumb_r * 0.8), (255, 255, 255))):
            cv2.circle(rgb, (thumb_x, cy), rr, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(mask, (thumb_x, cy), rr, 255, -1, lineType=cv2.LINE_AA)
        hl_r = max(1, int(thumb_r * 0.3))
        cv2.circle(rgb, (thumb_x - thumb_r // 3, cy - thumb_r // 3), hl_r, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        if t < 0.5:
            cv2.circle(rgb, (thumb_x, cy), max(1, int((THUMB_R - 3) * SS)), (0, 190, 255), -1, lineType=cv2.LINE_AA)
        else:
            gr = int((THUMB_R - 2) * SS)
            cv2.circle(rgb, (thumb_x - 2 * SS, cy), gr, (60, 25, 15), -1, lineType=cv2.LINE_AA)
            cv2.circle(rgb, (thumb_x + 1 * SS, cy), gr, (250, 250, 250), -1, lineType=cv2.LINE_AA)

        rgb = cv2.resize(rgb, (w1, h1), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w1, h1), interpolation=cv2.INTER_AREA)
        shadow = cv2.resize(shadow, (w1, h1), interpolation=cv2.INTER_AREA)
        return rgb, mask, shadow

    def draw(self, frame):
        """Small pill-style switch at the current anchor -- a shaded
        track that crossfades sky-blue (day) to indigo (night), a soft
        shadow underneath, and a glossy thumb that slides across as the
        transition advances and swaps a sun dot for a crescent, so the
        switch itself previews what a pinch will do before you commit."""
        if self.position is None:
            return frame

        rgb, mask, shadow = self._build_patch(self._amount)
        ph, pw = rgb.shape[:2]
        cx, cy = int(self.position[0]), int(self.position[1])
        x0, y0 = cx - pw // 2, cy - ph // 2
        x1, y1 = x0 + pw, y0 + ph

        fh, fw = frame.shape[:2]
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(fw, x1), min(fh, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return frame
        rx0, ry0 = cx0 - x0, cy0 - y0
        rx1, ry1 = rx0 + (cx1 - cx0), ry0 + (cy1 - cy0)

        region = frame[cy0:cy1, cx0:cx1].astype(np.float32)
        shadow_crop = shadow[ry0:ry1, rx0:rx1].astype(np.float32) / 255.0
        region *= (1.0 - shadow_crop[..., None] * SHADOW_STRENGTH)

        # alpha-blend with the mask's actual 0..255 gradient, not a
        # thresholded on/off cutoff -- a hard threshold here would throw
        # away exactly the edge smoothness the supersampling upstream
        # was for, and is what was making the pill look hard-edged
        # rather than properly rounded.
        alpha = (mask[ry0:ry1, rx0:rx1].astype(np.float32) / 255.0)[..., None]
        rgb_crop = rgb[ry0:ry1, rx0:rx1].astype(np.float32)
        region = region * (1.0 - alpha) + rgb_crop * alpha

        frame[cy0:cy1, cx0:cx1] = np.clip(region, 0, 255).astype(np.uint8)
        return frame


def tint_frame(frame, amount):
    """Blend the whole frame toward a dark tint as `amount` (0..1,
    DayNightToggle.amount) rises. Apply this to the raw camera frame
    BEFORE the light/moon is drawn on top, so the light stays vivid
    against a dimming backdrop instead of dimming along with it."""
    if amount <= 0.0:
        return frame
    strength = amount * MAX_DIM
    tint_layer = np.full_like(frame, TINT_COLOR)
    return cv2.addWeighted(frame, 1 - strength, tint_layer, strength, 0)