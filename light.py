"""Light: tracks a position and renders as a glowing orb with a smooth
radial gradient core (near-white hot centre fading to gold at the
edge), not a flat filled circle.

Motion is smoothed through an EMA so dragging looks fluid rather than
jittery — MediaPipe's raw landmark positions carry a small amount of
frame-to-frame noise that reads as a shaky/twitchy light otherwise.

The gradient and its blurred glow are both precomputed once in
__init__, since they only depend on radius/colour, not on where the
light currently is. draw() just crops and pastes those fixed tiles at
the current position each frame — no per-frame gradient or blur maths.

Full sun<->moon swapping (with the day/night toggle) is a separate,
later phase — this is a visual + motion polish pass on the one light
that exists right now.

set_held() drives a small "pop" when the light is grabbed and a
"settle" when it's released -- a Spring (not a plain ease) so it
overshoots slightly on the way up and dips slightly on the way down,
the way something physically picked up and set back down would.
"""

import cv2
import numpy as np

from utils import EMASmoother, Spring

GLOW_DOWNSCALE = 4
GLOW_BLUR_KERNEL = 9

# how much bigger / brighter the light gets at full "held" (spring
# value == 1.0) -- tune these first if the pop feels too subtle or
# too much
HELD_SCALE_BOOST = 0.3       # +30% size at full hold
HELD_BRIGHTNESS_BOOST = 0.35  # +35% brightness at full hold
HELD_SPRING_STIFFNESS = 0.25
HELD_SPRING_DAMPING = 0.65


def _radial_gradient_tile(diameter, radius, center_color, edge_color):
    """Square BGR tile with a smooth radial gradient circle -- center_color
    at the middle, fading to edge_color at `radius` -- and zero outside
    the circle. Returns (tile, boolean_mask_of_the_circle)."""
    c = diameter // 2
    yy, xx = np.mgrid[0:diameter, 0:diameter]
    dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    t = np.clip(dist / radius, 0, 1)
    t = t * t * (3 - 2 * t)  # smoothstep -- nicer falloff than linear
    tile = np.zeros((diameter, diameter, 3), dtype=np.float32)
    for i in range(3):
        tile[..., i] = center_color[i] * (1 - t) + edge_color[i] * t
    mask = dist <= radius
    tile[~mask] = 0
    return tile.astype(np.uint8), mask


class Light:
    def __init__(self, radius: int = 40, center_color=(235, 255, 255),
                 edge_color=(0, 170, 255), smoothing_alpha: float = 0.5):
        self.position = None
        self.radius = radius
        self.center_color = center_color
        self.edge_color = edge_color
        self.glow_radius = radius * 4
        self._smoother = EMASmoother(alpha=smoothing_alpha)
        self._held_spring = Spring(stiffness=HELD_SPRING_STIFFNESS, damping=HELD_SPRING_DAMPING)

        # crisp core: gradient tile exactly the size of the circle
        core_d = radius * 2
        self._core_tile, self._core_mask = _radial_gradient_tile(
            core_d, radius, center_color, edge_color)

        # soft glow: same gradient, drawn into a much bigger tile, then
        # bloomed via downsample->blur->upsample (cheap, same technique
        # verified in the previous pass -- ~0.5ms vs ~18ms for a naive
        # full-res blur) -- precomputed once, not per frame
        glow_d = self.glow_radius * 2
        glow_src, _ = _radial_gradient_tile(glow_d, radius, center_color, edge_color)
        small_d = max(1, glow_d // GLOW_DOWNSCALE)
        small = cv2.resize(glow_src, (small_d, small_d), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (GLOW_BLUR_KERNEL, GLOW_BLUR_KERNEL), 0)
        self._glow_tile = cv2.resize(small, (glow_d, glow_d), interpolation=cv2.INTER_LINEAR)

    def update_position(self, position):
        if position is not None:
            # smoothed, not raw -- kills per-frame landmark jitter while
            # still tracking the hand closely (see smoothing_alpha)
            self.position = self._smoother.update(position)

    def set_held(self, is_held: bool):
        """Call once per frame with the current grabbed/not-grabbed
        state (see light_control.LightDragController). Drives the pop
        on grab / settle on release; draw() reads the spring's current
        value each frame."""
        self._held_spring.set_target(1.0 if is_held else 0.0)

    def _paste(self, frame, tile, tile_center_offset, blend_mask=None):
        """Crop `tile` (a square, precomputed at construction time) to
        whatever part of it lands inside `frame` given the light's
        current position, then either add it (glow) or copy it through
        `blend_mask` (crisp core) onto frame in place."""
        x, y = int(self.position[0]), int(self.position[1])
        h, w = frame.shape[:2]
        r = tile_center_offset
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(w, x + r), min(h, y + r)
        if x1 <= x0 or y1 <= y0:
            return
        tx0, ty0 = x0 - (x - r), y0 - (y - r)
        tile_crop = tile[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)]
        roi = frame[y0:y1, x0:x1]
        if blend_mask is None:
            frame[y0:y1, x0:x1] = cv2.add(roi, tile_crop)
        else:
            mask_crop = blend_mask[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)]
            roi[mask_crop] = tile_crop[mask_crop]
            frame[y0:y1, x0:x1] = roi

    def draw(self, frame):
        if self.position is None:
            return frame

        amount = self._held_spring.update()
        amount = max(-0.5, min(1.5, amount))  # guard against pathological rapid grab/release cycling

        if abs(amount) < 0.01:
            # at rest: paste the precomputed tiles directly, no per-frame
            # resize/brighten cost -- this is the common case
            self._paste(frame, self._glow_tile, self.glow_radius)
            self._paste(frame, self._core_tile, self.radius, blend_mask=self._core_mask)
            return frame

        scale = 1.0 + amount * HELD_SCALE_BOOST
        brighten = 1.0 + amount * HELD_BRIGHTNESS_BOOST

        glow_r = max(1, int(self.glow_radius * scale))
        core_r = max(1, int(self.radius * scale))
        glow_tile = self._resized_bright(self._glow_tile, glow_r * 2, brighten)
        core_tile = self._resized_bright(self._core_tile, core_r * 2, brighten)
        core_mask = cv2.resize(
            (self._core_mask.astype(np.uint8) * 255), (core_r * 2, core_r * 2),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

        self._paste(frame, glow_tile, glow_r)
        self._paste(frame, core_tile, core_r, blend_mask=core_mask)
        return frame

    @staticmethod
    def _resized_bright(tile, new_size, brighten):
        resized = cv2.resize(tile, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        if brighten != 1.0:
            resized = np.clip(resized.astype(np.float32) * brighten, 0, 255).astype(np.uint8)
        return resized

    def bounding_box(self):
        """Axis-aligned bounding square around the light's CORE circle
        (not the soft outer glow) — this is what occlusion.py eclipses
        against, unchanged in meaning from Phase 3."""
        if self.position is None:
            return None
        x, y = self.position
        r = self.radius
        return x - r, y - r, x + r, y + r

    def mask(self, frame_shape):
        """Binary mask (uint8, 0/255) of the light's CORE circle, same
        shape as the video frame it'll be composited against."""
        m = np.zeros(frame_shape[:2], dtype=np.uint8)
        if self.position is not None:
            x, y = int(self.position[0]), int(self.position[1])
            cv2.circle(m, (x, y), self.radius, 255, -1)
        return m