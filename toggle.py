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

TOGGLE_RADIUS = 22         # on-screen size of the switch icon, px
GRAB_RADIUS = 50           # extra reach beyond the icon's own edge that still counts as "near", px -- tune here first
TRANSITION_RATE = 0.06     # per-frame ramp toward the target -- smaller = slower fade
MAX_DIM = 0.45             # night never dims past this fraction -- "little bit darker", not pitch black
TINT_COLOR = (40, 20, 10)  # BGR, deep navy -- the colour the frame fades toward at night


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
                if distance(pinch_point, self.position) <= GRAB_RADIUS + TOGGLE_RADIUS:
                    triggering = True
                    break

        if triggering and not self._was_triggering:
            self.is_night = not self.is_night  # flip on the rising edge only
        self._was_triggering = triggering

        target = 1.0 if self.is_night else 0.0
        self._amount += (target - self._amount) * TRANSITION_RATE

    def draw(self, frame):
        """Small switch icon at the current anchor. Crossfades pale-day
        to indigo-night and swaps a filled dot for a crescent, so the
        switch itself previews what a pinch will do before you commit."""
        if self.position is None:
            return frame

        x, y = int(self.position[0]), int(self.position[1])
        t = self._amount

        day_color = (200, 230, 255)
        night_color = (110, 60, 30)
        color = tuple(int(day_color[i] * (1 - t) + night_color[i] * t) for i in range(3))

        cv2.circle(frame, (x, y), TOGGLE_RADIUS, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), TOGGLE_RADIUS, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        glyph_r = max(1, TOGGLE_RADIUS // 2)
        if t < 0.5:
            # sun: simple filled dot
            cv2.circle(frame, (x, y), glyph_r, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        else:
            # moon: a white dot with a same-colour-as-switch dot biting
            # into it, leaving a crescent
            cv2.circle(frame, (x - glyph_r // 2, y), glyph_r, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x + glyph_r // 3, y), glyph_r, color, -1, lineType=cv2.LINE_AA)

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