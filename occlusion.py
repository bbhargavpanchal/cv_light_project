"""Eclipse compositing: per-hand occlusion of the light.

"Hand in front" vs "hand behind" is decided by hand ORIENTATION: show
the back of your hand to the camera and it eclipses the light; show
your palm and nothing happens, same as if the hand weren't there at
all. (Earlier versions used the hand's on-screen size as a stand-in for
depth -- flagged at the time as a placeholder for exactly this kind of
replacement. Orientation is the actual intended mechanic.)

Sequence per frame:
  1. Cheap bounding-box overlap test (hand rectangle vs light's
     bounding square) gates everything else — most frames, hands
     aren't anywhere near the light, so this skips the pixel work.
  2. If it overlaps and the back of the hand faces the camera:
     pixel-accurate eclipse — a convex hull over the hand's 21
     landmarks approximates its silhouette, intersected with the
     light's circular mask, and the hand's real camera pixels are
     restored into just that overlap region.
  3. If it overlaps but the palm faces the camera: nothing extra — the
     light was already drawn on top by default, so it just stays lit,
     same as if the hand were somewhere else on screen.
  4. No overlap at all: hand untouched, light sits wherever it is.

Each hand is evaluated independently, so two-hand eclipsing falls out
for free.
"""

import cv2
import numpy as np

from hand_tracker import palm_facing_camera


def _bbox_overlap(hand, light):
    lbox = light.bounding_box()
    if lbox is None:
        return False
    lx_min, ly_min, lx_max, ly_max = lbox
    hx_min, hy_min, hx_max, hy_max = hand.bounding_box()
    return not (
        hx_max < lx_min or hx_min > lx_max or
        hy_max < ly_min or hy_min > ly_max
    )


def _hand_mask(hand, frame_shape):
    pts = np.array(hand.landmarks_px, dtype=np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def apply_lighting(frame, light, hands, night_amount: float = 0.0):
    """Draws the light onto `frame` and applies per-hand eclipse
    occlusion on top of it. `night_amount` (0..1, from toggle.py) is
    passed straight through to light.draw() so the sun/moon crossfade
    keeps working the same whether or not a hand happens to be
    eclipsing it that frame.

    Returns (composited_frame, states) — states is a list of
    'front' | 'behind' | 'idle', one per hand, same order as `hands`.
    """
    if light.position is None:
        return frame, ["idle"] * len(hands)

    clean = frame.copy()  # real camera pixels, before the light is painted
    frame = light.draw(frame, night_amount=night_amount)  # default compositing == "behind": light on top

    states = []
    for hand in hands:
        if not _bbox_overlap(hand, light):
            states.append("idle")
            continue
        if palm_facing_camera(hand):
            states.append("behind")  # palm showing -- nothing changes
            continue

        hmask = _hand_mask(hand, frame.shape)
        lmask = light.mask(frame.shape)
        overlap = cv2.bitwise_and(hmask, lmask)
        frame[overlap > 0] = clean[overlap > 0]
        states.append("front")

    return frame, states