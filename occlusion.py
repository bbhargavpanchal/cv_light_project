"""Eclipse compositing: per-hand occlusion of the light.

Design note (the one real judgment call in this phase): the light has
no true depth of its own — it's a 2D position tracking your eyes — so
"hand in front" vs "hand behind" needs some stand-in for depth. Raw
MediaPipe landmark z is relative and noisy frame-to-frame, so instead
this uses the hand's on-screen SIZE as the depth proxy: a hand held
close to the camera fills more of the frame, so a bounding-box diagonal
past NEAR_THRESHOLD_RATIO of the frame height counts as "close enough
to eclipse". Push your hand toward the camera to trigger it, pull it
back to a normal gesture distance to let the light stay on top. This
also has a bonus: it's the same size signal Phase 6's depth-aware
scaling was already going to need, so there's nothing to throw away
later.

Sequence per frame:
  1. Cheap bounding-box overlap test (hand rectangle vs light's
     bounding square) gates everything else — most frames, hands
     aren't anywhere near the light, so this skips the pixel work.
  2. If it overlaps and the hand is "close": pixel-accurate eclipse —
     a convex hull over the hand's 21 landmarks approximates its
     silhouette, intersected with the light's circular mask, and the
     hand's real camera pixels are restored into just that overlap
     region.
  3. If it overlaps but the hand is "far": nothing extra — the light
     was already drawn on top by default, so the hand simply appears
     to sit behind it.
  4. No overlap at all: hand untouched, light sits wherever it is.

Each hand is evaluated independently, so two-hand eclipsing falls out
for free.
"""

import cv2
import numpy as np

# Fraction of frame height a hand's bounding-box diagonal must exceed to
# count as "close to the camera" (front / eclipsing). If eclipse
# triggers too eagerly, raise this; if it feels reluctant, lower it.
NEAR_THRESHOLD_RATIO = 0.35


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


def _hand_is_near(hand, frame_h):
    x_min, y_min, x_max, y_max = hand.bounding_box()
    diag = np.hypot(x_max - x_min, y_max - y_min)
    return diag > NEAR_THRESHOLD_RATIO * frame_h


def _hand_mask(hand, frame_shape):
    pts = np.array(hand.landmarks_px, dtype=np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def apply_lighting(frame, light, hands):
    """Draws the light onto `frame` and applies per-hand eclipse
    occlusion on top of it.

    Returns (composited_frame, states) — states is a list of
    'front' | 'behind' | 'idle', one per hand, same order as `hands`.
    """
    if light.position is None:
        return frame, ["idle"] * len(hands)

    clean = frame.copy()       # real camera pixels, before the light is painted
    frame = light.draw(frame)  # default compositing == "behind": light on top

    states = []
    for hand in hands:
        if not _bbox_overlap(hand, light):
            states.append("idle")
            continue
        if not _hand_is_near(hand, frame.shape[0]):
            states.append("behind")
            continue

        hmask = _hand_mask(hand, frame.shape)
        lmask = light.mask(frame.shape)
        overlap = cv2.bitwise_and(hmask, lmask)
        frame[overlap > 0] = clean[overlap > 0]
        states.append("front")

    return frame, states