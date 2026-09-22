import numpy as np
import cv2

SHADOW_BLUR_KSIZE = 21
HAND_DILATE_PX = 8


def build_hand_mask(shape, hands_data, dilate_px=HAND_DILATE_PX):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for hand in hands_data:
        pts = np.array(hand["points"], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0, lineType=cv2.LINE_AA)

    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def apply_eclipse(glow_layer, intensity_mask, hands_data, blur_ksize=SHADOW_BLUR_KSIZE):
    """Occlude the light glow wherever a hand overlaps it, with a soft blurred edge."""
    if not hands_data:
        return glow_layer, intensity_mask, None

    shape = intensity_mask.shape
    hand_mask = build_hand_mask(shape, hands_data)

    k = blur_ksize | 1
    soft_hand_mask = cv2.GaussianBlur(hand_mask, (k, k), 0)
    soft_hand_mask = np.clip(soft_hand_mask, 0.0, 1.0)

    occlusion = 1.0 - soft_hand_mask
    eclipsed_glow = glow_layer * occlusion[..., None]
    eclipsed_intensity = intensity_mask * occlusion

    return eclipsed_glow, eclipsed_intensity, soft_hand_mask
