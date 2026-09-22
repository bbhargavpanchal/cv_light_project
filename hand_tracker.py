"""Hand landmark tracking: exposes up to two hands' landmarks per frame.

Uses MediaPipe's Hand Landmarker (Tasks API). Downloads the model
bundle to a local cache next to this file on first run (~8 MB, one-off).
"""

import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# Standard MediaPipe 21-point hand topology — named fingertip indices,
# used later for pinch detection (toggle.py) and occlusion (occlusion.py)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
NUM_LANDMARKS = 21

# Bone connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print("Downloading hand landmarker model (one-off, ~8 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


class Hand:
    """One detected hand: pixel-space landmarks + handedness label."""

    __slots__ = ("landmarks_px", "landmarks_norm", "handedness", "score")

    def __init__(self, landmarks_px, landmarks_norm, handedness, score):
        self.landmarks_px = landmarks_px      # 21x (x, y) in pixel coords
        self.landmarks_norm = landmarks_norm  # 21x (x, y, z) normalized, z relative depth
        self.handedness = handedness          # "Left" or "Right", from the camera's view
        self.score = score

    def point(self, idx):
        return self.landmarks_px[idx]

    def bounding_box(self):
        """(x_min, y_min, x_max, y_max) in pixel coords, used later for
        depth-aware scaling and coarse occlusion checks."""
        xs = [p[0] for p in self.landmarks_px]
        ys = [p[1] for p in self.landmarks_px]
        return min(xs), min(ys), max(xs), max(ys)


class HandTracker:
    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5):
        _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def get_hands(self, frame_rgb, frame_w, frame_h):
        """Returns a list of Hand objects, one per detected hand (up to max_hands)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33  # VIDEO mode needs a monotonically increasing timestamp
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            px = [(lm.x * frame_w, lm.y * frame_h) for lm in hand_landmarks]
            norm = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

            label, score = "Unknown", 0.0
            if i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label = cat.category_name or "Unknown"
                score = cat.score or 0.0

            hands.append(Hand(px, norm, label, score))
        return hands

    def close(self):
        self._landmarker.close()


def draw_hand_landmarks(frame, hands, point_color=(0, 255, 0), line_color=(255, 255, 255)):
    """Debug-draw skeleton for a list of Hand objects onto frame in place."""
    import cv2

    for hand in hands:
        pts = hand.landmarks_px
        for a, b in HAND_CONNECTIONS:
            pa = (int(pts[a][0]), int(pts[a][1]))
            pb = (int(pts[b][0]), int(pts[b][1]))
            cv2.line(frame, pa, pb, line_color, 2)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 4, point_color, -1)
        # label handedness near the wrist
        wx, wy = pts[WRIST]
        cv2.putText(
            frame, f"{hand.handedness} ({hand.score:.2f})",
            (int(wx) - 20, int(wy) + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2,
        )
    return frame