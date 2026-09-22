"""Face landmark tracking: exposes a smoothed eye-midpoint position.

Uses MediaPipe's Face Landmarker (Tasks API). Downloads the model
bundle to a local cache next to this file on first run (~4 MB, one-off).
"""

import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from utils import EMASmoother

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

# Standard MediaPipe 468-point face mesh topology: outer/inner eye corners
LEFT_EYE_IDXS = [33, 133]
RIGHT_EYE_IDXS = [362, 263]


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print("Downloading face landmarker model (one-off, ~4 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


class FaceTracker:
    def __init__(self, smoothing_alpha: float = 0.3):
        _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._smoother = EMASmoother(alpha=smoothing_alpha)
        self._timestamp_ms = 0

    def get_eye_midpoint(self, frame_rgb, frame_w, frame_h):
        """Returns a smoothed (x, y) pixel position for the eye midpoint.
        Falls back to the last known position if no face is detected
        this frame, or None if no face has ever been detected."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33  # VIDEO mode needs a monotonically increasing timestamp
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return self._smoother.value

        landmarks = result.face_landmarks[0]

        def avg_point(idxs):
            xs = [landmarks[i].x for i in idxs]
            ys = [landmarks[i].y for i in idxs]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        left = avg_point(LEFT_EYE_IDXS)
        right = avg_point(RIGHT_EYE_IDXS)
        mid_x = (left[0] + right[0]) / 2 * frame_w
        mid_y = (left[1] + right[1]) / 2 * frame_h

        return self._smoother.update((mid_x, mid_y))

    def close(self):
        self._landmarker.close()