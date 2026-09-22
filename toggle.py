import time

from utils import distance, Smoother

PINCH_THRESHOLD_PX = 40
PROXIMITY_RADIUS_PX = 90
TOGGLE_COOLDOWN_S = 0.6
HEAD_OFFSET = (0, -160)  # anchor the light above the head, in pixels


class LightToggle:
    """Anchors the light to the head and toggles sun/moon on a pinch made near the light."""

    def __init__(self, light, pinch_threshold=PINCH_THRESHOLD_PX,
                 proximity_radius=PROXIMITY_RADIUS_PX, cooldown=TOGGLE_COOLDOWN_S):
        self.light = light
        self.pinch_threshold = pinch_threshold
        self.proximity_radius = proximity_radius
        self.cooldown = cooldown
        self._position_smoother = Smoother(alpha=0.25)
        self._last_toggle_time = 0.0
        self._armed = True  # prevents repeated toggles while a pinch is held

    def update(self, face_data, hands_data):
        now = time.perf_counter()

        if face_data is not None:
            eye_mid = face_data["eye_mid"]
            target = (eye_mid[0] + HEAD_OFFSET[0], eye_mid[1] + HEAD_OFFSET[1])
            position = self._position_smoother.update(target)
            self.light.set_position(position)
        else:
            self._position_smoother.reset()

        pinching_near_light = False
        for hand in hands_data:
            if hand["pinch_dist"] > self.pinch_threshold:
                continue
            if distance(hand["pinch_point"], self.light.position) <= self.proximity_radius:
                pinching_near_light = True
                break

        if pinching_near_light:
            if self._armed and (now - self._last_toggle_time) >= self.cooldown:
                self.light.toggle()
                self._last_toggle_time = now
                self._armed = False
        else:
            self._armed = True

        return pinching_near_light
