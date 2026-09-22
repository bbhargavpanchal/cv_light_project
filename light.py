"""Light: tracks a position and renders as a simple circle.
Sun/moon rendering and glow effects extend this in a later phase."""

import cv2
import numpy as np


class Light:
    def __init__(self, radius: int = 40, color=(0, 220, 255)):
        self.position = None
        self.radius = radius
        self.color = color  # BGR — warm yellow-orange, sun placeholder

    def update_position(self, position):
        if position is not None:
            self.position = position

    def draw(self, frame):
        if self.position is None:
            return frame
        x, y = int(self.position[0]), int(self.position[1])
        cv2.circle(frame, (x, y), self.radius, self.color, -1)
        return frame

    def bounding_box(self):
        """Axis-aligned bounding square around the light's circle, used
        as a cheap overlap test before any pixel-accurate mask work."""
        if self.position is None:
            return None
        x, y = self.position
        r = self.radius
        return x - r, y - r, x + r, y + r

    def mask(self, frame_shape):
        """Binary mask (uint8, 0/255) of the light's filled circle,
        same shape as the video frame it'll be composited against."""
        m = np.zeros(frame_shape[:2], dtype=np.uint8)
        if self.position is not None:
            x, y = int(self.position[0]), int(self.position[1])
            cv2.circle(m, (x, y), self.radius, 255, -1)
        return m