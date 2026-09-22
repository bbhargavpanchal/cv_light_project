"""Light: tracks a position and renders as a simple circle.
Sun/moon rendering and glow effects extend this in a later phase."""

import cv2


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