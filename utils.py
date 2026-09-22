"""Shared helpers: smoothing, interpolation, distance."""

import math


class EMASmoother:
    """Exponential moving average smoother for 2D points.

    Blends each new reading with the previous smoothed value to
    reduce jitter in tracked landmark positions.
    """

    def __init__(self, alpha: float = 0.3):
        # alpha closer to 1.0 = more responsive but jittery
        # alpha closer to 0.0 = smoother but laggier
        self.alpha = alpha
        self.value = None

    def update(self, new_point):
        x, y = new_point
        if self.value is None:
            self.value = (x, y)
        else:
            px, py = self.value
            self.value = (
                self.alpha * x + (1 - self.alpha) * px,
                self.alpha * y + (1 - self.alpha) * py,
            )
        return self.value

    def reset(self):
        self.value = None


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def lerp(a, b, t):
    return a + (b - a) * t