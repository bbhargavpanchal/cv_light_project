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


class Spring:
    """A damped spring easing toward a target scalar.

    Unlike EMASmoother (which only eases monotonically toward a value),
    a spring naturally overshoots and settles -- the "pop then settle"
    feel of something being picked up or put down, rather than a flat
    linear ramp. Call set_target() when the target changes, update()
    once per frame to advance it.
    """

    def __init__(self, stiffness: float = 0.25, damping: float = 0.65, initial: float = 0.0):
        self.value = initial
        self.velocity = 0.0
        self.target = initial
        self.stiffness = stiffness
        self.damping = damping

    def set_target(self, target: float):
        self.target = target

    def update(self):
        force = (self.target - self.value) * self.stiffness
        self.velocity = (self.velocity + force) * self.damping
        self.value += self.velocity
        return self.value