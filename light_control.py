"""Grab-and-drag control for the light: clench a fist near it to pick
it up, move your hand to drag it anywhere in frame, open your hand (or
move it away) to let go. Replaces Phase 1's eye-tracked positioning —
the light now just stays wherever it was last dropped.

Light itself stays a dumb position+renderer (see light.py) and
occlusion.py doesn't care how position got set either, so this module
is the only thing that changed to swap the control scheme.
"""

from utils import distance
from hand_tracker import is_fist

# How far (px) a fist's palm centre can be from the light and still
# grab it. Generous on purpose -- roughly-near should be enough, no
# need for pixel-perfect aim. Tune this first if grabbing feels too
# fussy or too eager.
GRAB_RADIUS = 70


class LightDragController:
    def __init__(self, grab_radius=GRAB_RADIUS, min_curled=4):
        self.held = False
        self.grab_radius = grab_radius
        self.min_curled = min_curled

    def update(self, light, hands):
        """Call once per frame, after hands are detected and before
        the light is drawn. Mutates light.position when grabbing or
        dragging, and always syncs light's held state afterwards so it
        can animate the pop-on-grab / settle-on-release itself."""
        fist_hands = [h for h in hands if is_fist(h, self.min_curled)]

        if self.held:
            if not fist_hands or light.position is None:
                self.held = False  # hand opened, or vanished -> drop it here
            else:
                # No persistent hand IDs from MediaPipe, so "same hand" is
                # approximated as whichever fist is nearest the light's
                # current position. Self-reinforcing: the hand actually
                # dragging is always ~0px away, since we moved the light
                # there last frame, so it wins this every time in practice.
                nearest = min(fist_hands, key=lambda h: distance(h.palm_center(), light.position))
                light.update_position(nearest.palm_center())
        elif light.position is not None:
            for hand in fist_hands:
                if distance(hand.palm_center(), light.position) <= self.grab_radius + light.radius:
                    self.held = True
                    light.update_position(hand.palm_center())
                    break

        light.set_held(self.held)