"""Grab-and-drag control for the light: clench a fist near it to pick
it up, move your hand to drag it anywhere in frame, open your hand (or
move it away) to let go. Replaces Phase 1's eye-tracked positioning —
the light now just stays wherever it was last dropped.

Also drives the light's depth-aware sizing: while a fist is actively
dragging it, the dragging hand's own apparent size (hand.scale(),
bigger when the hand is nearer the camera) is smoothed and pushed into
light.set_depth_scale() every frame, so moving your hand toward the
camera grows the light and pulling it back shrinks it again -- like
you're really holding it out at arm's length rather than sliding a
flat 2D icon around. It's frozen (not reset) the moment you let go, the
same "stays where you left it" behaviour position already has, since a
dropped light isn't attached to any hand's distance any more.

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

# hand.scale() (wrist-to-middle_MCP, px) that counts as "neutral"
# depth, i.e. where the light renders at its normal 1x size -- this is
# camera- and distance-dependent, so it's the first thing to tune if
# the light feels permanently oversized or undersized: hold your hand
# where you'd naturally drag the light and match this to what you see
# printed by the debug overlay, or just nudge it up/down by feel.
REFERENCE_HAND_SCALE = 95
MIN_DEPTH_SCALE = 0.5
MAX_DEPTH_SCALE = 1.8
DEPTH_SMOOTHING_RATE = 0.15  # per-frame ramp toward the target size, like toggle.py's fade


class LightDragController:
    def __init__(self, grab_radius=GRAB_RADIUS, min_curled=4):
        self.held = False
        self.grab_radius = grab_radius
        self.min_curled = min_curled
        self.depth_scale = 1.0

    def update(self, light, hands):
        """Call once per frame, after hands are detected and before
        the light is drawn. Mutates light.position when grabbing or
        dragging, and always syncs light's held state and depth scale
        afterwards so it can animate the pop-on-grab / settle-on-release
        and the depth-aware resize itself."""
        fist_hands = [h for h in hands if is_fist(h, self.min_curled)]
        dragging_hand = None

        if self.held:
            if not fist_hands or light.position is None:
                self.held = False  # hand opened, or vanished -> drop it here
            else:
                # No persistent hand IDs from MediaPipe, so "same hand" is
                # approximated as whichever fist is nearest the light's
                # current position. Self-reinforcing: the hand actually
                # dragging is always ~0px away, since we moved the light
                # there last frame, so it wins this every time in practice.
                dragging_hand = min(fist_hands, key=lambda h: distance(h.palm_center(), light.position))
                light.update_position(dragging_hand.palm_center())
        elif light.position is not None:
            for hand in fist_hands:
                if distance(hand.palm_center(), light.position) <= self.grab_radius + light.radius:
                    self.held = True
                    dragging_hand = hand
                    light.update_position(hand.palm_center())
                    break

        if dragging_hand is not None:
            target_scale = dragging_hand.scale() / REFERENCE_HAND_SCALE
            target_scale = max(MIN_DEPTH_SCALE, min(MAX_DEPTH_SCALE, target_scale))
            self.depth_scale += (target_scale - self.depth_scale) * DEPTH_SMOOTHING_RATE
        # else: frozen at whatever it last was -- not attached to a hand right now

        light.set_held(self.held)
        light.set_depth_scale(self.depth_scale)