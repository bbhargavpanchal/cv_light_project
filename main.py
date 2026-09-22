"""Grab-and-drag light + two-hand tracking + eclipse occlusion.

Eye-tracked positioning is gone: the light starts at frame centre, and
you move it by clenching a fist near it (grab), moving your hand
(drag), and opening your hand or pulling away (drop). Eye/face
tracking returns in Phase 4, repurposed for the head-anchored toggle
rather than the light.

The on-screen readout shows hands detected, whether the light is
currently held, and each hand's eclipse state (front / behind / idle)
so the two systems' interaction is easy to see while testing.
"""

import cv2

from hand_tracker import HandTracker, draw_hand_landmarks
from light import Light
from light_control import LightDragController
import occlusion


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    hand_tracker = HandTracker(max_hands=2)
    light = Light()
    drag = LightDragController()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror, feels natural — also matches
            h, w = frame.shape[:2]      # MediaPipe's handedness convention

            if light.position is None:
                light.update_position((w // 2, h // 2))  # visible + grabbable from frame one

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = hand_tracker.get_hands(frame_rgb, w, h)

            drag.update(light, hands)

            frame, states = occlusion.apply_lighting(frame, light, hands)
            frame = draw_hand_landmarks(frame, hands, states=states)

            cv2.putText(
                frame,
                f"Hands detected: {len(hands)}  |  Light: {'held' if drag.held else 'free'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            cv2.imshow("CV Light Project", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()