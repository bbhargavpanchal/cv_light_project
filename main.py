"""Grab-and-drag light + two-hand tracking + eclipse occlusion + a
head-anchored day/night toggle.

Eye-tracked positioning is gone: the light starts at frame centre, and
you move it by clenching a fist near it (grab), moving your hand
(drag), and opening your hand or pulling away (drop). Face tracking is
back, but repurposed -- it only anchors the little day/night switch
above your head, not the light itself. Pinch near that switch to flip
sun<->moon; pinching anywhere else does nothing, same "must actually
be near it" rule the light's own grab already follows.

The on-screen readout shows hands detected, whether the light is
currently held, and each hand's eclipse state (front / behind / idle)
so all three systems' interaction is easy to see while testing.
"""

import cv2

from hand_tracker import HandTracker, draw_hand_landmarks
from face_tracker import FaceTracker
from light import Light
from light_control import LightDragController
from toggle import DayNightToggle, tint_frame
import occlusion

# How far above the tracked eye-midpoint the toggle switch sits, px.
HEAD_ANCHOR_OFFSET_Y = 90


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    hand_tracker = HandTracker(max_hands=2)
    face_tracker = FaceTracker()
    light = Light()
    drag = LightDragController()
    toggle = DayNightToggle()

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

            eye_mid = face_tracker.get_eye_midpoint(frame_rgb, w, h)
            head_anchor = None
            if eye_mid is not None:
                head_anchor = (eye_mid[0], eye_mid[1] - HEAD_ANCHOR_OFFSET_Y)
            toggle.update(head_anchor, hands)

            drag.update(light, hands)

            frame = tint_frame(frame, toggle.amount)
            frame, states = occlusion.apply_lighting(frame, light, hands, night_amount=toggle.amount)
            frame = draw_hand_landmarks(frame, hands, states=states)
            frame = toggle.draw(frame)

            cv2.putText(
                frame,
                f"Hands: {len(hands)}  |  Light: {'held' if drag.held else 'free'}  |  "
                f"{'Night' if toggle.is_night else 'Day'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            cv2.imshow("CV Light Project", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hand_tracker.close()
        face_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()