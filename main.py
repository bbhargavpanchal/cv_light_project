"""Phase 2: webcam feed + eye-tracked light + two-hand tracking.

This phase is about confirming hand detection is reliable before wiring
up the eclipse effect. Both hands' skeletons are drawn, plus a simple
"hands detected" readout, so you can check tracking holds up as hands
move, overlap, or partially leave frame.
"""

import cv2

from face_tracker import FaceTracker
from hand_tracker import HandTracker, draw_hand_landmarks
from light import Light


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_tracker = FaceTracker(smoothing_alpha=0.3)
    hand_tracker = HandTracker(max_hands=2)
    light = Light()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror, feels natural — also matches
            h, w = frame.shape[:2]      # MediaPipe's handedness convention
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            eye_pos = face_tracker.get_eye_midpoint(frame_rgb, w, h)
            light.update_position(eye_pos)
            frame = light.draw(frame)

            hands = hand_tracker.get_hands(frame_rgb, w, h)
            frame = draw_hand_landmarks(frame, hands)

            cv2.putText(
                frame, f"Hands detected: {len(hands)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            cv2.imshow("CV Light Project - Phase 2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        face_tracker.close()
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()