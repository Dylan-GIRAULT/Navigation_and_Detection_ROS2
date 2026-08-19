"""
Simple demo that runs MediaPipe Hands, processes each detected hand through HandProcessor
and draws overlays on the frame.
"""
import cv2
import mediapipe as mp
from hand_processor import HandProcessor


def draw_label(frame, label, conf, x=10, y=30):
    cv2.putText(frame, f"{label} ({conf:.2f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    processor = HandProcessor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame_rgb)
        label = 'NO_HAND'
        conf = 0.0
        if res.multi_hand_landmarks:
            # process only first detected hand for simplicity
            hand_lms = res.multi_hand_landmarks[0]
            # convert to list of (x,y,z) in image coords normalized to [-0.5..0.5] around center using image size
            h, w, _ = frame.shape
            landmarks = []
            for lm in hand_lms.landmark:
                # normalized coordinates from mediapipe
                # we keep them as (x*w, y*h, z*w) relative units to preserve aspect
                landmarks.append((lm.x * w, lm.y * h, lm.z * w))

            # detect handedness if available
            handedness = None
            if res.multi_handedness:
                handedness = res.multi_handedness[0].classification[0].label
            label, conf, meta = processor.update(landmarks, handedness)
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        draw_label(frame, label, conf)
        cv2.imshow('gestures', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
