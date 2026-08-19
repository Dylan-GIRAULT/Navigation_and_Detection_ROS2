# ===============================
# mediapipe_modules/face_landmarks.py
# ===============================
import mediapipe as mp

class FaceLandmarkEngine:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1
        )

    def process(self, frame_rgb):
        results = self.mp_face.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]