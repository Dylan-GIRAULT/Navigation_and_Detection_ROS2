# ===============================
# features/eye_metrics.py
# ===============================
from .utils import euclidean

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

class EyeMetrics:
    def compute_EAR(self, landmarks, idx):
        p = [landmarks[i] for i in idx]
        A = euclidean((p[1].x, p[1].y), (p[5].x, p[5].y))
        B = euclidean((p[2].x, p[2].y), (p[4].x, p[4].y))
        C = euclidean((p[0].x, p[0].y), (p[3].x, p[3].y))
        return (A + B) / (2.0 * C)

    def compute(self, landmarks):
        left = self.compute_EAR(landmarks, LEFT_EYE_IDX)
        right = self.compute_EAR(landmarks, RIGHT_EYE_IDX)
        return (left + right) / 2 if abs(left - right) < 0.05 else min(left, right)