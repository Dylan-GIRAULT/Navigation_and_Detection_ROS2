# ===============================
# features/mouth_metrics.py
# ===============================
from .utils import euclidean

MOUTH_IDX = [13, 14, 78, 308]

class MouthMetrics:
    def compute_MAR(self, landmarks):
        top = landmarks[MOUTH_IDX[0]]
        bottom = landmarks[MOUTH_IDX[1]]
        left = landmarks[MOUTH_IDX[2]]
        right = landmarks[MOUTH_IDX[3]]
        return euclidean((top.x, top.y), (bottom.x, bottom.y)) / \
               euclidean((left.x, left.y), (right.x, right.y))
