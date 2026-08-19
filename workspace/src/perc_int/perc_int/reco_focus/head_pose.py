# ===============================
# features/head_pose.py
# ===============================
import numpy as np
from .utils import angle_between

# Indices MediaPipe (simplified key points)
NOSE = 1
LEFT_EYE = 33
RIGHT_EYE = 263
CHIN = 199

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm
def wrap_angle(angle):
    """
    Ramène un angle en degrés dans l'intervalle [-90, +90]
    en conservant la signification physique.
    """
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle
class HeadPoseEstimator:    
    def estimate(self, landmarks):
        nose = np.array([landmarks[NOSE].x, landmarks[NOSE].y, landmarks[NOSE].z])
        left = np.array([landmarks[LEFT_EYE].x, landmarks[LEFT_EYE].y, landmarks[LEFT_EYE].z])
        right = np.array([landmarks[RIGHT_EYE].x, landmarks[RIGHT_EYE].y, landmarks[RIGHT_EYE].z])
        chin = np.array([landmarks[CHIN].x, landmarks[CHIN].y, landmarks[CHIN].z])

        face_x = normalize(right - left)
        face_y = normalize(chin - nose)
        forward_head = normalize(np.cross(face_x, face_y))
        
        return forward_head, face_x, face_y
    
    def estimate_relative_angles(self,
                                 landmarks,
                                 forward_ref,
                                 face_x_ref,
                                 face_y_ref):
        """
        Calcule yaw/pitch RELATIFS au calibrage
        """
        forward, _, _ = self.estimate(landmarks)

        # Yaw relatif (gauche / droite)
        yaw_rad = np.arctan2(
            np.dot(forward, face_x_ref),
            np.dot(forward, forward_ref)
        )

        # Pitch relatif (haut / bas)
        pitch_rad = np.arctan2(
            np.dot(forward, face_y_ref),
            np.dot(forward, forward_ref)
        )

        return np.degrees(yaw_rad), np.degrees(pitch_rad), forward
