import numpy as np
import time

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v
    return v / n

class HeadCalibrationManager:
    def __init__(self,
                 duration_sec=2.5,
                 max_angle_std_deg=5.0):
        """
        duration_sec : durée de calibrage
        max_angle_std_deg : stabilité requise (écart-type angulaire)
        """
        self.duration_sec = duration_sec
        self.max_angle_std = np.radians(max_angle_std_deg)

        self.samples = []
        self.start_time = None

        self.forward_ref = None
        self.face_x_ref = None
        self.face_y_ref = None
        self.calibrated = False

    def is_calibrated(self):
        return self.calibrated

    def reset(self):
        self.samples = []
        self.start_time = None
        self.forward_ref = None
        self.face_x_ref = None
        self.face_y_ref = None
        self.calibrated = False

    def update(self, forward):
        """
        Appelé à chaque frame tant que non calibré
        """
        if self.calibrated:
            return

        now = time.time()

        if self.start_time is None:
            self.start_time = now

        self.forwards.append(forward)

        elapsed = now - self.start_time
        if elapsed < self.duration_sec:
            return

        # Vérifier stabilité
        if not self._is_stable():
            # recommence le calibrage
            self.reset()
            return

        # Calcul forward de référence
        mean_forward = np.mean(self.forwards, axis=0)
        self.forward_ref = mean_forward / np.linalg.norm(mean_forward)
        self.calibrated = True

    def get_reference(self):
        return self.forward_ref, self.face_x_ref, self.face_y_ref

    def _is_stable(self):
        """
        Vérifie que les forwards ne varient pas trop
        """
        if len(self.samples) < 2:
            return False

        ref = self.samples[0][0]  # forward du premier échantillon
        angles = []

        for s in self.samples[1:]:
            f = s[0]
            dot = np.clip(np.dot(ref, f), -1.0, 1.0)
            angle = np.arccos(dot)
            angles.append(angle)

        return np.std(angles) < self.max_angle_std
    

    def update(self, forward, face_x, face_y):
        """
        Appelé à chaque frame tant que non calibré.
        forward, face_x, face_y : vecteurs normalisés instantanés
        """
        if self.calibrated:
            return

        now = time.time()
        if self.start_time is None:
            self.start_time = now

        self.samples.append((forward, face_x, face_y))
        elapsed = now - self.start_time

        if elapsed < self.duration_sec:
            return

        # Vérifier stabilité
        if not self._is_stable():
            self.reset()
            return

        # Calculer les vecteurs de référence moyens
        forwards = np.array([s[0] for s in self.samples])
        face_xs = np.array([s[1] for s in self.samples])
        face_ys = np.array([s[2] for s in self.samples])

        self.forward_ref = normalize(np.mean(forwards, axis=0))
        self.face_x_ref = normalize(np.mean(face_xs, axis=0))
        self.face_y_ref = normalize(np.mean(face_ys, axis=0))

        self.calibrated = True
        print("Head pose calibration done")

