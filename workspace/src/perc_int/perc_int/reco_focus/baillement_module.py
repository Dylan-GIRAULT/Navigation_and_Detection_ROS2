from .settings import INITIAL_SCORE, MAR_YAWN_THRESH, MAR_YAWN_DUR_THRESH_SEC, PENALTY_PER_SEC, RECOVERY_PER_SEC, STRONG_MULTIPLIER
import time

class YawnDetector:
    def __init__(self,
                 mar_threshold=MAR_YAWN_THRESH,
                 min_duration_sec=MAR_YAWN_DUR_THRESH_SEC,
                 penalty_per_sec=PENALTY_PER_SEC,
                 multiplier=STRONG_MULTIPLIER,
                 max_penalty=INITIAL_SCORE,
                 recovery_per_sec=RECOVERY_PER_SEC
                 ):
        self.open_start_time = None
        self.yawn_count = 0

        self.mar_threshold = mar_threshold
        self.min_duration_sec = min_duration_sec
        
        self.yawn_detected = False      # baillement en cours
        self.multiplier_active = False  # activé dès le 1er baillement

        self.penalty_per_sec = penalty_per_sec
        self.multiplier = multiplier
        self.max_penalty = max_penalty

        self.recovery_per_sec = recovery_per_sec

        self.open_start_time = None
        self.total_penalty = 0.0

        self.last_time = None

    def update(self, mar):
        """
        Appelé à chaque frame
        Retourne la pénalité bouche à soustraire au score global
        """
        now = time.time()

        if self.last_time is None:
            self.last_time = now
            return self.total_penalty

        dt = now - self.last_time
        self.last_time = now

        if mar > self.mar_threshold:
            if self.open_start_time is None:
                self.open_start_time = now

            open_duration = now - self.open_start_time

            # Détection du baillement
            if (not self.yawn_detected and
                open_duration >= self.min_duration_sec):

                self.yawn_detected = True
                self.multiplier_active = True

            # Pénalité continue tant que bouche ouverte
            if self.yawn_detected:
                penalty = self.penalty_per_sec * dt
                if self.multiplier_active:
                    penalty *= self.multiplier

                self.total_penalty += penalty

        else:
            # bouche fermée → reset état
            self.open_start_time = None
            self.yawn_detected = False

            # récupération progressive
            self.total_penalty -= self.recovery_per_sec * dt
            self.total_penalty = max(self.total_penalty, 0.0)

        # Clamp de sécurité
        self.total_penalty = min(self.total_penalty, self.max_penalty)

        return self.total_penalty