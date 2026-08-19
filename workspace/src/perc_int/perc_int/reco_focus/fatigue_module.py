import time

from .settings import CLOSURE_DURATION_SEC, EAR_THRESHOLD, PENALTY_VALUE, RECOVERY_PER_SEC


class FatigueModule:
    def __init__(self,
                 ear_threshold=EAR_THRESHOLD,
                 closure_duration_sec=CLOSURE_DURATION_SEC,
                 penalty_value=PENALTY_VALUE,
                 recovery_per_sec=RECOVERY_PER_SEC):

        self.ear_threshold = ear_threshold
        self.closure_duration_sec = closure_duration_sec
        self.penalty_value = penalty_value
        self.recovery_per_sec = recovery_per_sec
        self.last_penalty_step = 0

        self.max_penalty = 1.0
        self.closed_start_time = None
        self.penalty = 0.0
        self.last_time = None
        self.penalty_applied = False

    def update_and_score(self, ear):
        """
        Appelé à chaque frame
        Ajoute +0.4 de pénalité toutes les 2s consécutives yeux fermés
        """
        now = time.time()

        if self.last_time is None:
            self.last_time = now
            return self.penalty

        dt = now - self.last_time
        self.last_time = now

        if ear < self.ear_threshold:
            # ----- YEUX FERMÉS -----
            if self.closed_start_time is None:
                self.closed_start_time = now
                self.last_penalty_step = 0  # reset palier

            closed_duration = now - self.closed_start_time

            # nombre de paliers de 2s atteints
            step = int(closed_duration // self.closure_duration_sec)

            if step > self.last_penalty_step:
                # ajouter la pénalité pour chaque nouveau palier atteint
                added_steps = step - self.last_penalty_step
                self.penalty += added_steps * self.penalty_value
                self.last_penalty_step = step

        else:
            # ----- YEUX OUVERTS → RECOVERY -----
            self.closed_start_time = None
            self.last_penalty_step = 0

            self.penalty -= self.recovery_per_sec * dt
            self.penalty = max(self.penalty, 0.0)

        self.penalty = min(self.penalty, self.max_penalty)
        return self.penalty

