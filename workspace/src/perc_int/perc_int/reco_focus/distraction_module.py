# ===============================
# inference/distraction_module.py
# ===============================
import time
import numpy as np
from .settings import INITIAL_SCORE, HEAD_YAW_FORWARD, HEAD_YAW_DISTRACT, HEAD_PITCH_DOWN, HEAD_YAW_FORWARD_SCORE, HEAD_YAW_DISTRACT_SCORE, PENALTY_PER_SEC, STRONG_MULTIPLIER, RECOVERY_PER_SEC

class DistractionModule:
    def __init__(self, 
                 initial_score=INITIAL_SCORE,
                 yaw_forward_score=HEAD_YAW_FORWARD_SCORE,
                 yaw_distract_score=HEAD_YAW_DISTRACT_SCORE,
                 yaw_forward_thresh=HEAD_YAW_FORWARD,
                 yaw_distract_thresh=HEAD_YAW_DISTRACT,
                 pitch_down_thresh=HEAD_PITCH_DOWN,
                 penalty_per_sec=PENALTY_PER_SEC,
                 strong_multiplier=STRONG_MULTIPLIER,
                 recovery_per_sec=RECOVERY_PER_SEC):
        self.initial_score = initial_score
        self.yaw_forward_score = yaw_forward_score
        self.yaw_distract_score = yaw_distract_score

        self.yaw_forward_thresh = yaw_forward_thresh
        self.yaw_distract_thresh = yaw_distract_thresh
        self.pitch_down_thresh = pitch_down_thresh

        self.penalty_per_sec = penalty_per_sec
        self.strong_multiplier = strong_multiplier
        self.recovery_per_sec = recovery_per_sec

        self.last_time = None
        self.penalty = 0.0
        self.max_penalty = initial_score

    def classify(self, yaw, pitch):
        """
        yaw, pitch : angles relatifs (en degrés)
        retourne un score ∈ [0, 1]
        """
        
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)

        if abs_yaw < self.yaw_forward_thresh and abs_pitch < self.pitch_down_thresh:
            base_score = self.initial_score
            is_distracted = False

        elif abs_yaw < self.yaw_distract_thresh:
            base_score = self.yaw_forward_score
            is_distracted = True

        else:
            base_score = self.yaw_distract_score
            is_distracted = True

        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return base_score

        dt = now - self.last_time
        self.last_time = now

        if is_distracted:
            factor = 1.0
            if abs_yaw > self.yaw_distract_thresh:
                factor = self.strong_multiplier

            self.penalty += self.penalty_per_sec * factor * dt
        else:
            # récupération progressive
            self.penalty -= self.recovery_per_sec * dt

        self.penalty = np.clip(self.penalty, 0.0, self.max_penalty)


        final_score = base_score - self.penalty
        return max(0.0, final_score)