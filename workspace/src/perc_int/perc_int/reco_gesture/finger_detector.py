import numpy as np
from .geom_utils import vec, length, angle_between
from .config import Config as cfg
    
def finger_fold_score(finger_landmarks, wrist_landmark):
    mcp, pip, dip, tip = finger_landmarks
    a1 = 180 - angle_between(vec(pip, mcp), vec(pip, dip))
    a2 = 180 - angle_between(vec(dip, pip), vec(dip, tip))
    a3 = 180 - angle_between(vec(mcp, wrist_landmark), vec(mcp, pip))
    
    score1 = min(a1 / 90.0, 1.0)
    score2 = min(a2 / 90.0, 1.0)
    score3 = min(a3 / 90.0, 1.0)
    return float((score1 + score2 + score3) / 3.0)

def is_finger_folded(finger_landmarks, wrist_landmark, threshold=0.6):
    return finger_fold_score(finger_landmarks, wrist_landmark) > threshold

def finger_pos(score):
    if score > cfg.T_FOLD_SCORE:
        return "FOLDED"
    elif score < cfg.T_EXTEND_SCORE:
        return "EXTENDED"
    else:
        return "AMBIG"