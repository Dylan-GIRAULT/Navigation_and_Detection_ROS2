import numpy as np
from .geom_utils import vec, cross, normalize, angle_between
from .config import Config as cfg


def compute_palm_normal(landmarks):
    
    p_index = landmarks[5]
    p_pinky = landmarks[17]
    p_middle = landmarks[9]
    p_wrist = landmarks[0]

    v1 = vec(p_wrist, p_middle)
    v2 = vec(p_index, p_pinky)

    normal = cross(v1, v2)
    normal = normalize(normal)
    return v1, v2, normal


def hand_local_axes(landmarks):
    x = vec(landmarks[5], landmarks[17])
    y = compute_palm_normal(landmarks)
    x = normalize(x)
    z = normalize(np.cross(x, y))

    y = normalize(np.cross(z, x))
    return x, y, z


def hand_orientation(y, x, palm_normal, world_up=(0, -1, 0), world_right=(1, 0, 0), world_forward=(0, 0, 1)):
    ang_z = angle_between(palm_normal, np.array(world_up))
    ang_y = angle_between(y, np.array(world_right))
    ang_x = angle_between(palm_normal, np.array(world_forward))

    incl = "AMBIG"
    dir = "AMBIG"
    prof = "AMBIG"

    if abs(ang_x-90) <= cfg.ORIENTATION_ANGLE_TOLERANCE:
        prof = "FRONTAL"
    elif abs(ang_x-0) <= cfg.ORIENTATION_ANGLE_TOLERANCE or abs(ang_x-180) <= cfg.ORIENTATION_ANGLE_TOLERANCE: 
        prof = "PROFILE"

    if abs(ang_z-90) <= cfg.INCLINSAISON_VERTICAL_ANGLE_TOLERANCE:
        incl = "VERTICAL"
    elif abs(ang_z-180) <= cfg.INCLINSAISON_HORIZONTAL_ANGLE_TOLERANCE or abs(ang_z) <= cfg.INCLINSAISON_HORIZONTAL_ANGLE_TOLERANCE: 
        incl = "HORIZONTAL"
    
    if abs(ang_y-90) <= cfg.ORIENTATION_ANGLE_TOLERANCE:
        dir = "VERTICAL"
    elif abs(ang_y-180) <= cfg.ORIENTATION_ANGLE_TOLERANCE or abs(ang_y) <= cfg.ORIENTATION_ANGLE_TOLERANCE: 
        dir = "HORIZONTAL"
            
    return incl, ang_z, dir, ang_y, prof, ang_x