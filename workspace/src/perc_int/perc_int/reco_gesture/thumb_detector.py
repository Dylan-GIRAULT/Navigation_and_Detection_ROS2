import numpy as np
from .geom_utils import vec, normalize, length
from .orientation_detector import hand_local_axes


def thumb_direction(landmarks):
    start = landmarks[2]
    tip = landmarks[4]
    v = vec(start, tip)
    return normalize(v)


def thumb_direction_compass(landmarks):
    t = thumb_direction(landmarks)
    world_x = np.array([1, 0, 0]) 
    world_y = np.array([0, -1, 0])
    world_z = np.array([0, 0, 1]) 
    comps = {'X': np.dot(t, world_x), 'Y': np.dot(t, world_y), 'Z': np.dot(t, world_z)}

    axis = max(('X', 'Y', 'Z'), key=lambda k: abs(comps[k]))
    val = comps[axis]
    if abs(val) < 0.45:
        return 'UNKNOWN', 0.0
    if axis == 'X':
        return ('RIGHT' if val > 0 else 'LEFT'), abs(val)
    if axis == 'Y':
        return ('UP' if val > 0 else 'DOWN'), abs(val)
    if axis == 'Z':
        return ('FORWARD' if val > 0 else 'BACK'), abs(val)
    return 'UNKNOWN', 0.0
