import math
import numpy as np


def vec(a, b):
    return np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]], dtype=float)


def length(v):
    return float(np.linalg.norm(v))


def normalize(v):
    n = length(v)
    if n == 0:
        return np.zeros_like(v)
    return v / n


def dot(u, v):
    return float(np.dot(u, v))


def angle_between(u, v, safe_clip=True):
    uu = normalize(u)
    vv = normalize(v)
    d = dot(uu, vv)
    if safe_clip:
        d = max(-1.0, min(1.0, d))
    return math.degrees(math.acos(d))


def cross(u, v):
    return np.cross(u, v)


def centroid(points):
    pts = np.array(points, dtype=float)
    return tuple(np.mean(pts, axis=0).tolist())
