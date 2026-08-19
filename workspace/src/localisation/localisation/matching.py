import numpy as np
import math
from ament_index_python.packages import get_package_share_directory
import os
import csv
from scipy.spatial import cKDTree

# --- fonctions CSV et segments ---
def load_circuit_points():
    pkg_share = get_package_share_directory('localisation')
    csv_path = os.path.join(pkg_share, 'better_circuit_enu_simplify.csv')
    points = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            points.append({
                "type": row["segment"],
                "xy": np.array([float(row["E"]), float(row["N"])])
            })
    return points

def build_segments(points):
    segments = []
    for i in range(len(points)-1):
        if points[i]["type"] == points[i+1]["type"]:
            segments.append({
                "type": points[i]["type"],
                "p1": points[i]["xy"],
                "p2": points[i+1]["xy"]
            })
    return segments

def sample_segment_thick(p1, p2, step=0.5, width=4.0):
    p1 = np.array(p1)
    p2 = np.array(p2)
    length = np.linalg.norm(p2 - p1)
    n_points = max(2, int(length / step))
    t = np.linspace(0, 1, n_points)
    center_line = (1 - t[:, None]) * p1 + t[:, None] * p2

    v = p2 - p1
    if np.linalg.norm(v) == 0:
        return center_line
    n = np.array([-v[1], v[0]])
    n /= np.linalg.norm(n)

    n_steps = max(2, int(width / step))
    offsets = np.linspace(-width/2, width/2, n_steps)
    points = []
    for off in offsets:
        points.append(center_line + off * n)

    return np.vstack(points)

def generate_map_point_cloud(segments, step=0.5, width=4.0):
    points = []
    for seg in segments:
        pts = sample_segment_thick(seg["p1"], seg["p2"], step, width)
        points.append(pts)
    if points:
        return np.vstack(points)
    else:
        return np.zeros((0,2))

def filter_segments_near_position(segments, x, y, radius=20.0):
    pos = np.array([x, y])
    filtered = []
    for seg in segments:
        if np.linalg.norm(seg["p1"] - pos) <= radius or np.linalg.norm(seg["p2"] - pos) <= radius:
            filtered.append(seg)
    return filtered

# --- ICP 2D accéléré ---
def icp(source_points: np.ndarray,
        x_init: float,
        y_init: float,
        theta_init: float = None,
        x_prev: float = None,
        y_prev: float = None,
        map_segments: list = None,
        max_iter: int = 30,          # Limitation des itérations
        tol: float = 1e-3,
        step_map: float = 0.5,
        width_map: float = 6.0,
        radius_local: float = 20.0):

    # 1️⃣ Charger la carte si nécessaire
    if map_segments is None:
        points_csv = load_circuit_points()
        map_segments = build_segments(points_csv)

    # 2️⃣ Filtrer les segments proches
    map_segments = filter_segments_near_position(map_segments, x_init, y_init, radius_local)
    if len(map_segments) == 0:
        return None, None, None

    # 3️⃣ Générer nuage de points carte
    map_points = generate_map_point_cloud(map_segments, step_map, width_map)

    # 4️⃣ Initialisation translation et rotation
    t = np.array([x_init, y_init])
    if theta_init is None and x_prev is not None and y_prev is not None:
        theta_init = math.atan2(y_init - y_prev, x_init - x_prev)
    if theta_init is None:
        R = np.eye(2)
        theta_init = 0.0
    else:
        R = np.array([[math.cos(theta_init), -math.sin(theta_init)],
                      [math.sin(theta_init),  math.cos(theta_init)]])

    # 5️⃣ Transformer le nuage source selon l’estimation initiale
    src = (R @ source_points.T).T + t

    # 6️⃣ Construire KD-tree pour la carte
    tree = cKDTree(map_points)

    # 7️⃣ Boucle ICP
    for i in range(max_iter):
        # Trouver les plus proches voisins via KD-tree
        dists, indices = tree.query(src)
        tgt = map_points[indices]

        # Centroides
        src_centroid = np.mean(src, axis=0)
        tgt_centroid = np.mean(tgt, axis=0)
        src_centered = src - src_centroid
        tgt_centered = tgt - tgt_centroid

        # Matrice de covariance + SVD
        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R_iter = Vt.T @ U.T
        if np.linalg.det(R_iter) < 0:
            Vt[1,:] *= -1
            R_iter = Vt.T @ U.T

        t_iter = tgt_centroid - R_iter @ src_centroid

        # Mettre à jour src
        src = (R_iter @ src.T).T + t_iter

        # Critère d’arrêt
        angle_diff = np.arccos(np.clip((np.trace(R_iter) - 1)/2, -1.0, 1.0))
        if np.linalg.norm(t_iter) < tol and angle_diff < tol:
            break

        # Accumuler rotation et translation
        R = R_iter @ R
        t = R_iter @ t + t_iter

    theta = math.atan2(R[1,0], R[0,0])
    return t[0], t[1], theta
