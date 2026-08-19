import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import math
import matplotlib.pyplot as plt

# -------------------------------
# Chargement de la map avec segments
# -------------------------------
def load_map_with_segments(csv_path):
    df = pd.read_csv(csv_path)
    points = df[["E", "N"]].to_numpy()
    segments = df["segment"].to_numpy()  # array de même longueur
    return points, segments

# -------------------------------
# Extraction du scan depuis une frame
# -------------------------------
def extract_scan_from_frame(frame, threshold=200):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(binary > 0)
    scan = np.column_stack((xs, -ys))  # inversion Y pour correspondre à la carte
    return scan

# -------------------------------
# Normalisation globale pour ICP
# -------------------------------
def normalize(map_pts, scan_pts):
    map_c = map_pts - map_pts.mean(axis=0)
    scan_c = scan_pts - scan_pts.mean(axis=0)
    scale = np.ptp(map_c, axis=0).mean() / np.ptp(scan_c, axis=0).mean()
    scan_c *= scale
    return map_c, scan_c, map_pts.mean(axis=0)

# -------------------------------
# Rotation
# -------------------------------
def rotate(points, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return (R @ points.T).T

# -------------------------------
# Open3D PointCloud
# -------------------------------
def to_pcd(points):
    pts3d = np.column_stack((points, np.zeros(len(points))))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    return pcd

# -------------------------------
# ICP global
# -------------------------------
def icp(source_pts, target_pts, init_T, max_dist=5.0):
    source = to_pcd(source_pts)
    target = to_pcd(target_pts)
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_dist,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result

def global_icp(map_pts, scan_pts, angle_step_deg=15):
    best = None
    all_results = []

    for deg in range(-180, 180, angle_step_deg):
        theta = np.deg2rad(deg)
        rotated = rotate(scan_pts, theta)
        init_T = np.eye(4)
        init_T[:2, :2] = [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ]
        result = icp(rotated, map_pts, init_T)
        entry = {
            "deg": deg,
            "T": result.transformation,
            "fitness": result.fitness,
            "rmse": result.inlier_rmse,
            "corr": len(result.correspondence_set)
        }
        all_results.append(entry)
        if best is None or (
            entry["fitness"] > best["fitness"] and
            entry["rmse"] < best["rmse"]
        ):
            best = entry

    return best, all_results

# -------------------------------
# Confidence metrics
# -------------------------------
def confidence_metrics(best, all_results):
    rmses = [r["rmse"] for r in all_results]
    confidence = 0.6 * best["fitness"] + 0.4 * np.exp(-best["rmse"])
    ambiguous = np.std(rmses) < 0.05
    return confidence, ambiguous

# -------------------------------
# Extraction de la pose dans ENU
# -------------------------------
def extract_pose_in_enu(T, map_mean):
    """
    T : transformation ICP sur les points normalisés
    map_mean : centre original de la map (avant normalisation)
    """
    x_icp, y_icp = T[0, 3], T[1, 3]
    x_enu = x_icp + map_mean[0]
    y_enu = y_icp + map_mean[1]
    theta = math.atan2(T[1, 0], T[0, 0])
    return x_enu, y_enu, theta

# -------------------------------
# Trouver le segment le plus proche
# -------------------------------
def find_closest_segment(x, y, map_pts, segments):
    dists = np.linalg.norm(map_pts - np.array([x, y]), axis=1)
    idx_min = np.argmin(dists)
    return segments[idx_min]

# -------------------------------
# Pipeline complet pour la vidéo
# -------------------------------
def process_video(csv_map, video_path, frame_step=5):
    map_pts, map_segments = load_map_with_segments(csv_map)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    poses = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_step != 0:
            frame_id += 1
            continue

        scan = extract_scan_from_frame(frame)
        if len(scan) < 100:
            frame_id += 1
            continue

        map_n, scan_n, map_mean = normalize(map_pts, scan)
        best, all_results = global_icp(map_n, scan_n)
        confidence, ambiguous = confidence_metrics(best, all_results)
        x, y, theta = extract_pose_in_enu(best["T"], map_mean)
        segment_name = find_closest_segment(x, y, map_pts, map_segments)

        poses.append({
            "frame": frame_id,
            "x": x,
            "y": y,
            "theta": np.degrees(theta),
            "confidence": confidence,
            "ambiguous": ambiguous,
            "segment": segment_name
        })

        print(f"[Frame {frame_id}] x={x:.2f}, y={y:.2f}, "
              f"θ={np.degrees(theta):.1f}°, "
              f"conf={confidence:.2f}, "
              f"ambig={ambiguous}, "
              f"zone={segment_name}")

        frame_id += 1

    cap.release()
    return poses, map_pts, map_segments

# -------------------------------
# Affichage de la carte et des positions
# -------------------------------
def plot_map_with_poses(map_pts, map_segments, poses):
    plt.figure(figsize=(10,8))
    ax = plt.gca()

    # Affichage de la carte (points du circuit)
    unique_segments = np.unique(map_segments)
    colors = plt.cm.get_cmap("tab20", len(unique_segments))

    for i, seg_name in enumerate(unique_segments):
        mask = map_segments == seg_name
        pts = map_pts[mask]
        ax.plot(pts[:,0], pts[:,1], linestyle='--', label=seg_name, color=colors(i))

    # Affichage des poses détectées
    xs = [p["x"] for p in poses]
    ys = [p["y"] for p in poses]
    ax.plot(xs, ys, c='red', marker='o', label="Trajectoire détectée")

    ax.set_aspect('equal')
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True)
    ax.legend()
    plt.title("Carte ENU + localisation des scans")
    plt.show()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    csv_path = "circuit_enu_simplify.csv"
    video_path = "Nuage de points sans bruit.mp4"

    poses, map_pts, map_segments = process_video(csv_path, video_path, frame_step=5)
    plot_map_with_poses(map_pts, map_segments, poses)
