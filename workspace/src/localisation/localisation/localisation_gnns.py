import numpy as np
import matplotlib.pyplot as plt
import csv

from map_affichage_enu import wgs84_to_enu      


def load_enu_csv(csv_file):
    segments = {}

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            seg = row["segment"]
            E = float(row["E"])
            N = float(row["N"])
            U = float(row["U"])

            if seg not in segments:
                segments[seg] = []

            segments[seg].append([E, N, U])

    for k in segments:
        segments[k] = np.array(segments[k], dtype=float)

    return segments


def find_closest_segment_and_inside(segments_enu, point_enu, road_half_width=1.5):
    px, py, _ = point_enu

    best_seg = None
    best_dist = float('inf')
    best_inside = False

    for seg, pts in segments_enu.items():
        if len(pts) < 2:
            continue

        xy = pts[:, :2]

        d = np.linalg.norm(xy - np.array([px, py]), axis=1)
        idx = np.argmin(d)

        closest_x, closest_y = xy[idx]
        dist = d[idx]

        if idx == 0:
            dx = xy[1,0] - xy[0,0]
            dy = xy[1,1] - xy[0,1]
        elif idx == len(xy) - 1:
            dx = xy[-1,0] - xy[-2,0]
            dy = xy[-1,1] - xy[-2,1]
        else:
            dx = xy[idx+1,0] - xy[idx-1,0]
            dy = xy[idx+1,1] - xy[idx-1,1]

        norm = np.hypot(dx, dy)
        if norm < 1e-8:
            continue
        tx, ty = dx/norm, dy/norm

        nx, ny = -ty, tx   

        vx = px - closest_x
        vy = py - closest_y

        lateral = vx * nx + vy * ny
        inside = abs(lateral) <= road_half_width

        if dist < best_dist:
            best_dist = dist
            best_seg = seg
            best_inside = inside

    return best_seg, best_dist, best_inside



def get_segment_info_from_gnss(lat, lon, alt,segments_enu,lat_ref, lon_ref, alt_ref,road_half_width=1.5):
    E, N, U = wgs84_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref)

    seg, dist, inside = find_closest_segment_and_inside(segments_enu,(E, N, U),road_half_width=road_half_width)

    return seg, dist, inside, (E, N, U)

def plot_map_with_point(segments_enu, point_enu, closest_segment=None):
    plt.figure(figsize=(10, 10))

    for seg, pts in segments_enu.items():
        x = pts[:, 0]
        y = pts[:, 1]

        if seg == closest_segment:
            plt.plot(x, y, linewidth=4, label=f"{seg} (closest)", alpha=0.8)
        else:
            plt.plot(x, y, linewidth=2, alpha=0.4)

    px, py, _ = point_enu
    plt.scatter(px, py, s=120, edgecolors='black', facecolors='yellow', zorder=5, label="Robot")

    plt.xlabel("E (m)")
    plt.ylabel("N (m)")
    plt.title("Carte ENU + Position Robot")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    plt.show()


#Exemple
"""
if __name__ == "__main__":
    csv_enu = "circuit_enu.csv"
    segments_enu = load_enu_csv(csv_enu)
    lat_ref = 49.401322
    lon_ref = 2.797066
    alt_ref = 80.5
    #Donnée GNSS exemple
    lat = 49.4020766
    lon = 2.7948369
    alt = 82.0

    seg, dist, inside, point_enu = get_segment_info_from_gnss(lat, lon, alt,segments_enu,lat_ref, lon_ref, alt_ref)

    print("Segment le plus proche :", seg)
    print("Distance au segment :", dist)
    print("Entre les bords :", inside)
    print("Position ENU :", point_enu)

    #Visualisation (optionnel)
    #plot_map_with_point(segments_enu, point_enu, closest_segment=seg)
"""
