import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, sqrt, radians

def geodetic_to_ecef(lat, lon, h):
    a = 6378137.0
    e2 = 6.69437999014e-3

    lat = np.radians(lat)
    lon = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + h) * np.sin(lat)    

    return X, Y, Z

def ecef_to_enu(X, Y, Z, lat0, lon0, h0):
    X0, Y0, Z0 = geodetic_to_ecef(lat0, lon0, h0)

    dx = X - X0
    dy = Y - Y0
    dz = Z - Z0

    lat0 = radians(lat0)
    lon0 = radians(lon0)

    t = -np.sin(lon0)*dx + np.cos(lon0)*dy
    e = t

    t = -np.sin(lat0)*np.cos(lon0)*dx - np.sin(lat0)*np.sin(lon0)*dy + np.cos(lat0)*dz
    n = t

    t =  np.cos(lat0)*np.cos(lon0)*dx + np.cos(lat0)*np.sin(lon0)*dy + np.sin(lat0)*dz
    u = t

    return e, n, u

def wgs84_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    X, Y, Z = geodetic_to_ecef(lat, lon, alt)
    return ecef_to_enu(X, Y, Z, lat_ref, lon_ref, alt_ref)


def load_latlon_segments(csv_file):
    segments = defaultdict(list)
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg = row["segment"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            segments[seg].append((lat, lon))
    for k in list(segments.keys()):
        segments[k] = np.array(segments[k], dtype=float)
    return segments


def convert_segments_to_enu(segments_latlon, lat_ref, lon_ref, alt_ref):
    segments_enu = {}
    for seg, pts in segments_latlon.items():
        if pts.size == 0:
            segments_enu[seg] = np.zeros((0,3))
            continue

        lats = pts[:,0]
        lons = pts[:,1]
        alts = np.zeros_like(lats)  

        en = [wgs84_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref)
              for lat, lon, alt in zip(lats, lons, alts)]

        segments_enu[seg] = np.array(en)
    return segments_enu

def save_enu_csv(segments_enu, out_file):
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["segment", "E", "N", "U"])
        for seg, pts in segments_enu.items():
            for e, n, u in pts:
                w.writerow([seg, f"{e:.6f}", f"{n:.6f}", f"{u:.6f}"])
    print(f"[INFO] ENU sauvegardé dans {out_file}")


def compute_unit_tangents(points_xy):
    n = len(points_xy)
    tx = np.zeros(n)
    ty = np.zeros(n)

    if n == 0:
        return tx, ty, tx, ty

    x = points_xy[:,0]
    y = points_xy[:,1]

    for i in range(n):
        if i == 0 and n>1:
            dx = x[1] - x[0]
            dy = y[1] - y[0]
        elif i == n-1 and n>1:
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
        elif n>2:
            dx = x[i+1] - x[i-1]
            dy = y[i+1] - y[i-1]
        else:
            dx = 0.0
            dy = 0.0

        norm = np.hypot(dx, dy)
        if norm < 1e-8:
            if i>0:
                tx[i], ty[i] = tx[i-1], ty[i-1]
            else:
                tx[i], ty[i] = 1.0, 0.0
        else:
            tx[i] = dx / norm
            ty[i] = dy / norm

    nx = -ty
    ny = tx

    return tx, ty, nx, ny


def plot_segments_with_borders(segments_enu, road_half_width_m=3.0, show=True):
    plt.figure("CARTE ENU")
    ax = plt.gca()

    for seg, pts in segments_enu.items():
        if len(pts) == 0:
            continue
        x = pts[:,0]
        y = pts[:,1]

        tx, ty, nx, ny = compute_unit_tangents(np.column_stack((x,y)))

        left_x  = x + road_half_width_m * nx
        left_y  = y + road_half_width_m * ny
        right_x = x - road_half_width_m * nx
        right_y = y - road_half_width_m * ny

        ax.plot(x, y, '--', linewidth=1)
        ax.plot(left_x, left_y, linewidth=1)
        ax.plot(right_x, right_y, linewidth=1)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True)
    ax.legend()
    if show:
        plt.show()



if __name__ == "__main__":
    csv_in = "circuit.csv"
    csv_out = "circuit_enu.csv"

    lat_ref = 49.401322
    lon_ref = 2.797066
    alt_ref = 80.5

    if not os.path.exists(csv_in):
        raise SystemExit(f"Fichier {csv_in} introuvable")

    segs_latlon = load_latlon_segments(csv_in)
    segs_enu = convert_segments_to_enu(segs_latlon, lat_ref, lon_ref, alt_ref)
    #save_enu_csv(segs_enu, csv_out)

    plot_segments_with_borders(segs_enu, road_half_width_m=1.5)

"""ros_parameters : 
        lat_ref : 49.401322
        lon_ref : 2.797066
        alt_ref : 80.5
"""
