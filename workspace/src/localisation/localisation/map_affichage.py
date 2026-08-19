#L'affichage est pour l'instant fait dans le repère géographique WGS84 (lat, lon) (EPSG:4326)


import numpy as np
import matplotlib.pyplot as plt

import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

def load_segments(csv_file):
    segments = defaultdict(list)

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg = row["segment"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            segments[seg].append((lat, lon))

    for seg in segments:
        segments[seg] = np.array(segments[seg])

    return segments



def calcule_normales(points):
    lat = points[:,0]
    lon = points[:,1]

    n = len(points)
    vx = np.zeros(n)
    vy = np.zeros(n)

    for i in range(n):
        if i == 0:
            dx = lon[i+1] - lon[i]
            dy = lat[i+1] - lat[i]
        elif i == n-1:
            dx = lon[i] - lon[i-1]
            dy = lat[i] - lat[i-1]
        else:
            dx = lon[i+1] - lon[i-1]
            dy = lat[i+1] - lat[i-1]

        norm = np.sqrt(dx*dx + dy*dy)
        if norm < 1e-12:
            if i > 0:
                vx[i], vy[i] = vx[i-1], vy[i-1]
            else:
                vx[i], vy[i] = 1.0, 0.0
        else:
            vx[i] = dx / norm
            vy[i] = dy / norm

    ny, nx = vx, -vy
    return nx, ny


def plot_track_with_borders(csv_file, road_width=0.00005):
    segments = load_segments(csv_file)

    plt.figure("Circuit complet")
    fig = plt.gca()

    for seg_name, pts in segments.items():
        lat = pts[:,0]
        lon = pts[:,1]

        nx, ny = calcule_normales(pts)

        left_lat  = lat + road_width * ny
        left_lon  = lon + road_width * nx
        right_lat = lat - road_width * ny
        right_lon = lon - road_width * nx

        plt.plot(lon, lat, '--',)
        plt.plot(left_lon,  left_lat)
        plt.plot(right_lon, right_lat)

    plt.axis("equal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()



# Exemple d'appel :
plot_track_with_borders("circuit.csv", road_width=0.00005)
