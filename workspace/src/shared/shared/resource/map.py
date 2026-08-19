import numpy as np
from ament_index_python.packages import get_package_share_directory
import os

package_share_directory = get_package_share_directory('shared')
csv_file_path = os.path.join(package_share_directory, 'resource', 'better_circuit_enu_simplify.csv') # Path to your CSV file
csv_file_path_cheat_map = os.path.join(package_share_directory, 'resource', 'SevilleCoubureSignee.csv') # Path to your CSV file
csv_file_path_map_bag = os.path.join(package_share_directory, 'resource', 'map_bag.csv') # Path to your CSV file


def generate_specific_rows(filePath, segment="rond_point_1"):
    with open(filePath) as f:
        # using enumerate to track line no.
        for i, line in enumerate(f):
            if i == 0:
                continue
            if segment == "":
                yield line[line.find(",")+1:]
            elif line.startswith(segment):
                yield line.removeprefix(segment+",")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
gen_ligne_droite = generate_specific_rows(os.path.join(__location__, csv_file_path), "ligne_droite")
map_ligne_droite = np.loadtxt(gen_ligne_droite, comments="#", delimiter=",", unpack=False, usecols=[0,1]).T
gen_rond_point_haut = generate_specific_rows(os.path.join(__location__, csv_file_path), "rond_point_haut")
map_rond_point_haut = np.loadtxt(gen_rond_point_haut, comments="#", delimiter=",", unpack=False, usecols=[0,1]).T
gen_rond_point_bas = generate_specific_rows(os.path.join(__location__, csv_file_path), "rond_point_bas")
map_rond_point_bas = np.loadtxt(gen_rond_point_bas, comments="#", delimiter=",", unpack=False, usecols=[0,1]).T

gen_cheat_map = generate_specific_rows(os.path.join(__location__, csv_file_path_cheat_map), "map")
cheat_map = np.loadtxt(gen_cheat_map, comments="#", delimiter=",", unpack=False, usecols=[0,1]).T
gen_map_bag = generate_specific_rows(os.path.join(__location__, csv_file_path_map_bag), "map")
map_bag = np.loadtxt(gen_map_bag, comments="#", delimiter=",", unpack=False, usecols=[0,1]).T

# Compute curvature for each map using gradient method
def compute_curvature(map_data):
    x = map_data[0]
    y = map_data[1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature
curvature_ligne_droite = compute_curvature(map_ligne_droite)
curvature_rond_point_haut = compute_curvature(map_rond_point_haut)
curvature_rond_point_bas = compute_curvature(map_rond_point_bas)
curvature_cheat_map = compute_curvature(cheat_map)
# Do a moving average to smooth the curvature
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
curvature_ligne_droite = moving_average(curvature_ligne_droite)
curvature_rond_point_haut = moving_average(curvature_rond_point_haut)
curvature_rond_point_bas = moving_average(curvature_rond_point_bas)
curvature_cheat_map = moving_average(curvature_cheat_map)