import numpy as np

from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from shared.enums.enum_segment_type import ROND_POINT_BAS, ROND_POINT_HAUT, LIGNE_DROITE, NO_SEGMENT
from shared.enums.enum_vehicule_action import RIGHT,STOP, START
import os


# Get current path of this file
current_path = os.path.dirname(os.path.abspath(__file__))


# Initialize empty lists for different circuit segments
# Represent list of points for each segment
tab_rond_point_bas = []
tab_rond_point_haut = []
tab_ligne_droite = []

# Represent list of points for decision
tab_decision_point = []

# Name of the circuit file
package_share_shared = get_package_share_directory('shared')
name_file = Path(package_share_shared) / 'resource' / 'better_circuit_enu_simplify.csv'

# Name of the decision area
package_share_navigation = get_package_share_directory('navigation')
name_file_decision = Path(package_share_navigation) / 'resource' / 'better_circuit_enu_decision_point.csv'

# Load circuit data from file if it exists
file = Path(name_file)
file_decision = Path(name_file_decision)

if file.is_file() and file_decision.is_file(): # if file exist
    m = np.loadtxt(name_file, comments="#", delimiter=",", unpack=False, skiprows=1, usecols=(1,2,3))
    m_type= np.loadtxt(name_file, comments="#", delimiter=",", unpack=False, skiprows=1, usecols=(0), dtype=str)

    m_decision = np.loadtxt(name_file_decision, comments="#", delimiter=",", unpack=False, skiprows=1, usecols=(1,2,3))

    
    for i in range(len(m_type)):
        element = m_type[i]
        if element == 'rond_point_bas':
            tab_rond_point_bas.append(m[i])
        elif element == "rond_point_haut":
            tab_rond_point_haut.append(m[i])
        elif element == "ligne_droite":
            tab_ligne_droite.append(m[i])
        else:
            print("Weird value :", element)

    tab_rond_point_bas = np.array(tab_rond_point_bas).T
    tab_rond_point_haut = np.array(tab_rond_point_haut).T
    tab_ligne_droite = np.array(tab_ligne_droite).T

    for i in range(len(m_decision)):
        tab_decision_point.append(m_decision[i])
    
    tab_decision_point = np.array(tab_decision_point).T


global dist_minimal_to_choose
dist_minimal_to_choose = 2.0

def is_in_decision_area(x : list[float]) -> bool:
    """
     x : position and rotatin [x,y,theta]
     return : true if in decision area else false
     """
    
    global dist_minimal_to_choose
    dist_min_decision_point = float('inf')

    if len(tab_decision_point) != 0: # decision area
        for i in range(tab_decision_point.shape[1]):
            dist = np.linalg.norm(x[0:2] - tab_decision_point[0:2,i])
            if dist < dist_min_decision_point:
                dist_min_decision_point = dist

    if dist_min_decision_point < dist_minimal_to_choose:
        return True
    else:
        return False

# Function to determine the circuit segment based on position
def get_circuit_pos(x: list[float], vehicule_action: int, actual_segment: int) -> int:
    """
     x : position and rotatin [x,y,theta]
     vehicule_action : current vehicle action
     actual_segment : current segment type
     return : segment type
     """
    """"""
    dist_min_rond_point_bas = float('inf')
    dist_min_rond_point_haut = float('inf')
    dist_min_ligne_droite = float('inf')
    dist_min_decision_point = float('inf')

    global dist_minimal_to_choose

    # Will get for each segment type the minimal distance to the vehicule

    if len(tab_rond_point_bas) != 0: # rond point bas
        for i in range(tab_rond_point_bas.shape[1]):
            dist = np.linalg.norm(x[0:2] - tab_rond_point_bas[0:2,i])
            if dist < dist_min_rond_point_bas:
                dist_min_rond_point_bas = dist
    
    
    if len(tab_rond_point_haut) != 0: # rond point haut
        for i in range(tab_rond_point_haut.shape[1]):
            dist = np.linalg.norm(x[0:2] - tab_rond_point_haut[0:2,i])
            if dist < dist_min_rond_point_haut:
                dist_min_rond_point_haut = dist
    
    if len(tab_ligne_droite) != 0: # ligne droite
        for i in range(tab_ligne_droite.shape[1]):
            dist = np.linalg.norm(x[0:2] - tab_ligne_droite[0:2,i])
            if dist < dist_min_ligne_droite:
                dist_min_ligne_droite = dist
    
    b_decision_area = is_in_decision_area(x)


    # Default segment type is same as before
    segment_type = actual_segment

    # Verification when vehicule is stopped

    if(actual_segment == NO_SEGMENT and vehicule_action != START):
        # vehicule is stopped and no started yet
        return NO_SEGMENT
    elif(actual_segment == NO_SEGMENT and vehicule_action == START):
        if dist_min_rond_point_bas < dist_minimal_to_choose:
            # vehicule is starting near "rond point bas"
            segment_type = ROND_POINT_BAS
        elif dist_min_rond_point_haut < dist_minimal_to_choose:
            # vehicule is starting near "rond point haut"
            segment_type = ROND_POINT_HAUT
        else: 
            # vehicule is starting far from any "rond point"
            segment_type = LIGNE_DROITE




    # if we are near a decision point
    if b_decision_area:
        # If the vehicule go right
        if vehicule_action == RIGHT: 
            segment_type = LIGNE_DROITE
        else:
            if dist_min_rond_point_bas < dist_min_rond_point_haut:
                segment_type = ROND_POINT_BAS
            else:
                segment_type = ROND_POINT_HAUT

    if(vehicule_action == STOP):
        return NO_SEGMENT

    # Verification coherence "rond point" and distance
    if segment_type == ROND_POINT_BAS:
        if dist_min_rond_point_bas > dist_minimal_to_choose:
            # too far
            segment_type = LIGNE_DROITE
    elif segment_type == ROND_POINT_HAUT:
        if dist_min_rond_point_haut > dist_minimal_to_choose:
            # too far
            segment_type = LIGNE_DROITE
    
    return segment_type