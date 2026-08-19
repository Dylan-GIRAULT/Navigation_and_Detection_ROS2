# Define constants for circuit segment types
ROND_POINT_BAS = 0
SORTIE_ROND_POINT_BAS = 1
ENTREE_ROND_POINT_BAS = 2
ROND_POINT_HAUT = 3
SORTIE_ROND_POINT_HAUT = 4
ENTREE_ROND_POINT_HAUT = 5
LIGNE_DROITE = 6
NO_SEGMENT = 7


def segment_to_str(segment: int) -> str:
    if segment == ROND_POINT_BAS:
        return "ROND_POINT_BAS"
    elif segment == SORTIE_ROND_POINT_BAS:
        return "SORTIE_ROND_POINT_BAS"
    elif segment == ENTREE_ROND_POINT_BAS:
        return "ENTREE_ROND_POINT_BAS"
    elif segment == ROND_POINT_HAUT:
        return "ROND_POINT_HAUT"
    elif segment == SORTIE_ROND_POINT_HAUT:
        return "SORTIE_ROND_POINT_HAUT"
    elif segment == ENTREE_ROND_POINT_HAUT:
        return "ENTREE_ROND_POINT_HAUT"
    elif segment == LIGNE_DROITE:
        return "LIGNE_DROITE"
    elif segment == NO_SEGMENT:
        return "NO_SEGMENT"
    else:
        return "UNKNOWN_SEGMENT"