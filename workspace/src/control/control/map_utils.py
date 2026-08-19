from shared.resource.map import (
    map_ligne_droite,
    map_rond_point_bas,
    map_rond_point_haut,
    cheat_map,
    curvature_ligne_droite,
    curvature_rond_point_bas,
    curvature_rond_point_haut,
    curvature_cheat_map,
)
from shared.enums.enum_segment_type import (
    ROND_POINT_BAS,
    SORTIE_ROND_POINT_BAS,
    ENTREE_ROND_POINT_BAS,
    ROND_POINT_HAUT,
    SORTIE_ROND_POINT_HAUT,
    ENTREE_ROND_POINT_HAUT,
    LIGNE_DROITE,
    NO_SEGMENT,
)

class NoMapToFollowException(Exception):
    pass

def segment_to_map(segment: int, use_cheat_map: bool = False):
    if (segment == ROND_POINT_BAS
        or segment == SORTIE_ROND_POINT_BAS
        or segment == ENTREE_ROND_POINT_BAS):
        return cheat_map if use_cheat_map else map_rond_point_bas
    if (segment == ROND_POINT_HAUT
        or segment == SORTIE_ROND_POINT_HAUT
        or segment == ENTREE_ROND_POINT_HAUT):
        return cheat_map if use_cheat_map else map_rond_point_haut
    elif segment == LIGNE_DROITE:
        return cheat_map if use_cheat_map else map_ligne_droite
    else:
        raise NoMapToFollowException
