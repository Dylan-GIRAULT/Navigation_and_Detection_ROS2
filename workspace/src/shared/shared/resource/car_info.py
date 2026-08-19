import numpy as np


# Car constants
LONGUEUR = 4.084
LARGEUR  = 1.730
EMPATTEMENT = 2.588
VOIE_AV = 1.506
VOIE_AR = 1.489
d_arar = 0.5 # distance entre l'arrière de la voiture et le centre des roues arrières, pas indiqué dans wikipédia, à mesurer
rayon_roue = 0.311 # rayon des roues, pas indiqué dans wikipédia, à mesurer
demi_largeur_roue = 0.090 # demi-largeur des roues, pas indiqué dans wikipédia, à mesurer
L = EMPATTEMENT
centre_vehicule = np.array([[0.0], [0.0], [0.0]])
