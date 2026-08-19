import numpy as np


def sub_mod_pi2(x,y):
#pour 2 angles passes en parametre, calcule l'ecart e=x-y 
#retourne une valeur entre -pi et pi
#fct utile pour calculer des ecarts angulaires dans des commandes 
#car un ecart angulaire de commande doit rester petit

    if (x>=y):
        while np.abs(x-y)>np.pi:
            x=x-2*np.pi
    else:
        while np.abs(x-y)>np.pi:
            y=y-2*np.pi
    return x-y



def ecart_pente_frenet(A,B,X):
    #Calcule :
    #   - e = ecart lateral signe
    #   - p = pente signee du segment [AB] (angle alpha)
    #En entree, on passe les deux points A et B et la position x de la
    #voiture
    #Le repere de Frenet a le point A pour origine et B definit l'axe x
    
    M = X[0:2]#position 
    xa=A[0]
    ya=A[1]
    xb=B[0]
    yb=B[1]
    alpha=np.arctan2((yb-ya),(xb-xa))
    #mat homogene
    O_T_A=np.array([
        [np.cos(alpha), -np.sin(alpha), xa],
        [np.sin(alpha),  np.cos(alpha), ya],
        [0            ,    0          ,  1]])
    #dans RA (repere de frenet)
    A_M = np.dot(np.linalg.inv(O_T_A), np.append(M, np.array([1])))
    axm=A_M[0]
    aym=A_M[1]
    
    A_B = np.dot(np.linalg.inv(O_T_A), np.append(B, np.array([1])))
    axb=A_B[0]
#    ayb=A_B[1]
    
    if (axm<0):
        e=np.sign(aym)*np.linalg.norm(A-M)
    elif (axm>axb):
        e=np.sign(aym)*np.linalg.norm(B-M)
    else:
        e=aym

    p=alpha
    
    return e,p



def closest_segment(X, m, prev_A=np.array([])):
    """
    Parameters
    ----------
    X: list
        (x, y, theta) state of the car
    m: array
        Map
    prev_A: array (x,y)
        Previous closest segment, leave empty if None
    
    Returns
    -------
    e: float
        Error between car and closest segment
    p: float
        Slope of the segment (rad)
    A: array
        First segment point (x,y)
    B: array
        Second segment point (x,y)
    """

    e=np.inf
    ecart_ang=np.inf#ecart angulaire cap et orientation segment
    indice = 0
    for j in range(0, m.size//2-1):
        A=np.array([m[0,j], m[1,j]])
        B=np.array([m[0,j+1], m[1,j+1]])
        dk,pk=ecart_pente_frenet(A,B,X)
        ecart_ang_j=sub_mod_pi2(pk,X[2])
        #on cherche le segment qui minimise l'ecart et la distance
        if (np.abs(dk)+np.abs(ecart_ang_j))<(np.abs(e)+np.abs(ecart_ang)):
            #ce segment est plus optimal que ceux qu'on a traités avant
            e=dk
            p=pk
            ecart_ang=ecart_ang_j
            indice = j
    
    #On affiche le segment matche qui va de A vers B
    A=m[:,indice]
    B=m[:,indice+1]
    
    return e,p,A,B



def estimate_curvature(m, A, B, pos):
    
    # Compute curvature using gradient method
    def compute_curvature(map):
        x = map[0]
        y = map[1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        return curvature
    curvatures = compute_curvature(m)

    # Do a moving average to smooth the curvature
    def moving_average(data, window_size=5):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    curvatures = moving_average(curvatures)

    # Compute the curvature at points A and B
    idxA = int(np.argmin(np.linalg.norm(m - A.reshape(2, 1), axis=0)))
    idxB = int(np.argmin(np.linalg.norm(m - B.reshape(2, 1), axis=0)))
    curvature_A = curvatures[idxA]
    curvature_B = curvatures[idxB]
    # Project the position pos onto the segment [A, B]
    AB = B - A
    AB_norm_squared = np.dot(AB, AB)
    if AB_norm_squared < 1e-12:
        return 0.5 * (curvature_A + curvature_B)
    t = np.dot(pos - A, AB) / AB_norm_squared
    t = np.clip(t, 0.0, 1.0)
    # Perform linear interpolation to compute the curvature at point P
    curvature_P = (1 - t) * curvature_A + t * curvature_B
    return curvature_P
