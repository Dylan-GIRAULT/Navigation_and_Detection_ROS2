import numpy as np
import os

data_path = "dataset_npy/frame_00000.npy"
if os.path.exists(data_path):
    data = np.load(data_path)
    # data est [N, 5] -> x, y, z, intensity, label
    labels = data[:, 4]
    nb_route = np.sum(labels == 1)
    nb_total = len(labels)
    
    print(f"Analyse de {data_path}:")
    print(f"Points totaux : {nb_total}")
    print(f"Points étiquetés ROUTE (label=1) : {nb_route}")
    print(f"X min/max : {data[:,0].min():.2f} / {data[:,0].max():.2f}")
    print(f"Y min/max : {data[:,1].min():.2f} / {data[:,1].max():.2f}")
else:
    print("Fichier non trouvé.")
