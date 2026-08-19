import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import os

# --- IMPORTS OPENPCDET ---
from spconv.pytorch.utils import PointToVoxel
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.backbones_2d import BaseBEVBackbone
from pcdet.models.backbones_3d.vfe.pillar_vfe import PillarVFE
from pcdet.models.backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter

# ==========================================
# CONFIGURATION
# ==========================================
# Ton fichier bag
BAG_FILE = "/home/younes/Documents/sy27-zoe/workspace/src/bagfiles/sy27_road_ground_truth_live/sy27_road_ground_truth_live_0.mcap"
TOPIC = "/points"
FRAME_INDEX = 150 # On prend la 150ème image

# Chemins relatifs à ton installation
BASE_DIR = "/home/younes/Documents/sy27-zoe/workspace/src/perc_ext/road_segmentation/road_segmentation"
MODEL_PATH = f"{BASE_DIR}/model/road_segmentation_epoch_10.pth"
CFG_PATH = f"{BASE_DIR}/cfgs/pointpillar.yaml"

# ==========================================
# CLASSE MODELE
# ==========================================
class PointPillarsRoadSegmentation(nn.Module):
    def __init__(self, model_cfg, num_point_features=4):
        super().__init__()
        self.model_cfg = model_cfg
        # Range critique
        self._pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
        _voxel_size = [0.16, 0.16, 4]
        _grid_size = [432, 496, 1]
        
        self.vfe = PillarVFE(model_cfg=model_cfg.VFE, num_point_features=num_point_features, point_cloud_range=self._pc_range, voxel_size=_voxel_size)
        self.scatter = PointPillarScatter(model_cfg=model_cfg.MAP_TO_BEV, grid_size=_grid_size)
        self.backbone_2d = BaseBEVBackbone(model_cfg=model_cfg.BACKBONE_2D, input_channels=model_cfg.MAP_TO_BEV.NUM_BEV_FEATURES)
        self.seg_head = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1) 
        )

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        return self.seg_head(batch_dict['spatial_features_2d'])

# ==========================================
# UTILITAIRES
# ==========================================
def parse_pointcloud(msg):
    # Décodage manuel des bytes ROS
    dtype_list = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('i', '<f4')]
    if msg.point_step > 16: dtype_list.append(('padding', 'u1', (msg.point_step - 16,)))
    
    data = np.frombuffer(msg.data, dtype=np.uint8)
    arr = np.frombuffer(data, dtype=dtype_list)
    
    # DOWNSAMPLING (Optimisation RAM) : On garde 1 point sur 2
    arr = arr[::2]
    
    res = np.zeros((len(arr), 4), dtype=np.float32)
    res[:,0]=arr['x']; res[:,1]=arr['y']; res[:,2]=arr['z']; res[:,3]=arr['i']
    return res[~np.isnan(res).any(axis=1)]

def filter_range(points):
    # Crop strict : [0, -39.68, -3, 69.12, 39.68, 1]
    mask = (points[:, 0] > 0) & (points[:, 0] < 69.12) & \
           (points[:, 1] > -39.68) & (points[:, 1] < 39.68) & \
           (points[:, 2] > -3) & (points[:, 2] < 1)
    return points[mask]

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Chargement modèle : {MODEL_PATH}")
    
    # 1. Chargement IA
    # Astuce du dossier courant pour la config
    cwd_backup = os.getcwd()
    os.chdir(BASE_DIR)
    cfg_from_yaml_file(CFG_PATH, cfg)
    os.chdir(cwd_backup)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointPillarsRoadSegmentation(model_cfg=cfg.MODEL).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 2. Voxel Generator
    voxel_generator = PointToVoxel(
        vsize_xyz=[0.16, 0.16, 4], 
        coors_range_xyz=[0, -39.68, -3, 69.12, 39.68, 1],
        num_point_features=4, max_num_voxels=16000, max_num_points_per_voxel=32
    )

    # 3. Lecture du Bag
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    print(f"Ouverture du bag : {Path(BAG_FILE).name}")
    
    with AnyReader([Path(BAG_FILE)], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == TOPIC]
        if not connections:
            print(f"Topic {TOPIC} introuvable !"); exit()

        # On saute jusqu'à la frame voulue
        count = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if count < FRAME_INDEX:
                count += 1
                continue
            
            print(f"Traitement de la frame #{count}...")
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # --- TRAITEMENT ---
            points = parse_pointcloud(msg)
            
            # Adaptation
            if points[:, 3].max() > 1.0: points[:, 3] /= 255.0
            points[:, 2] -= (1.73 - 1.60) # Hauteur robot
            
            # Filtre (Vital pour la RAM)
            points = filter_range(points)
            print(f"Points restants après filtre : {len(points)}")
            
            if len(points) < 100:
                print("Pas assez de points valides dans cette frame.")
                break

            # Inférence
            input_points = torch.from_numpy(points)
            voxels, coords, num_points = voxel_generator(input_points)
            
            coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=coords.dtype), coords], dim=1)
            batch_dict = {
                'voxels': voxels.to(device),
                'voxel_coords': coords.to(device),
                'voxel_num_points': num_points.to(device),
                'batch_size': 1
            }
            
            with torch.no_grad():
                output = model(batch_dict)
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Affichage
            plt.figure(figsize=(10, 10))
            plt.imshow(np.flipud(prob_map), cmap='gray')
            plt.title(f"Segmentation Route - Frame {count}")
            plt.savefig("result_test.png")
            print("SUCCÈS ! Image sauvegardée : result_test.png")
            break
