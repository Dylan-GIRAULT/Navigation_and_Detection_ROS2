import os
import sys
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# --- IMPORTS OPENPCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
from spconv.pytorch.utils import PointToVoxel
from pcdet.models.backbones_2d import BaseBEVBackbone
from pcdet.models.backbones_3d.vfe.pillar_vfe import PillarVFE
from pcdet.models.backbones_2d.map_to_bev.pointpillar_scatter import PointPillarScatter

# ==========================================
# 1. ARCHITECTURE DU MODÈLE
# ==========================================
class PointPillarsRoadSegmentation(nn.Module):
    def __init__(self, model_cfg, num_point_features=4):
        super().__init__()
        self.model_cfg = model_cfg
        self._pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
        _voxel_size = [0.16, 0.16, 4]
        
        self.vfe = PillarVFE(model_cfg=model_cfg.VFE, num_point_features=num_point_features, 
                             point_cloud_range=self._pc_range, voxel_size=_voxel_size)
        self.scatter = PointPillarScatter(model_cfg=model_cfg.MAP_TO_BEV, grid_size=[432, 496, 1])
        self.backbone_2d = BaseBEVBackbone(model_cfg=model_cfg.BACKBONE_2D, 
                                           input_channels=model_cfg.MAP_TO_BEV.NUM_BEV_FEATURES)
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1) 
        )

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        if batch_dict['pillar_features'].ndim == 1:
             batch_dict['pillar_features'] = batch_dict['pillar_features'].unsqueeze(0)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        return self.seg_head(batch_dict['spatial_features_2d'])

# ==========================================
# 2. CONFIGURATION ET DATASET
# ==========================================
DATA_DIR = "/home/younes/Documents/sy27-zoe/workspace/data_extraction/dataset"
CFG_PATH = "/home/younes/Documents/sy27-zoe/workspace/src/perc_ext/perc_ext/cfgs/pointpillar.yaml"
# Le modèle de sauvegarde principal
SAVE_PATH = "/home/younes/Documents/sy27-zoe/workspace/data_extraction/road_model_v3_FROM_SCRATCH.pth"

PCDET_TOOLS = "/home/younes/Documents/OpenPCDet/tools"
old_cwd = os.getcwd()
os.chdir(PCDET_TOOLS)
cfg_from_yaml_file(CFG_PATH, cfg)
os.chdir(old_cwd)

class RoadDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = sorted([f.replace('.png', '') for f in os.listdir(os.path.join(root_dir, 'masks')) if f.endswith('.png')])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        pts = np.load(os.path.join(self.root_dir, 'points', name + '.npy'))
        mask = cv2.imread(os.path.join(self.root_dir, 'masks', name + '.png'), 0)
        
        if pts[:, 3].max() > 1.0: pts[:, 3] /= 255.0
        
        # Transpose indispensable pour aligner le masque avec la sortie réseau
        mask = mask.T 
        target = (mask > 127).astype(np.float32)
        return torch.from_numpy(pts).float(), torch.from_numpy(target).float()

# ==========================================
# 3. ENTRAÎNEMENT
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")
    print(">>> MODE: From Scratch (Poids aléatoires, sans KITTI)")

    voxel_generator = PointToVoxel(
        vsize_xyz=[0.16, 0.16, 4], 
        coors_range_xyz=[0, -39.68, -3, 69.12, 39.68, 1],
        num_point_features=4, max_num_voxels=16000, max_num_points_per_voxel=32,
        device=device
    )

    model = PointPillarsRoadSegmentation(model_cfg=cfg.MODEL).to(device)

    # --- NOTE: CHARGEMENT DESACTIVER POUR REPARTIR DE ZERO ---
    # model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))

    loader = DataLoader(RoadDataset(DATA_DIR), batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    # Augmentation des époques (le départ de zéro est plus lent)
    NUM_EPOCHS = 100 
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for pts, target in loader:
            optimizer.zero_grad()
            batch_voxels, batch_coords, batch_num_points = [], [], []
            for b in range(pts.shape[0]):
                v, c, n = voxel_generator(pts[b].to(device))
                c = torch.cat([torch.full((c.shape[0], 1), b, device=device), c], dim=1)
                batch_voxels.append(v); batch_coords.append(c); batch_num_points.append(n)
            
            batch_dict = {'voxels': torch.cat(batch_voxels), 'voxel_coords': torch.cat(batch_coords), 
                          'voxel_num_points': torch.cat(batch_num_points), 'batch_size': pts.shape[0]}

            output = model(batch_dict).squeeze(1)
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss/len(loader):.4f}")
        
        # Sauvegarde intermédiaire toutes les 20 époques pour comparer
        if (epoch + 1) % 20 == 0:
            temp_path = SAVE_PATH.replace('.pth', f'_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), temp_path)
            print(f">>> Backup sauvegardé : {temp_path}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Entraînement final terminé. Sauvegardé : {SAVE_PATH}")

if __name__ == "__main__":
    train()