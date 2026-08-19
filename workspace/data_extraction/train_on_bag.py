import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from spconv.pytorch.utils import PointToVoxel

# --- IMPORTS OPENPCDET ---
from pcdet.config import cfg, cfg_from_yaml_file
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
        # Configuration standard de PointPillars
        self._pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
        _voxel_size = [0.16, 0.16, 4]
        _grid_size = [432, 496, 1]
        
        self.vfe = PillarVFE(model_cfg=model_cfg.VFE, num_point_features=num_point_features, 
                             point_cloud_range=self._pc_range, voxel_size=_voxel_size)
        self.scatter = PointPillarScatter(model_cfg=model_cfg.MAP_TO_BEV, grid_size=_grid_size)
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

class BagDataset(Dataset):
    def __init__(self, npy_dir):
        self.files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')]
        print(f"[DATASET] {len(self.files)} frames chargées.")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return torch.from_numpy(data[:, :4]).float(), torch.from_numpy(data[:, 4]).float()

# ==========================================
# 3. MAIN TRAINING SCRIPT
# ==========================================
def main():
    PCDET_ROOT = "/home/younes/Documents/OpenPCDet/tools" 
    BASE_WS = "/home/younes/Documents/sy27-zoe/workspace"
    NPY_DIR = os.path.join(BASE_WS, "data_extraction/dataset_npy")
    MODEL_PATH = os.path.join(BASE_WS, "install/perc_ext/share/perc_ext/model/pandaset_final.pth")
    CFG_PATH = os.path.join(BASE_WS, "src/perc_ext/perc_ext/cfgs/pointpillar.yaml")
    SAVE_PATH = os.path.join(BASE_WS, "data_extraction/pandaset_tuned_bag.pth")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Entraînement sur : {DEVICE}")

    os.chdir(PCDET_ROOT)
    try:
        cfg_from_yaml_file(CFG_PATH, cfg)
    except Exception as e:
        print(f"[ERREUR] Échec du chargement : {e}")
        return

    voxel_generator = PointToVoxel(
        vsize_xyz=[0.16, 0.16, 4], 
        coors_range_xyz=[0, -39.68, -3, 69.12, 39.68, 1],
        num_point_features=4, 
        max_num_voxels=16000, 
        max_num_points_per_voxel=32,
        device=DEVICE  
    )

    model = PointPillarsRoadSegmentation(model_cfg=cfg.MODEL).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    dataset = BagDataset(NPY_DIR)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = optim.Adam(model.seg_head.parameters(), lr=0.0005)
    pos_weight = torch.tensor([30.0]).to(DEVICE) # Poids élevé pour compenser le déséquilibre
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    print("Début du Fine-Tuning...")
    for epoch in range(5):
        epoch_loss = 0
        for i, (pts, lbls) in enumerate(train_loader):
            pts, lbls = pts.squeeze(0).to(DEVICE), lbls.squeeze(0).to(DEVICE)
            
            # --- TRANSFORMATION : RECENTRAGE DYNAMIQUE ---
            road_mask = (lbls == 1)
            pts_transformed = pts.clone()
            
            if road_mask.any():
                # On calcule le centre de la route actuelle
                mean_x = pts[road_mask, 0].mean()
                mean_y = pts[road_mask, 1].mean()
                
                # On téléporte les points pour que la route soit à X=25m, Y=0m
                # Cela garantit que les points rentrent dans PC_RANGE [0, 69] et [-39, 39]
                pts_transformed[:, 0] = pts[:, 0] - mean_x + 25.0
                pts_transformed[:, 1] = pts[:, 1] - mean_y + 0.0
            
            # Voxelization
            voxels, coords, num_points = voxel_generator(pts_transformed)
            batch_idx = torch.zeros((coords.shape[0], 1), device=DEVICE)
            coords_with_batch = torch.cat([batch_idx, coords], dim=1)
            
            optimizer.zero_grad()
            logits = model({'voxels': voxels, 'voxel_coords': coords_with_batch, 'voxel_num_points': num_points, 'batch_size': 1})
            
            with torch.no_grad():
                target = torch.zeros((1, 1, 496, 432), device=DEVICE)
                # On utilise road_mask sur les points transformés
                road_pts_transformed = pts_transformed[road_mask]
                
                if len(road_pts_transformed) > 0:
                    x_idxs = ((road_pts_transformed[:, 0] - 0) / 0.16).long()
                    y_idxs = ((road_pts_transformed[:, 1] - (-39.68)) / 0.16).long()
                    
                    valid = (x_idxs >= 0) & (x_idxs < 432) & (y_idxs >= 0) & (y_idxs < 496)
                    if valid.any():
                        target[0, 0, y_idxs[valid], x_idxs[valid]] = 1.0

            # Diagnostic
            if i % 100 == 0:
                print(f"[DIAG] Frame {i}: {int(target.sum())} pixels route dans Target")

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n[SUCCÈS] Modèle sauvegardé : {SAVE_PATH}")

if __name__ == "__main__":
    main()