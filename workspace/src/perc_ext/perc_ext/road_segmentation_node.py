#!/home/younes/pcdet_env/bin/python3
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2 

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge

# Hack numpy pour compatibilité OpenPCDet
np.bool = np.bool_

# --- IMPORTS OPENPCDET ---
from spconv.pytorch.utils import PointToVoxel
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
# 2. LE NOEUD ROS 2
# ==========================================
class RoadSegmentationNode(Node):
    def __init__(self):
        super().__init__('road_segmentation_node')
        
        base_dir = "/home/younes/Documents/sy27-zoe/workspace/src/perc_ext/perc_ext"
        default_model = "/home/younes/Documents/sy27-zoe/workspace/data_extraction/road_model_v3_final.pth"
        default_cfg = os.path.join(base_dir, 'cfgs', 'pointpillar.yaml')

        self.declare_parameter('model_path', default_model)
        self.declare_parameter('cfg_path', default_cfg)
        self.declare_parameter('input_topic', '/points')
        self.declare_parameter('threshold', 0.92) 

        # PARAMÈTRES DU MASQUE
        self.declare_parameter('exclusion_base', 500)
        self.declare_parameter('exclusion_dist', 200)
        self.declare_parameter('corridor_length', 220)
        self.declare_parameter('corridor_width', 60)

        self.model_path = self.get_parameter('model_path').value
        self.cfg_path = self.get_parameter('cfg_path').value
        self.input_topic = self.get_parameter('input_topic').value

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(self.model_path, self.cfg_path)
        
        self.voxel_generator = PointToVoxel(
            vsize_xyz=[0.16, 0.16, 4], 
            coors_range_xyz=[0, -39.68, -3, 69.12, 39.68, 1],
            num_point_features=4, max_num_voxels=16000, max_num_points_per_voxel=32,
            device=self.device
        )

        self.bridge = CvBridge()
        
        # MODIFICATION : Utilisation d'une QoS par défaut (10) pour maximiser la compatibilité
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.lidar_callback, 10)
        self.pub_mask = self.create_publisher(Image, '/perception/road_mask', 10)
        
        #self.get_logger().info(f">>> Node Initialisé sur {self.input_topic} (QoS Compatible)")

    def load_model(self, model_path, cfg_path):
        PCDET_TOOLS = "/home/younes/Documents/OpenPCDet/tools"
        cwd_backup = os.getcwd()
        os.chdir(PCDET_TOOLS)
        try:
            cfg_from_yaml_file(cfg_path, cfg)
        finally:
            os.chdir(cwd_backup)
        self.model = PointPillarsRoadSegmentation(model_cfg=cfg.MODEL).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_combined_mask(self, h, w):
        mask = np.ones((h, w), dtype=np.uint8) * 255
        base = self.get_parameter('exclusion_base').value
        dist = self.get_parameter('exclusion_dist').value
        corr_len = self.get_parameter('corridor_length').value
        corr_width = self.get_parameter('corridor_width').value
        
        center_y, center_x = h // 2, w // 2

        pts_left = np.array([[0, center_y - base//2], [0, center_y + base//2], [dist, center_y]], np.int32)
        pts_right = np.array([[w, center_y - base//2], [w, center_y + base//2], [w - dist, center_y]], np.int32)
        cv2.fillPoly(mask, [pts_left], 0)
        cv2.fillPoly(mask, [pts_right], 0)

        cv2.rectangle(mask, (0, h - corr_len), (center_x - corr_width//2, h), 0, -1)
        cv2.rectangle(mask, (center_x + corr_width//2, h - corr_len), (w, h), 0, -1)
        return mask

    def pointcloud2_to_numpy(self, msg):
        try:
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)
            return arr[:, :4]
        except Exception: return None

    def lidar_callback(self, msg):
        #self.get_logger().info("--- Nouveau message Lidar reçu ---")
        
        points = self.pointcloud2_to_numpy(msg)
        if points is None or len(points) == 0:
            #self.get_logger().warn("⚠️ Erreur conversion numpy (message vide ?)")
            return 

        if points[:, 3].max() > 1.0: points[:, 3] /= 255.0

        mask_z = (points[:, 2] > -2.5) & (points[:, 2] < 0.5)
        points = points[mask_z]
        if len(points) < 50:
            #self.get_logger().warn(f"⚠️ Trop peu de points ({len(points)})")
            return

        # Recentrage
        mean_x, mean_y = points[:, 0].mean(), points[:, 1].mean()
        pts_transformed = points.copy()
        pts_transformed[:, 0] = points[:, 0] - mean_x + 35.0 
        pts_transformed[:, 1] = points[:, 1] - mean_y + 0.0

        # Inférence
        input_pts = torch.from_numpy(pts_transformed).to(self.device)
        voxels, coords, num_points = self.voxel_generator(input_pts)
        if voxels.shape[0] == 0:
            #self.get_logger().warn("⚠️ Voxelisation vide")
            return
        
        coords = torch.cat([torch.zeros((coords.shape[0], 1), device=self.device), coords], dim=1)
        batch_dict = {'voxels': voxels, 'voxel_coords': coords, 'voxel_num_points': num_points, 'batch_size': 1}

        with torch.no_grad():
            output = self.model(batch_dict)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

        # Post-traitement
        prob_map = prob_map.T 
        mask_binary = (prob_map > self.get_parameter('threshold').value).astype(np.uint8) * 255
        
        kernel = np.ones((2,2), np.uint8)
        mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        mask_smooth = cv2.GaussianBlur(mask_clean, (3, 3), 0)
        _, mask_final = cv2.threshold(mask_smooth, 128, 255, cv2.THRESH_BINARY)
        mask_final = np.flipud(np.fliplr(mask_final))

        # Masque combiné
        h, w = mask_final.shape
        mask_final = cv2.bitwise_and(mask_final, self.get_combined_mask(h, w))
        
        # Publication
        out_msg = self.bridge.cv2_to_imgmsg(mask_final, encoding="mono8")
        out_msg.header = msg.header
        self.pub_mask.publish(out_msg)
        #self.get_logger().info("✅ Masque publié.")

def main(args=None):
    rclpy.init(args=args)
    node = RoadSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()