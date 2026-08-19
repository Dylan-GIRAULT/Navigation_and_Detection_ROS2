import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2 

# ROS 2 Imports
import rclpy
from rclpy.qos import qos_profile_sensor_data
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
# 1. DÉFINITION DU MODÈLE
# ==========================================
class PointPillarsRoadSegmentation(nn.Module):
    def __init__(self, model_cfg, num_point_features=4):
        super().__init__()
        self.model_cfg = model_cfg
        
        # On garde le range large ici pour ne pas casser les poids
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
# 2. LE NOEUD ROS 2 (VERSION CORRIDOR LARGE Y)
# ==========================================
class RoadSegmentationNode(Node):
    def __init__(self):
        super().__init__('road_segmentation_node')
        
        # --- 1. CHEMINS ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_model = os.path.join(base_dir, 'model', 'road_segmentation_epoch_10.pth')
        default_cfg = os.path.join(base_dir, 'cfgs', 'pointpillar.yaml')

        # --- 2. PARAMETRES ---
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('cfg_path', default_cfg)
        self.declare_parameter('input_topic', '/points')
        self.declare_parameter('robot_height', 1.80)

        model_path = self.get_parameter('model_path').value
        cfg_path = self.get_parameter('cfg_path').value
        input_topic = self.get_parameter('input_topic').value
        self.robot_height = self.get_parameter('robot_height').value

        #self.get_logger().info(f"Modèle : {os.path.basename(model_path)}")

        # --- 3. CHARGEMENT IA ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path, cfg_path, base_dir)
        
        self.voxel_generator = PointToVoxel(
            vsize_xyz=[0.16, 0.16, 4], 
            coors_range_xyz=[0, -39.68, -3, 69.12, 39.68, 1],
            num_point_features=4, max_num_voxels=16000, max_num_points_per_voxel=32
        )

        # --- 4. SETUP ROS ---
        self.bridge = CvBridge()
        self.sub = self.create_subscription(PointCloud2, input_topic, self.lidar_callback, qos_profile_sensor_data)
        self.pub_mask = self.create_publisher(Image, '/perception/road_mask', 10)
        
        #self.get_logger().info(">>> Noeud Prêt (Mode Y=10m / Z Strict)")

    def load_model(self, model_path, cfg_path, base_dir):
        if not os.path.exists(model_path):
            #self.get_logger().error(f"Modèle introuvable : {model_path}")
            sys.exit(1)
        cwd_backup = os.getcwd()
        os.chdir(base_dir) 
        try:
            cfg_from_yaml_file(cfg_path, cfg)
        finally:
            os.chdir(cwd_backup)
        self.model = PointPillarsRoadSegmentation(model_cfg=cfg.MODEL).to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
        except Exception as e:
            #self.get_logger().error(f"Erreur chargement poids: {e}")
            sys.exit(1)

    def pointcloud2_to_numpy(self, msg):
        field_names = [f.name for f in msg.fields]
        if 'intensity' in field_names:
            intensity_name = 'intensity'
        elif 'i' in field_names:
            intensity_name = 'i'
        else:
            intensity_name = field_names[3] if len(field_names) > 3 else None

        try:
            if intensity_name:
                dtype_list = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), (intensity_name, '<f4')]
            else:
                 dtype_list = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]

            current_size = sum([4 for _, _ in dtype_list]) 
            if msg.point_step > current_size:
                dtype_list.append(('padding', 'u1', (msg.point_step - current_size,)))
            
            arr = np.frombuffer(msg.data, dtype=dtype_list)
            res = np.zeros((len(arr), 4), dtype=np.float32)
            res[:, 0] = arr['x']
            res[:, 1] = arr['y']
            res[:, 2] = arr['z']
            if intensity_name:
                res[:, 3] = arr[intensity_name]
            
            return res[~np.isnan(res).any(axis=1)]
            
        except Exception as e:
            #self.get_logger().error(f"Erreur conversion: {e}")
            return None

    def adapt_pandora(self, points):
        # 1. Ajustement Intensité
        if points[:, 3].max() > 1.0:
             points[:, 3] /= 255.0
        
        # 2. Ajustement Hauteur Robot
        delta_z = 1.73 - self.robot_height
        points[:, 2] -= delta_z

        # --- 3. DEFINITION DU CORRIDOR ---
        
        # A. RESTRICTION EN X (Profondeur)
        # On garde 30m ou 40m
        X_MAX = 40.0
        mask_x = (points[:, 0] > 0) & (points[:, 0] < X_MAX)
        points = points[mask_x]

        # B. RESTRICTION EN Y (Largeur) -> ÉLARGIE ICI
        # On passe à 10 mètres (soit 20m de large total)
        Y_LIMIT = 10.0 
        mask_y = (points[:, 1] > -Y_LIMIT) & (points[:, 1] < Y_LIMIT)
        points = points[mask_y]

        # C. RESTRICTION EN Z (Hauteur / Sol)
        # Toujours strict pour la route plate
        Z_MIN = -2.5
        Z_MAX = -1.40 
        mask_z = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
        points = points[mask_z]

        # ---------------------------------------------
        
        return points

    def lidar_callback(self, msg):
        points = self.pointcloud2_to_numpy(msg)
        if points is None or len(points) == 0:
            return 
        
        points = self.adapt_pandora(points)

        if len(points) == 0:
            return

        input_points = torch.from_numpy(points)
        voxels, coords, num_points = self.voxel_generator(input_points)
        
        if voxels.shape[0] == 0: return
        
        coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=coords.dtype), coords], dim=1)
        
        batch_dict = {
            'voxels': voxels.to(self.device),
            'voxel_coords': coords.to(self.device),
            'voxel_num_points': num_points.to(self.device),
            'batch_size': 1
        }

        with torch.no_grad():
            output = self.model(batch_dict)
            TEMPERATURE = 1.0
            output = output / TEMPERATURE
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

        #self.get_logger().info(f"Max Confiance: {prob_map.max():.2f}")

        # --- POST-TRAITEMENT ---
        SEUIL = 0.25  
        prob_map[prob_map <= SEUIL] = 0.0 

        mask_img = (prob_map * 255).astype(np.uint8)

        # Nettoyage minimal
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 5 
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                cv2.drawContours(mask_img, [cnt], -1, 0, -1) 

        mask_img = np.flipud(mask_img)
        
        out_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
        out_msg.header = msg.header
        self.pub_mask.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RoadSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()