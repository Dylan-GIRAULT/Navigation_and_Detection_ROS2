import os
import sys
import numpy as np
import torch
import torch.nn as nn

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
# 2. LE NOEUD ROS 2 (VERSION DEBUG)
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
        
        # Générateur de Voxels
        self.voxel_generator = PointToVoxel(
            vsize_xyz=[0.16, 0.16, 4], 
            coors_range_xyz=[0, -39.68, -3, 69.12, 39.68, 1],
            num_point_features=4, max_num_voxels=16000, max_num_points_per_voxel=32
        )

        # --- 4. SETUP ROS (QoS Sensor Data pour Pandora) ---
        self.bridge = CvBridge()
        self.sub = self.create_subscription(PointCloud2, input_topic, self.lidar_callback, qos_profile_sensor_data)
        self.pub_mask = self.create_publisher(Image, '/perception/road_mask', 10)
        
        #self.get_logger().info(">>> Noeud Prêt (Mode DEBUG) ! En attente de points...")

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
        """
        Conversion robuste avec logs d'erreurs
        """
        # Liste des champs disponibles dans le message
        field_names = [f.name for f in msg.fields]
        
        # Choix dynamique : 'intensity' (Pandora/Hesai) ou 'i' (OpenPCDet standard)
        if 'intensity' in field_names:
            intensity_name = 'intensity'
        elif 'i' in field_names:
            intensity_name = 'i'
        else:
            # Si aucun des deux, on prend le 4ème champ au hasard ou on met 0
            #self.get_logger().warn(f"Pas de champ intensité trouvé ! Champs dispos: {field_names}")
            intensity_name = field_names[3] if len(field_names) > 3 else None

        try:
            # 1. Définition de la structure attendue
            if intensity_name:
                dtype_list = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), (intensity_name, '<f4')]
            else:
                 dtype_list = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]

            # 2. Gestion du Padding (Octets vides envoyés par le LiDAR)
            current_size = sum([4 for _, _ in dtype_list]) 
            if msg.point_step > current_size:
                dtype_list.append(('padding', 'u1', (msg.point_step - current_size,)))
            
            # 3. Lecture brute du buffer
            arr = np.frombuffer(msg.data, dtype=dtype_list)
            
            # 4. Conversion Numpy
            res = np.zeros((len(arr), 4), dtype=np.float32)
            res[:, 0] = arr['x']
            res[:, 1] = arr['y']
            res[:, 2] = arr['z']
            if intensity_name:
                res[:, 3] = arr[intensity_name]
            
            return res[~np.isnan(res).any(axis=1)]
            
        except Exception as e:
            #self.get_logger().error(f"!!! CRASH CONVERSION !!!")
            #self.get_logger().error(f"Erreur: {e}")
            #self.get_logger().error(f"PointStep: {msg.point_step}, Champs: {field_names}")
            return None

    def adapt_pandora(self, points):
        if points[:, 3].max() > 1.0:
            points[:, 3] /= 255.0
        delta_z = 1.73 - self.robot_height
        points[:, 2] -= delta_z
        return points

    def lidar_callback(self, msg):
        # --- DEBUG 1 : ON A REÇU QUELQUE CHOSE ---
        # #self.get_logger().info(f"Callback déclenché ! FrameID: {msg.header.frame_id}")

        # 1. Conversion
        points = self.pointcloud2_to_numpy(msg)
        if points is None or len(points) == 0:
            return # L'erreur a déjà été logguée dans pointcloud2_to_numpy
        
        # 2. Adaptation
        points = self.adapt_pandora(points)

        # 3. Voxelisation
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

        # 4. Inférence
        with torch.no_grad():
            output = self.model(batch_dict)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

        #self.get_logger().info(f"Inférence OK. Prob Max: {prob_map.max():.2f}")
        #self.get_logger().info(f"Prob Min: {prob_map.min():.2f}")

        # --- MODIFICATION ICI : FILTRE DE CONFIANCE ---
        SEUIL = 0.5 # Essayez 0.5, 0.6 ou 0.7
        prob_map[prob_map < SEUIL] = 0.0  # On met à noir tout ce qui est incertain
        # ---------------------------------------------

        # 5. Publication
        mask_img = (prob_map * 255).astype(np.uint8)
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