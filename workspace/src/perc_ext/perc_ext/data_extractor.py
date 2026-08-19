#!/home/younes/pcdet_env/bin/python3
import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import message_filters # Pour synchroniser les deux topics

class DataExtractorNode(Node):
    def __init__(self):
        super().__init__('data_extractor_node')
        
        # Configuration des dossiers
        self.save_path = "/home/younes/Documents/sy27-zoe/workspace/data_extraction/dataset"
        os.makedirs(os.path.join(self.save_path, 'points'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'masks'), exist_ok=True)

        # Synchronisation des topics /road et /nonroad
        self.sub_road = message_filters.Subscriber(self, PointCloud2, '/road')
        self.sub_nonroad = message_filters.Subscriber(self, PointCloud2, '/nonroad')
        
        # On autorise un léger décalage (slop) entre les deux messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_road, self.sub_nonroad], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

        self.frame_idx = 0
        #self.get_logger().info(f"Extraction prête. Sauvegarde dans : {self.save_path}")

    def pc2_to_numpy(self, msg):
        # Conversion simple du buffer PointCloud2 en float32
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)
        return arr[:, :4] # On garde X, Y, Z, Intensité

    def callback(self, road_msg, nonroad_msg):
        # 1. Conversion
        road_pts = self.pc2_to_numpy(road_msg)
        nonroad_pts = self.pc2_to_numpy(nonroad_msg)

        if len(road_pts) == 0:
            return

        # --- CORRECTION : RECENTRAGE DYNAMIQUE ---
        # On calcule le centre pour que la route ne soit pas hors-champ
        mean_x, mean_y = points[:, 0].mean(), points[:, 1].mean()

        # On décale TOUS les points pour qu'ils rentrent dans la zone [0, 69.12]
        # On centre à 25m devant pour simuler la vue de l'IA
        def center_cloud(pts):
            new_pts = pts.copy()
            new_pts[:, 0] = pts[:, 0] - mean_x + 25.0
            new_pts[:, 1] = pts[:, 1] - mean_y + 0.0
            return new_pts

        road_pts_centered = center_cloud(road_pts)
        nonroad_pts_centered = center_cloud(nonroad_pts)
        all_points_centered = np.concatenate([road_pts_centered, nonroad_pts_centered], axis=0)

        # 2. Paramètres de grille
        pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
        voxel_size = [0.16, 0.16]
        grid_size = [432, 496]

        mask = np.zeros((grid_size[0], grid_size[1]), dtype=np.uint8)

        # 3. Projection sur le masque (Target)
        res_x = (road_pts_centered[:, 0] - pc_range[0]) / voxel_size[0]
        res_y = (road_pts_centered[:, 1] - pc_range[1]) / voxel_size[1]

        valid = (res_x >= 0) & (res_x < grid_size[0]) & (res_y >= 0) & (res_y < grid_size[1])
        
        if valid.any():
            ix = res_x[valid].astype(np.int32)
            iy = res_y[valid].astype(np.int32)
            mask[ix, iy] = 255

            # --- ÉPAISSIR LE MASQUE ---
            # On utilise un kernel plus gros (5x5) pour être sûr de voir la route
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

        # 4. Sauvegarde
        np.save(os.path.join(self.save_path, 'points', f"{self.frame_idx:04d}.npy"), all_points_centered)
        cv2.imwrite(os.path.join(self.save_path, 'masks', f"{self.frame_idx:04d}.png"), mask)

        #self.get_logger().info(f"Frame {self.frame_idx} : {len(road_pts)} points road détectés.")
        self.frame_idx += 1

def main():
    rclpy.init()
    node = DataExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
