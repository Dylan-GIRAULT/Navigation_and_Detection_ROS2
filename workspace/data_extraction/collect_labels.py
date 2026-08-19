import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import os
from sensor_msgs_py import point_cloud2

class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')
        
        # Souscriptions
        self.sub_road = self.create_subscription(PointCloud2, '/road', self.road_callback, 10)
        self.sub_nonroad = self.create_subscription(PointCloud2, '/nonroad', self.nonroad_callback, 10)
        
        self.road_buffer = None
        self.output_dir = "dataset_npy"
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_count = 0
        
        #self.get_logger().info("En attente des données /road et /nonroad...")

    def pc2_to_numpy(self, msg):
        try:
            # On récupère les données brutes
            # Votre point_step est de 32 octets
            point_step = msg.point_step
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)

            # Extraction précise selon vos offsets : x:0, y:4, z:8, intensity:16
            x = data[:, 0:4].view('<f4')
            y = data[:, 4:8].view('<f4')
            z = data[:, 8:12].view('<f4')
            
            # On prend l'intensité à l'offset 16 (1 octet) et on la normalise
            intensity = data[:, 16:17].astype(np.float32) / 255.0

            # On réunit le tout en [N, 4]
            points = np.hstack([x, y, z, intensity])
            return points
            
        except Exception as e:
            #self.get_logger().error(f"Erreur décodage : {e}")
            return np.zeros((0, 4), dtype=np.float32)

    def road_callback(self, msg):
        self.road_buffer = self.pc2_to_numpy(msg)

    def nonroad_callback(self, msg):
        if self.road_buffer is None:
            return
            
        nonroad_pts = self.pc2_to_numpy(msg)
        
        # Création des labels (1 pour route, 0 pour non-route)
        road_labels = np.ones((len(self.road_buffer), 1), dtype=np.float32)
        nonroad_labels = np.zeros((len(nonroad_pts), 1), dtype=np.float32)
        
        # Fusion des points [N, 4] et des labels [N, 1]
        road_data = np.hstack([self.road_buffer, road_labels])
        nonroad_data = np.hstack([nonroad_pts, nonroad_labels])
        
        # Dataset final [N, 5] (x, y, z, intensity, label)
        final_frame = np.vstack([road_data, nonroad_data])
        
        # Sauvegarde
        filename = os.path.join(self.output_dir, f"frame_{self.frame_count:05d}.npy")
        np.save(filename, final_frame)
        
        #self.get_logger().info(f"Frame {self.frame_count} sauvegardée ({len(final_frame)} points)")
        self.frame_count += 1
        self.road_buffer = None # Reset pour synchronisation simple

def main():
    rclpy.init()
    node = DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt de l'extraction.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

