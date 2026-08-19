import rclpy
from rclpy.node import Node

import math
import numpy as np

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, Point32, PolygonStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from pyproj import Transformer, CRS

from localisation import matching
"""
Copied from:
https://gist.github.com/salmagro/2e698ad4fbf9dae40244769c5ab74434#file-euler_from_quaternion-py
"""


class Quaternion:
    w: float
    x: float
    y: float
    z: float


def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cy * cp * cr + sy * sp * sr
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    return q


def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in **first** place) to euler roll, pitch, yaw
    quaternion = [w, x, y, z]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')

        # ===============================
        # Référence ENU
        # ===============================
        self.lat_ref = 49.401322
        self.lon_ref = 2.797066
        self.alt_ref = 80.5

        self.crs_wgs84 = CRS.from_epsg(4979)   # lat, lon, alt
        self.crs_ecef = CRS.from_epsg(4978)    # ECEF XYZ

        self.tf_wgs84_to_ecef = Transformer.from_crs(
            self.crs_wgs84, self.crs_ecef, always_xy=True
        )

        self.ref_x, self.ref_y, self.ref_z = self.tf_wgs84_to_ecef.transform(
            self.lon_ref, self.lat_ref, self.alt_ref
        )

        # ===============================
        # Subscribers
        # ===============================
        self.create_subscription(
            Image,
            '/perception/road_mask',
            self.image_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/septentrio/odom_geo',
            self.septentrio_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/span/odom',
            self.span_callback,
            10
        )

        self.get_logger().info("ICP Node démarré")

        # ===============================
        # Publishers
        # ===============================
        self.icp_pose_pub = self.create_publisher(Float64MultiArray, '/localisation/icp_pose', 10)
        self.span_path_pub = self.create_publisher(Path, '/span/path', 10)
        self.icp_path_pub = self.create_publisher(Path, '/icp/path', 10)
        self.error_marker_pub = self.create_publisher(Marker, '/localisation/error', 10)

        # RViz: voitures
        self.icp_car_pub = self.create_publisher(MarkerArray, '/display/car/icp_car', 10)
        self.septentrio_car_pub = self.create_publisher(MarkerArray, '/display/car/septentrio_car', 10)
        self.span_car_pub = self.create_publisher(MarkerArray, '/display/car/span_car', 10)

        # ===============================
        # États internes
        # ===============================
        self.image = None
        self.prev_gnss = None                 # np.array([x, y])
        self.enu_pose = None                  # (x, y, theta)
        self.latest_span_pose = None           # (x, y)
        self.last_sept_msg = None
        self.last_span_msg = None
        self.max_time_diff = 0.05  # secondes (50 ms)
        self.local_segments = None
        self.last_icp_pose = None
        self.last_icp_time = None
        # ===============================
        # Carte
        # ===============================
        self.circuit_points = matching.load_circuit_points()
        self.circuit_segments = matching.build_segments(self.circuit_points)

        # RViz: carte
        self.map_pub = self.create_publisher(PolygonStamped, '/display/map/full', 10)
        self.publish_map()

        # ===============================
        # RViz paths
        # ===============================
        self.span_path = Path()
        self.span_path.header.frame_id = 'map'

        self.icp_path = Path()
        self.icp_path.header.frame_id = 'map'

        # ===============================
        # Timer ICP
        # ===============================
        self.create_timer(0.1, self.run_icp)

    # =====================================================
    # Callbacks
    # =====================================================
    def image_callback(self, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8)
        self.image = img.reshape((msg.height, msg.width))

    def septentrio_callback(self, msg: Odometry):
        lat = msg.pose.pose.position.x
        lon = msg.pose.pose.position.y
        alt = msg.pose.pose.position.z

        x_ecef, y_ecef, z_ecef = self.tf_wgs84_to_ecef.transform(lon, lat, alt)

        lat0 = math.radians(self.lat_ref)
        lon0 = math.radians(self.lon_ref)

        dx = x_ecef - self.ref_x
        dy = y_ecef - self.ref_y
        dz = z_ecef - self.ref_z

        x_enu = -math.sin(lon0) * dx + math.cos(lon0) * dy
        y_enu = (
            -math.sin(lat0) * math.cos(lon0) * dx
            - math.sin(lat0) * math.sin(lon0) * dy
            + math.cos(lat0) * dz
        )

        curr = np.array([x_enu, y_enu])

        if self.prev_gnss is not None:
            delta = curr - self.prev_gnss
            if np.linalg.norm(delta) > 1e-3:
                theta = math.atan2(delta[1], delta[0])
            else:
                theta = None
        else:
            theta = None

        self.prev_gnss = curr
        self.enu_pose = (curr[0], curr[1], theta)
        self.last_sept_msg = msg
        self.local_segments = self.get_local_segments(np.array([x_enu, y_enu]))

    def span_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.latest_span_pose = (x, y)

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0

        self.span_path.poses.append(pose)
        self.span_path.header.stamp = pose.header.stamp
        self.span_path_pub.publish(self.span_path)
        self.last_span_msg = msg

    # =====================================================
    # ICP Loop
    # =====================================================
    def msgs_are_synced(self):
        if self.last_sept_msg is None or self.last_span_msg is None:
            return False
        t1 = self.last_sept_msg.header.stamp
        t2 = self.last_span_msg.header.stamp
        dt = abs((t1.sec + t1.nanosec*1e-9) - (t2.sec + t2.nanosec*1e-9))
        return dt < self.max_time_diff

    def image_to_points(self, image, threshold=200):
        ys, xs = np.where(image > threshold)
        pts = np.column_stack((xs, -ys)).astype(np.float64)
        pts -= np.mean(pts, axis=0)
        return pts

    def get_local_segments(self, gnss_xy, max_dist=5.0):
        selected = []
        for seg in self.circuit_segments:
            p1, p2 = seg["p1"], seg["p2"]
            ab = p2 - p1
            t = np.clip(np.dot(gnss_xy - p1, ab) / max(0.001, np.dot(ab, ab)), 0.0, 1.0)
            proj = p1 + t * ab
            if np.linalg.norm(gnss_xy - proj) < max_dist:
                selected.append(seg)
        return selected

    # -------------------------------
    # Affichage RViz
    # -------------------------------
    def publish_map(self):
        segments = self.circuit_segments
        poly = PolygonStamped()
        poly.header.frame_id = "map"
        poly.header.stamp = self.get_clock().now().to_msg()
        poly.polygon.points = []
        for seg in segments:
            for pt in [seg["p1"], seg["p2"]]:
                p = Point32()
                p.x = pt[0]
                p.y = pt[1]
                p.z = 0.0
                poly.polygon.points.append(p)
        self.map_pub.publish(poly)

    def publish_car_marker(self, x, y, theta, pub, ns="car", color=(0,0,1)):
        marker_array = MarkerArray()
        car = Marker()
        car.header.frame_id = "map"
        car.header.stamp = self.get_clock().now().to_msg()
        car.id = 0
        car.ns = ns
        car.type = Marker.CUBE
        car.action = Marker.ADD
        car.scale.x = 4.0
        car.scale.y = 2.0
        car.scale.z = 1.5
        car.color.r = color[0]
        car.color.g = color[1]
        car.color.b = color[2]
        car.color.a = 0.5
        car.pose.position.x = x
        car.pose.position.y = y
        car.pose.position.z = 0.0
        q = quaternion_from_euler(0,0,theta)
        car.pose.orientation.x = q.x
        car.pose.orientation.y = q.y
        car.pose.orientation.z = q.z
        car.pose.orientation.w = q.w
        marker_array.markers.append(car)
        pub.publish(marker_array)

    # -------------------------------
    # Boucle principale ICP
    # -------------------------------
    def run_icp(self):
        current_time = self.get_clock().now().to_msg()
        if self.image is None or self.enu_pose is None or self.latest_span_pose is None:
            return
        if not self.msgs_are_synced():
            return

        source_points = self.image_to_points(self.image)
        if source_points.shape[0] < 20:
            return

        x_gnss, y_gnss, theta_init = self.enu_pose
        local_segments = self.local_segments

        x_icp, y_icp, theta_icp = matching.icp(source_points, x_gnss, y_gnss, theta_init, local_segments)
        if x_icp is None:
            return

        if self.last_icp_pose is None and self.last_icp_time is None:
            vx=0.0
            vy=0.0
            speed = 0.0
        else:
            dt = (current_time.nanoseconds - self.last_icp_time.nanoseconds) * 1e-9
            dt = max(dt, 1e-6)
            dx = x_icp - self.last_icp_pose[0]
            dy = y_icp - self.last_icp_pose[1]
            vx = dx / dt
            vy = dy / dt
            speed = math.sqrt(vx**2 + vy**2)
        self.last_icp_pose = (x_icp, y_icp)
        self.last_icp_time = current_time
        # Publier ICP
        msg = Float64MultiArray()
        msg.data = [float(x_icp), float(y_icp), float(theta_icp), float(speed)]
        self.icp_pose_pub.publish(msg)

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x_icp
        pose.pose.position.y = y_icp
        pose.pose.orientation.w = 1.0
        self.icp_path.poses.append(pose)
        self.icp_path.header.stamp = pose.header.stamp
        self.icp_path_pub.publish(self.icp_path)

        x_span, y_span = self.latest_span_pose
        theta_span = 0.0  

        # Affichage voitures + ligne d'erreur
        self.publish_car_marker(x_icp, y_icp, theta_icp, self.icp_car_pub, ns="icp", color=(0,0,1))
        self.publish_car_marker(x_gnss, y_gnss, theta_init, self.septentrio_car_pub, ns="septentrio", color=(0,1,0))
        self.publish_car_marker(x_span, y_span, theta_span, self.span_car_pub, ns="span", color=(1,0,0))

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "icp_span_error"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color.r = 1.0
        marker.color.a = 1.0
        p1 = Point(x=x_span, y=y_span, z=0.0)
        p2 = Point(x=x_icp, y=y_icp, z=0.0)
        marker.points = [p1, p2]
        self.error_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = ICPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# Dernière nouvel version my beloved