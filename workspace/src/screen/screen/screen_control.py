import numpy as np

import shared.resource.quaternion as quaternion

import rclpy
from rclpy.node import Node

from std_msgs.msg import (
    Float64MultiArray
)

from geometry_msgs.msg import (
    Polygon,
    PolygonStamped,
    PoseStamped,
    Point32,
)
from nav_msgs.msg import (
    Odometry,
)
from visualization_msgs.msg import (
    Marker,
    MarkerArray,
)





from shared.resource.car_info import (
    LONGUEUR,
    LARGEUR,
    EMPATTEMENT,
    VOIE_AV,
    VOIE_AR,
    d_arar  ,# distance entre l'arrière de la voiture et le centre des roues arrières, pas indiqué dans wikipédia, à mesurer
    rayon_roue, # rayon des roues, pas indiqué dans wikipédia, à mesurer
    demi_largeur_roue,# demi-largeur des roues, pas indiqué dans wikipédia, à mesurer
    centre_vehicule,
)



class ControlDisplayNode(Node):
    def __init__(self):
        super().__init__('control_display_node')


        self.sub_car = self.create_subscription(Float64MultiArray, "control/controler_info", self.callback_car, 10)
        self.pub_car = self.create_publisher(MarkerArray, "display/car/car", 10)
        self.pub_car_pose = self.create_publisher(PoseStamped, "display/car/pose", 10)
        self.pub_closest_segment = self.create_publisher(Marker, "display/car/closest_segment", 10)
        self.pub_closest_segment2 = self.create_publisher(Marker, "display/car/closest_segment2", 10)


    def callback_car(self, msg):
        """
        Takes the state of the car and publish a MarkerArray to visualize in Rviz

        Parameters
        ----------
        msg: Float64MultiArray
            (x, y, theta, v, delta, closest_segment.x, closest_segment.y) state of the car
        """
        X = msg.data
        id = 0

        # TODO: Utiliser TF2 pour les quaternions
        msg_whole_car = MarkerArray()
        msg_whole_car.markers = []

        car = Marker()
        car.header.stamp = self.get_clock().now().to_msg()
        car.header.frame_id = "world"
        car.id = id; id+=1
        car.type = Marker.CUBE
        car.action = Marker.ADD
        car.scale.x = LONGUEUR
        car.scale.y = LARGEUR
        car.scale.z = 1.5
        car.color.r = 0.0
        car.color.g = 0.0
        car.color.b = 1.0
        car.color.a = 0.5
        car.pose.position.x = X[0]
        car.pose.position.y = X[1]
        car.pose.position.z = 0.0
        q = quaternion.quaternion_from_euler(0,0,X[2])
        car.pose.orientation.x = q.x
        car.pose.orientation.y = q.y
        car.pose.orientation.z = q.z
        car.pose.orientation.w = q.w
        msg_whole_car.markers.append(car)

        car_pos = Marker()
        car_pos.header = car.header
        car_pos.id = id; id+=1
        car_pos.type = Marker.SPHERE
        car_pos.action = Marker.ADD
        car_pos.scale.x = 0.2
        car_pos.scale.y = 0.2
        car_pos.scale.z = 0.2
        car_pos.color.r = 1.0
        car_pos.color.g = 0.0
        car_pos.color.b = 0.0
        car_pos.color.a = 1.0
        car_pos.pose.position.x = X[0]
        car_pos.pose.position.y = X[1]
        car_pos.pose.position.z = 1.5
        car_pos.pose.orientation = car.pose.orientation
        msg_whole_car.markers.append(car_pos)

        essieu_avant_pos= Marker()
        essieu_avant_pos.header = car.header
        essieu_avant_pos.id = id; id+=1
        essieu_avant_pos.type = Marker.SPHERE
        essieu_avant_pos.action = Marker.ADD
        essieu_avant_pos.scale.x = 0.2
        essieu_avant_pos.scale.y = 0.2
        essieu_avant_pos.scale.z = 0.2
        essieu_avant_pos.color.r = 1.0
        essieu_avant_pos.color.g = 0.0
        essieu_avant_pos.color.b = 0.0
        essieu_avant_pos.color.a = 1.0
        essieu_avant_pos.pose.position.x = X[-3]
        essieu_avant_pos.pose.position.y = X[-2]
        essieu_avant_pos.pose.position.z = 1.5
        essieu_avant_pos.pose.orientation = car.pose.orientation
        msg_whole_car.markers.append(essieu_avant_pos)
        

        # Wheels
        Roue=np.array([
            [0],
            [0],
            [1]])
        #transformation homogène 2D pour transformer tous les points du chassis
        #dans le repère de travail
        T=np.array([
            [np.cos(X[2]), -np.sin(X[2]), X[0]],
            [np.sin(X[2]), np.cos(X[2]) , X[1]],
            [0           , 0            , 1]])

        #transformation homogène 2D du repère de la roue AV-D dans RM
        Havd=np.array([
            [np.cos(X[4]),-np.sin(X[4]), EMPATTEMENT-centre_vehicule[0,0]],
            [np.sin(X[4]), np.cos(X[4]),-VOIE_AV/2.0-centre_vehicule[1,0]],
            [0           , 0           , 1]])
        Ravd=T@Havd@Roue #transfos chainées
        wheel_front_right = Marker()
        wheel_front_right.header = car_pos.header
        wheel_front_right.id = id; id+=1
        wheel_front_right.type = Marker.CYLINDER
        wheel_front_right.action = Marker.ADD
        wheel_front_right.scale.x = rayon_roue*2
        wheel_front_right.scale.y = rayon_roue*2
        wheel_front_right.scale.z = demi_largeur_roue*2
        wheel_front_right.color.a = 1.0
        wheel_front_right.color.r = 0.0
        wheel_front_right.color.g = 0.0
        wheel_front_right.color.b = 1.0
        wheel_front_right.pose.position.x = Ravd[0,0]
        wheel_front_right.pose.position.y = Ravd[1,0]
        wheel_front_right.pose.position.z = 0.0
        q = quaternion.quaternion_from_euler(np.pi/2, 0.0, X[2]+X[4])
        wheel_front_right.pose.orientation.x = q.x
        wheel_front_right.pose.orientation.y = q.y
        wheel_front_right.pose.orientation.z = q.z
        wheel_front_right.pose.orientation.w = q.w
        msg_whole_car.markers.append(wheel_front_right)

        #transformation homogène 2D du repère de la roue AV-G dans RM
        Havg=np.array([
            [np.cos(X[4]),-np.sin(X[4]), EMPATTEMENT-centre_vehicule[0,0]],
            [np.sin(X[4]), np.cos(X[4]), VOIE_AV/2.0-centre_vehicule[1,0]],
            [0           , 0           , 1]])
        Ravg=T@Havg@Roue #transfos chainées
        wheel_front_left = Marker()
        wheel_front_left.header = car_pos.header
        wheel_front_left.id = id; id+=1
        wheel_front_left.type = Marker.CYLINDER
        wheel_front_left.action = Marker.ADD
        wheel_front_left.scale.x = rayon_roue*2
        wheel_front_left.scale.y = rayon_roue*2
        wheel_front_left.scale.z = demi_largeur_roue*2
        wheel_front_left.color.a = 1.0
        wheel_front_left.color.r = 0.0
        wheel_front_left.color.g = 0.0
        wheel_front_left.color.b = 1.0
        wheel_front_left.pose.position.x = Ravg[0,0]
        wheel_front_left.pose.position.y = Ravg[1,0]
        wheel_front_left.pose.position.z = 0.0
        q = quaternion.quaternion_from_euler(np.pi/2, 0.0, X[2]+X[4])
        wheel_front_left.pose.orientation.x = q.x
        wheel_front_left.pose.orientation.y = q.y
        wheel_front_left.pose.orientation.z = q.z
        wheel_front_left.pose.orientation.w = q.w
        msg_whole_car.markers.append(wheel_front_left)

        #transformation homogène 2D du repère de la roue AR-D dans RM
        Hard=np.array([[1, 0, -centre_vehicule[0,0]], [0, 1, -VOIE_AR/2.0-centre_vehicule[1,0]], [0, 0, 1]])
        Rard=T@Hard@Roue #transfos chainées
        wheel_back_right = Marker()
        wheel_back_right.header = car_pos.header
        wheel_back_right.id = id; id+=1
        wheel_back_right.type = Marker.CYLINDER
        wheel_back_right.action = Marker.ADD
        wheel_back_right.scale.x = rayon_roue*2
        wheel_back_right.scale.y = rayon_roue*2
        wheel_back_right.scale.z = demi_largeur_roue*2
        wheel_back_right.color.a = 1.0
        wheel_back_right.color.r = 0.0
        wheel_back_right.color.g = 0.0
        wheel_back_right.color.b = 1.0
        wheel_back_right.pose.position.x = Rard[0,0]
        wheel_back_right.pose.position.y = Rard[1,0]
        wheel_back_right.pose.position.z = 0.0
        q = quaternion.quaternion_from_euler(np.pi/2, 0.0, X[2])
        wheel_back_right.pose.orientation.x = q.x
        wheel_back_right.pose.orientation.y = q.y
        wheel_back_right.pose.orientation.z = q.z
        wheel_back_right.pose.orientation.w = q.w
        msg_whole_car.markers.append(wheel_back_right)

        #transformation homogène 2D du repère de la roue AR-G dans RM
        Harg=np.array([[1, 0, -centre_vehicule[0,0]], [0, 1, VOIE_AR/2.0-centre_vehicule[1,0]], [0, 0, 1]])
        Rarg=T@Harg@Roue #transfos chainées
        wheel_back_right = Marker()
        wheel_back_right.header = car_pos.header
        wheel_back_right.id = id; id+=1
        wheel_back_right.type = Marker.CYLINDER
        wheel_back_right.action = Marker.ADD
        wheel_back_right.scale.x = rayon_roue*2
        wheel_back_right.scale.y = rayon_roue*2
        wheel_back_right.scale.z = demi_largeur_roue*2
        wheel_back_right.color.a = 1.0
        wheel_back_right.color.r = 0.0
        wheel_back_right.color.g = 0.0
        wheel_back_right.color.b = 1.0
        wheel_back_right.pose.position.x = Rarg[0,0]
        wheel_back_right.pose.position.y = Rarg[1,0]
        wheel_back_right.pose.position.z = 0.0
        q = quaternion.quaternion_from_euler(np.pi/2, 0.0, X[2])
        wheel_back_right.pose.orientation.x = q.x
        wheel_back_right.pose.orientation.y = q.y
        wheel_back_right.pose.orientation.z = q.z
        wheel_back_right.pose.orientation.w = q.w
        msg_whole_car.markers.append(wheel_back_right)

        self.pub_car.publish(msg_whole_car)

        car_pose = PoseStamped()
        car_pose.header = car.header
        car_pose.pose = car.pose
        car_pose.pose.position.z = 0.5
        car_pose.pose.orientation = car.pose.orientation
        self.pub_car_pose.publish(car_pose)

        closest_segment = Marker()
        closest_segment.header.stamp = self.get_clock().now().to_msg()
        closest_segment.header.frame_id = "world"
        closest_segment.type = Marker.SPHERE
        closest_segment.action = Marker.ADD
        closest_segment.scale.x = 0.7
        closest_segment.scale.y = 0.7
        closest_segment.scale.z = 0.2
        closest_segment.color.r = 0.0
        closest_segment.color.g = 1.0
        closest_segment.color.b = 0.0
        closest_segment.color.a = 0.5
        closest_segment.pose.position.x = X[5]
        closest_segment.pose.position.y = X[6]
        closest_segment.pose.position.z = 0.0
        self.pub_closest_segment.publish(closest_segment)

        closest_segment2 = Marker()
        closest_segment2.header.stamp = self.get_clock().now().to_msg()
        closest_segment2.header.frame_id = "world"
        closest_segment2.type = Marker.SPHERE
        closest_segment2.action = Marker.ADD
        closest_segment2.scale.x = 0.7
        closest_segment2.scale.y = 0.7
        closest_segment2.scale.z = 0.2
        closest_segment2.color.r = 0.5
        closest_segment2.color.g = 0.0
        closest_segment2.color.b = 0.5
        closest_segment2.color.a = 0.5
        closest_segment2.pose.position.x = X[7]
        closest_segment2.pose.position.y = X[8]
        closest_segment2.pose.position.z = 0.0
        self.pub_closest_segment2.publish(closest_segment2)


def main(args=None):
    rclpy.init(args=args)
    node = ControlDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
