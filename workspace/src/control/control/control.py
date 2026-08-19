import numpy as np

import control.map_utils as map_utils
from control.controler import Controler
import shared.resource.quaternion as quaternion

import rclpy
from rclpy.node import Node

from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import (
    Float64MultiArray,
    Float32,
    UInt32,
)
from geometry_msgs.msg import (
    PolygonStamped,
    Point32,
)
from nav_msgs.msg import (
    Odometry,
)



class ControlNode(Node):
    def __init__(self, controler: Controler):
        """Interface between Controler class and ROS"""
        super().__init__('control_node')
        self.controler = controler

        self.declare_parameter("offline", True)
        self.offline = self.get_parameter("offline").value
        self.declare_parameter("use_cheat_map", False)
        self.use_cheat_map = self.get_parameter("use_cheat_map").value

        self.declare_parameter("control_only", False)
        self.control_only = self.get_parameter("control_only").value
        if self.control_only:
            self.set_map_from_segment(map_utils.LIGNE_DROITE)

        self.declare_parameter("gain_proportionnel", self.controler.gain_proportionnel)
        self.declare_parameter("gain_integrale", self.controler.gain_integrale)
        self.declare_parameter("integral_max", self.controler.integral_max)
        self.declare_parameter("gain_ecart", self.controler.gain_ecart)
        self.declare_parameter("gain_pente", self.controler.gain_pente)
        self.declare_parameter("gain_courbure", self.controler.gain_courbure)
        self.declare_parameter("a_lat_max", self.controler.a_lat_max)
        self.declare_parameter("front_axle_distance", self.controler.front_axle_distance)
        self.declare_parameter("delta_speed_factor", self.controler.delta_speed_factor)
        self.wheel_angle_factor = 15.0
        self.acceleration_factor = 6.0
        self.negative_acceleration_factor = 20.0
        self.declare_parameter("wheel_angle_factor", self.wheel_angle_factor)
        self.declare_parameter("acceleration_factor", self.acceleration_factor)
        self.declare_parameter("negative_acceleration_factor", self.negative_acceleration_factor)
        # Set these in RQT: Plugins -> Configuration -> Dynamic Reconfigure, or in params.yaml
        self.add_on_set_parameters_callback(self.callback_params)

        # self.sub_span = self.create_subscription(Odometry, "/span/odom", self.callback_span, 10)
        self.last_header_time = None

        self.sub_nav_segment = self.create_subscription(UInt32, "/navigation_to_control_segment", self.callback_segment, 10)
        self.sub_nav_position = self.create_subscription(Float64MultiArray, "/navigation_to_control_position", self.callback_position, 10)

        self.pub_acceleration = self.create_publisher(Float32, "/mabx/torque_setpoint", 10)
        self.pub_steering_wheel_angle = self.create_publisher(Float32, "/mabx/steering_angle_setpoint", 10)

        self.pub_map = self.create_publisher(PolygonStamped, "/control_map", 10)
        self.pub_controler_info = self.create_publisher(Float64MultiArray, "/control/controler_info", 10)

        self.is_stoping = False

    def callback_params(self, params):
        for param in params:
            if param.name == "gain_proportionnel":
                self.controler.gain_proportionnel = param.value
            elif param.name == "gain_integrale":
                self.controler.gain_integrale = param.value
            elif param.name == "integral_max":
                self.controler.integral_max = param.value
            elif param.name == "gain_ecart":
                self.controler.gain_ecart = param.value
            elif param.name == "gain_pente":
                self.controler.gain_pente = param.value
            elif param.name == "gain_courbure":
                self.controler.gain_courbure = param.value
            elif param.name == "MAX_WHEEL_ANGLE":
                self.controler.MAX_WHEEL_ANGLE = param.value

            elif param.name == "wheel_angle_factor":
                self.wheel_angle_factor = param.value
            elif param.name == "acceleration_factor":
                self.acceleration_factor = param.value
            elif param.name == "negative_acceleration_factor":
                self.negative_acceleration_factor = param.value
            elif param.name == "a_lat_max":
                self.controler.a_lat_max = param.value
            elif param.name == "front_axle_distance":
                self.controler.front_axle_distance = param.value
            elif param.name == "delta_speed_factor":
                self.controler.delta_speed_factor = param.value

            elif param.name == "use_cheat_map":
                self.use_cheat_map = param.value

        return SetParametersResult(successful=True)


    def callback_segment(self, msg):
        #self.get_logger().warn(f"--- Callback segment: {msg.data} ---")
        self.set_map_from_segment(msg.data)


    def set_map_from_segment(self, segment):
        #self.get_logger().warn(f"--- Set map from segment: {segment} ---")
        try:
            map = map_utils.segment_to_map(segment, self.use_cheat_map)
            self.controler.set_map_to_follow(map)
            self.start()
        except:
            self.stop_car()


    def start(self):
        self.is_stoping = False
        self.controler.is_stoping = False


    def stop_car(self):
        self.is_stoping = True
        self.controler.is_stoping = True
        # self.send_command(0.0, -50.0)


    def callback_position(self, msg):
        if self.last_header_time == None:
            self.last_header_time = float(self.get_clock().now().nanoseconds) * 1e-9
            self.controler.set_observed_state(msg.data[0], msg.data[1], msg.data[2])
            self.controler.simulated_state = self.controler.observed_state
            return

        msg_time = float(self.get_clock().now().nanoseconds) * 1e-9
        
        x, y, theta = msg.data[0], msg.data[1], msg.data[2]
        speed = msg.data[3]

        self.one_step(msg_time, x, y, theta, speed)


    def callback_span(self, msg):
        if self.last_header_time == None:
            self.last_header_time = msg.header.stamp.nanosec *1e-9 + msg.header.stamp.sec
            return

        msg_time = msg.header.stamp.nanosec *1e-9 + msg.header.stamp.sec

        pose_with_cv = msg.pose # PoseWithCovariance
        pose = pose_with_cv.pose # Pose
        position = pose.position # Point (x,y,z)
        orientation = pose.orientation # Quaternion (x,y,z,w)

        x, y = position.x, position.y
        _,_,theta = quaternion.euler_from_quaternion(orientation)

        l = msg.twist.twist.linear # Getting car mesured speed
        speed = np.sqrt(l.x**2+l.y**2+l.z**2)

        self.one_step(msg_time, x, y, theta, speed)


    def one_step(self, msg_time, x, y, theta, speed):
        # self.get_logger().warn(f"--- One step {"(SIMULATION) " if self.offline else ""}---")

        self.controler.Te = float(msg_time - self.last_header_time)
        # self.get_logger().info(f"MESSAGE TIME: {msg_time},   LAST HEADER: {self.last_header_time}")
        self.last_header_time = msg_time

        self.controler.set_observed_state(x, y, theta)
        self.controler.observed_speed = speed

        wheel_angle, acceleration = self.controler.one_step(simulation=self.offline)
        self.send_command(wheel_angle, acceleration)


    def send_command(self, wheel_angle, acceleration):
        if self.is_stoping:
            # return
            wheel_angle = 0.0
            acceleration = -50.0

        self.send_display_info()
        self.send_steering_wheel_angle_command(wheel_angle)
        self.send_acceleration_command(acceleration)


    def send_steering_wheel_angle_command(self, wheel_angle):
        msg_steering_wheel_angle = Float32()
        msg_steering_wheel_angle.data = wheel_angle * self.wheel_angle_factor # Converting from wheel angle to steering wheel angle
        self.pub_steering_wheel_angle.publish(msg_steering_wheel_angle)


    def send_acceleration_command(self, acceleration):
        msg = Float32()
        msg.data = acceleration

        # Adjusting values for the car
        if msg.data >= 0.0:
            msg.data *= self.acceleration_factor
        else:
            msg.data *= self.negative_acceleration_factor * self.acceleration_factor

        msg.data = min(msg.data, 100.0)
        msg.data = max(msg.data, -1000.0)

        self.pub_acceleration.publish(msg)


    def send_display_info(self):
        msg_map = PolygonStamped()
        msg_map.polygon.points = []
        for point in self.controler.map_to_follow.T:
            p = Point32()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0
            msg_map.polygon.points.append(p)
        msg_map.header.stamp = self.get_clock().now().to_msg()
        msg_map.header.frame_id = "world"
        self.pub_map.publish(msg_map)

        msg_controler_info = Float64MultiArray()
        msg_controler_info.data = [self.controler.observed_state[0],
                                   self.controler.observed_state[1],
                                   self.controler.observed_state[2],
                                   self.controler.observed_speed,
                                   self.controler.wheel_angle,
                                   self.controler.closest_segment[0],
                                   self.controler.closest_segment[1],
                                   self.controler.closest_segment2[0],
                                   self.controler.closest_segment2[1],
                                   self.controler.curvature,
                                   self.controler.desired_lateral_error-self.controler.lateral_error,
                                   self.controler.angle_error,
                                   self.controler.desired_speed,
                                   self.controler.delta_ecart,
                                   self.controler.delta_pente,
                                   self.controler.delta_courbure,
                                   self.controler.front_axle_state[0],
                                   self.controler.front_axle_state[1],
                                   self.controler.front_axle_state[2],]
        self.pub_controler_info.publish(msg_controler_info)

#         if self.is_stoping:
#             self.get_logger().info("Car is stoping!")
#
#         self.get_logger().info(f"Delta: {self.controler.wheel_angle:.4f} rad, \tErreur latérale: {self.controler.desired_lateral_error-self.controler.lateral_error:.4f} m, \tCourbure: {self.controler.curvature:.4f} m-1, \tVitesse: {self.controler.observed_speed:.4f} m/s")
#         self.get_logger().info(f"State: {self.controler.observed_state}")
#         self.get_logger().info(f"Time since last header: {self.controler.Te:.4f} s")
#         self.get_logger().info(f"Ecart vitesse: {self.controler.desired_speed - self.controler.observed_speed:.4f} m/s")
#         self.get_logger().info(f"Accélération: {self.controler.acceleration:.4f} m/s2")



def main(args=None):
    rclpy.init(args=args)
    controler = Controler()
    node = ControlNode(controler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
