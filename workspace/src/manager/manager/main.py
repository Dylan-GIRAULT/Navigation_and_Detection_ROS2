import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32, Int32MultiArray

import os

global managerNode
managerNode = None

class ManagerNode(Node):
    def __init__(self):
        super().__init__('ManagerNode')
        self.pub = self.create_publisher(UInt32, 'stop_signal', 10)

        self.perception_int_sub = self.create_subscription(
            UInt32,
            'perception_int_to_navigation',
            self.listener_perception_int,
            10)
        
        self.perception_ext_sub = self.create_subscription(
            Int32MultiArray,
            'perception_ext_to_localisation',
            self.listener_perception_ext,
            10)
        
        self.localisation_sub = self.create_subscription(
            Int32MultiArray,
            'localisation_to_navigation',
            self.listener_localisation,
            10)

        self.navigation_sub = self.create_subscription(
            UInt32,
            'navigation_to_control',
            self.listener_navigation,
            10)

    def listener_perception_int(self, msg):
        # TODO
        pass

    def listener_perception_ext(self, msg):
        # TODO
        pass
        

    def listener_localisation(self, msg):
        # TODO
        pass

    def listener_navigation(self, msg):
        # TODO
        pass


def init():
    rclpy.init()
    global managerNode

    if managerNode is not None:
        managerNode.destroy_node()
    managerNode = ManagerNode()

    try:
        rclpy.spin(managerNode)
    except KeyboardInterrupt:
        pass
    finally:
        clean()

def clean():
    global managerNode

    if managerNode is not None:
        managerNode.destroy_node()
        managerNode = None  

def end():
    clean()
    rclpy.try_shutdown()


if __name__ == '__main__':
    init()