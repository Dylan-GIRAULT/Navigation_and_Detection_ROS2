
import numpy as np
import time
import random

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock



class SimulationNode(Node):
    def __init__(self, initial_state):
        super().__init__('simulation_node')

        self.controler_state = initial_state

        self.pub_position = self.create_publisher(Float64MultiArray, 'localisation_to_navigation', 10)

        self.sub_controler_info = self.create_subscription(Float64MultiArray, "/control/controler_info", self.callback_controler_info, 10)

        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.current_sim_time_ns = 0

    def callback_clock(self, time):
        self.current_sim_time_ns += time * 1e9
        msg = Clock()
        msg.clock.sec = int(self.current_sim_time_ns * 1e-9)
        msg.clock.nanosec = int(self.current_sim_time_ns % (1e9))
        self.pub_clock.publish(msg)

    def callback_controler_info(self, msg):
        self.controler_state = np.array([msg.data[0], msg.data[1], msg.data[2]])

    def send_position(self):
        #self.get_logger().info("Sending position")
        msg = Float64MultiArray()
        msg.data = self.controler_state
        self.pub_position.publish(msg)



def simulation_loop(simu: SimulationNode, simulation_duration):

    current_simulation_time = 0
    while current_simulation_time < simulation_duration:
        simu.send_position()
        rclpy.spin_once(simu, timeout_sec=0) # SimulationNode can still receive messages while simulating

        current_duration = random.uniform(0.02, 0.2)
        time.sleep(current_duration)
        current_simulation_time += current_duration
        simu.callback_clock(current_duration)



def main(args=None):
    rclpy.init(args=args)

    node = SimulationNode(np.array([-95.459, 125.47, 0]))
    simulation_loop(node, 180)