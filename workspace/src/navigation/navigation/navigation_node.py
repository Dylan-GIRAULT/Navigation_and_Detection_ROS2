import numpy as np

import time
from navigation.circuit_pos import get_circuit_pos, is_in_decision_area
from navigation.utils import ResettableTimer

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32, Float64MultiArray, Bool
from nav_msgs.msg import Odometry

from shared.enums.enum_segment_type import segment_to_str, LIGNE_DROITE

from shared.enums.enum_vehicule_action import START, RIGHT, LEFT, STOP, NO_HAND, action_to_str

import shared.resource.quaternion as quaternion

# Timer class that can be reset


# Function to reset action to NO_HAND
def reset_action_to_no_hand() -> None:
    global actuel_action
    global buffer_last_action_receive
    global navigationNode

    actuel_action = NO_HAND
    buffer_last_action_receive = NO_HAND
    navigationNode.get_logger().info('Action reset to NO_HAND due to timeout')


# Timer to reset action to NO_HAND after 10 seconds of inactivity
timer_to_reset_action = ResettableTimer(10.0, reset_action_to_no_hand)

# actuel action of the vehicle
global actuel_action
actuel_action = STOP

global actual_segment
actual_segment = LIGNE_DROITE

# current position of the vehicle
global position
position = None

global buffer_last_action_receive
buffer_last_action_receive = NO_HAND
global timer
timer = None

# Subscriber nodes
global navigationNode
navigationNode = None

def update_action(new_action: int) -> None:
    global actuel_action
    actuel_action = new_action
    timer_to_reset_action.start()
    main()

def update_position(new_position: list[float]) -> None:
    global position
    position = new_position
    send_position(position)
    main()

def send_position(pos: list[float]) -> None:
    if navigationNode is None:
        return
    try:
        msg = Float64MultiArray()
        msg.data = pos
        navigationNode.position_pub.publish(msg)
        #navigationNode.get_logger().info("Published position: '%s'" % str(msg))
    except Exception as e:
        navigationNode.get_logger().error("Failed to publish position: %s" % str(e))

def send_segment(segment : int) -> None:
    if navigationNode is None:
        return
    try:
        msg = UInt32()
        msg.data = LIGNE_DROITE
        navigationNode.segment_pub.publish(msg)
        segment_text = segment_to_str(msg.data)
        navigationNode.get_logger().info(f'Published segment: "{segment_text}"')
    except Exception as e:
        navigationNode.get_logger().error(f'Failed to publish segment: "{str(e)}"')

def send_decision_area(in_decision_area: bool) -> None:
    if navigationNode is None:
        return
    try:
        msg = Bool()
        msg.data = in_decision_area
        navigationNode.decision_area_pub.publish(msg)
        navigationNode.get_logger().info(f'Published decision area status: "{str(msg.data)}"')
    except Exception as e:
        navigationNode.get_logger().error(f'Failed to publish decision area status: "{str(e)}"')

class NavigationNode(Node):
    def __init__(self):
        super().__init__('NavigationNode')
        self.segment_pub = self.create_publisher(UInt32, '/navigation_to_control_segment', 10)

        self.position_pub = self.create_publisher(Float64MultiArray, '/navigation_to_control_position', 10)

        self.decision_area_pub = self.create_publisher(Bool, 'navigation_to_screen', 10)

        self.perception_sub = self.create_subscription(
            UInt32,
            'perception_int_to_navigation',
            self.listener_perception,
            10)
        self.position_sub = self.create_subscription(
            Float64MultiArray,
            'localisation_to_navigation',
            self.listener_position,
            10)
        self.manager_sub = self.create_subscription(
            UInt32,
            'stop_signal',
            self.listener_manager,
            10)
        self.sub_span = self.create_subscription(
            Odometry,
            "/span/odom",
            self.callback_span,
            10)

        # tell if the last action has already been used to not calculate multiple times the same action
        self.action_already_used = False


    def callback_span(self, msg) -> None:
        pose_with_cv = msg.pose # PoseWithCovariance
        pose = pose_with_cv.pose # Pose
        position = pose.position # Point (x,y,z)
        orientation = pose.orientation # Quaternion (x,y,z,w)

        x, y = position.x, position.y
        _,_,theta = quaternion.euler_from_quaternion(orientation)

        l = msg.twist.twist.linear # Getting car mesured speed
        speed = np.sqrt(l.x**2+l.y**2+l.z**2)

        update_position([x,y,theta,speed])


    def listener_perception(self, msg) -> None:
        global timer
        global buffer_last_action_receive
        global position
        global navigationNode

        new_action = msg.data

        # time in seconds
        time_max = 2
        time_max_for_stopping = 0.5

        action_text = action_to_str(new_action)
        
        # if position is None
        if position is None:
            self.get_logger().info("Position not defined, ignoring action")
            timer = time.time()
            return
        
        # if not in decision area and the user doesn't want to STOP or START
        if not is_in_decision_area(position) and not (new_action == STOP or new_action == START):
            self.get_logger().info("Not in decision area, ignoring action")
            timer = time.time()
            return

        if timer is None:
            timer = time.time()
        
        # if the action is different from the last action
        if buffer_last_action_receive != new_action:
            buffer_last_action_receive = new_action
            self.action_already_used = False
            self.get_logger().info(f'Buffered action: "{action_text}"')
            timer = time.time()
        
        # if the action is STOP and 0.5s have passed since last change 
        condition_stop = ( new_action == STOP and timer + time_max_for_stopping < time.time() )

        # if the action is other action and 2s have passed since last change
        condition_start = ( new_action == START and timer + time_max < time.time() )
        condition_right = ( new_action == RIGHT and timer + time_max < time.time() )
        condition_left = ( new_action == LEFT and timer + time_max < time.time() )

        if (condition_stop or condition_start or condition_right or condition_left) and not self.action_already_used:
            timer = time.time()
            self.action_already_used = True
            self.get_logger().info(f'Activate action: "{action_text}"')
            update_action(new_action)
        elif self.action_already_used:
            self.get_logger().info(f'Action "{action_text}" already used, waiting for new action')
            timer_to_reset_action.start() # restart the timer to reset action

    def listener_position(self, msg) -> None:
        new_position = msg.data
        #self.get_logger().info(f'Received position: "{str(new_position)}"')
        update_position(new_position)


    def listener_manager(self, msg) -> None:
        end()
        self.get_logger().info('Received stop signal, shutting down.')

def init() -> None:
    rclpy.init()
    global navigationNode
    clean()

    if navigationNode is not None:
        navigationNode.destroy_node()
    navigationNode = NavigationNode()
    

    try:
        rclpy.spin(navigationNode)
    except KeyboardInterrupt:
        pass
    finally:
        clean()


def clean() -> None:
    global navigationNode
    
    if navigationNode is not None:
        navigationNode.destroy_node()
        navigationNode = None


def end() -> None:
    clean()
    rclpy.try_shutdown()

def main() -> None:
    global actuel_action
    global actual_segment
    global position

    if position is None:
        navigationNode.get_logger().info("Position not initialized, waiting...")
        return

    actual_segment = get_circuit_pos(position, actuel_action, actual_segment)

    send_segment(actual_segment) 

    send_decision_area(is_in_decision_area(position))

if __name__ == '__main__':
    init()

# Tout ça grace à une State machine