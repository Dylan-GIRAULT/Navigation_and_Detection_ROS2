import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # <--- message type pour la variable label
import cv2
import mediapipe as mp
from sensor_msgs.msg import Image
import numpy as np
from std_msgs.msg import UInt32, Int32MultiArray

from cv_bridge import CvBridge

from .reco_gesture.hand_processor import HandProcessor

import shared
from shared.enums.enum_vehicule_action import RIGHT, LEFT, STOP, START, NO_HAND


def draw_label(frame, label, conf, x=10, y=30):
    cv2.putText(frame, f"{label} ({conf:.2f})",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2)


class GestureDemoNode(Node):

    def __init__(self):
        super().__init__("perception_interieure")

        #self.get_logger().info("Gesture Demo Node started")

        # === Init camera ===
        # '/mako/image_raw',
        self.image_sub = self.create_subscription(
            Image,
            '/mako/image_raw',
            self.image_callback,
            10
        )
        self.frame = None
        self.bridge = CvBridge()

        # === Init MediaPipe ===
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # === Init your processor ===
        self.processor = HandProcessor()

        # === Init publisher ROS2 ===
        self.label_pub = self.create_publisher(UInt32, '/perception_int_gesture_label', 10)

        # Timer = boucle à ~30 FPS
        self.timer = self.create_timer(0.033, self.loop)

    def loop(self):
        if self.frame is None:
            return

        frame = self.frame.copy()

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run mediapipe
        res = self.hands.process(frame_rgb)
        label = NO_HAND
        conf = 0.0

        if res.multi_hand_landmarks:
            hand_lms = res.multi_hand_landmarks[0]
            h, w, _ = frame_rgb.shape

            landmarks = []
            for lm in hand_lms.landmark:
                landmarks.append((lm.x * w, lm.y * h, lm.z * w))

            handedness = None
            if res.multi_handedness:
                handedness = res.multi_handedness[0].classification[0].label

            label, conf, meta = self.processor.update(landmarks, handedness)
            self.mp_draw.draw_landmarks(frame_rgb, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)



        msg = UInt32()
        msg.data = label
        self.label_pub.publish(msg)
        #draw_label(frame, label, conf)
        #cv2.imshow('gestures', frame)
        #if cv2.waitKey(1) & 0xFF == 27:  # ESC
        #    #self.get_logger().info("ESC pressed, shutting down.")
        #    rclpy.shutdown()

    def destroy_node(self):
        self.hands.close()
        cv2.destroyAllWindows()
        super().destroy_node()

    def image_callback(self, msg: Image):
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame = self.bridge.imgmsg_to_cv2(msg)
        if frame is not None:
            self.frame = frame



def main(args=None):
    rclpy.init(args=args)
    node = GestureDemoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
