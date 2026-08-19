import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class PerceptionRouterNode(Node):

    def __init__(self):
        super().__init__("perception_int_router")

        #self.get_logger().info("Perception Router Node started")

        # Dernière valeur reçue du gesture label
        self.last_gesture_label = 4  # valeur par défaut sûre

        # === Subscribers ===
        self.focus_sub = self.create_subscription(
            UInt32,
            "/perception_int_focus_score",
            self.focus_callback,
            10
        )

        self.gesture_sub = self.create_subscription(
            UInt32,
            "/perception_int_gesture_label",
            self.gesture_callback,
            10
        )

        # === Publisher navigation ===
        self.nav_pub = self.create_publisher(
            UInt32,
            "/perception_int_to_navigation",
            10
        )

        # === Publisher du flux vidéo compressé ===
        #self.image_pub = self.create_publisher(
        #    CompressedImage,
        #    "/mako/image_color/compressed",
        #    10
        #)

        # === OpenCV VideoCapture ===
        #self.cap = cv2.VideoCapture(0)

        # === Timer pour lire la caméra et publier le flux ===
        #self.timer = self.create_timer(0.033, self.publish_frame)  # ~30 FPS

    def gesture_callback(self, msg: UInt32):
        self.last_gesture_label = msg.data
        #self.get_logger().debug(f"Gesture label received: {self.last_gesture_label}")

    def focus_callback(self, msg: UInt32):
        output = UInt32()

        if msg.data >= 50:
            output.data = self.last_gesture_label
            #self.get_logger().debug(f"Focus OK → forwarding gesture {output.data}")
        else:
            output.data = 0
            #self.get_logger().debug("Focus NOT OK → sending default value 4")

        self.nav_pub.publish(output)

    def publish_frame(self):
        # Lire une frame depuis la caméra
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convertir en CompressedImage
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()

        # Publier sur le topic ROS2
        self.image_pub.publish(msg)

        # Optionnel : afficher la vidéo localement
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
            #self.get_logger().info("ESC pressed, shutting down.")
            rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionRouterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


# Salut ! J’utilise Whatsapp.