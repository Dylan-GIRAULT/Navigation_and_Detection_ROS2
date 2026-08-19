#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

# Enums pour les gestes
from shared.enums.enum_vehicule_action import RIGHT, LEFT, STOP, START, NO_HAND, action_to_str

class Perc_Int_Display_Node(Node):
    def __init__(self):
        super().__init__('perc_int_screen_node')

        # Subscribers
        self.gesture_sub = self.create_subscription(
            UInt32, 'perception_int_gesture_label', self.callback_gesture, 10
        )
        self.alert_sub = self.create_subscription(
            UInt32, 'perception_int_focus_score', self.callback_alert, 10
        )

        # Publisher Marker
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

        # Internal state
        self.current_gesture = NO_HAND
        self.current_alert_score = 100  # placeholder

        # Timer pour republier régulièrement les markers
        self.timer = self.create_timer(0.1, self.publish_markers)

        self.fixed_frame = "pandora"

    def callback_gesture(self, msg):
        self.current_gesture = msg.data

    def callback_alert(self, msg):
        self.current_alert_score = msg.data

    def get_gesture_text(self):
        return action_to_str(self.current_gesture)

    def get_alert_text_and_color(self):
        """Convertit le score en texte et couleur"""
        score = self.current_alert_score
        if score == 0:
            return ("DANGER", ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))  # rouge
        elif score == 50:
            return ("WARNING", ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0))  # orange
        else:
            return ("OK", ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0))  # vert

    def publish_markers(self):
        # --- Marker pour le geste ---
        gesture_marker = Marker()
        gesture_marker.header.frame_id = self.fixed_frame
        gesture_marker.header.stamp = self.get_clock().now().to_msg()
        gesture_marker.ns = "gesture"
        gesture_marker.id = 0
        gesture_marker.type = Marker.TEXT_VIEW_FACING
        gesture_marker.action = Marker.ADD
        gesture_marker.pose.position.x = 0.5
        gesture_marker.pose.position.y = 0.0
        gesture_marker.pose.position.z = 1.5
        gesture_marker.scale.z = 0.5
        gesture_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # blanc
        gesture_marker.text = f"Geste: {self.get_gesture_text()}"

        # --- Marker pour l'alerte ---
        alert_marker = Marker()
        alert_marker.header.frame_id = self.fixed_frame
        alert_marker.header.stamp = self.get_clock().now().to_msg()
        alert_marker.ns = "alert"
        alert_marker.id = 1
        alert_marker.type = Marker.TEXT_VIEW_FACING
        alert_marker.action = Marker.ADD
        alert_marker.pose.position.x = 0.5
        alert_marker.pose.position.y = 0.0
        alert_marker.pose.position.z = 2.5
        alert_marker.scale.z = 0.7

        alert_text, alert_color = self.get_alert_text_and_color()
        alert_marker.text = alert_text
        alert_marker.color = alert_color

        # Publier les markers
        self.marker_pub.publish(gesture_marker)
        self.marker_pub.publish(alert_marker)


def main(args=None):
    rclpy.init(args=args)
    node = Perc_Int_Display_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
