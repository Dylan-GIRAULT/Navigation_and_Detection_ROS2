import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32

import cv2
import mediapipe as mp
import time

from .reco_focus.face_landmarks import FaceLandmarkEngine
from .reco_focus.head_pose import HeadPoseEstimator
from .reco_focus.eye_metrics import EyeMetrics
from .reco_focus.mouth_metrics import MouthMetrics
from .reco_focus.baillement_module import YawnDetector
from .reco_focus.distraction_module import DistractionModule
from .reco_focus.fatigue_module import FatigueModule
from .reco_focus.driver_attention_score import DriverAttentionScorer
from .reco_focus.head_calibration import HeadCalibrationManager


class DriverFocusNode(Node):

    def __init__(self):
        super().__init__("perception_focus")

        #self.get_logger().info("Driver Focus Node started")

        # === Camera ===
        self.cap = cv2.VideoCapture(0)

        # === Engines ===
        self.face_engine = FaceLandmarkEngine()
        self.head_pose = HeadPoseEstimator()
        self.eye_metrics = EyeMetrics()
        self.mouth_metrics = MouthMetrics()
        self.yawn_detector = YawnDetector()
        self.distraction = DistractionModule()
        self.fatigue = FatigueModule()
        self.scorer = DriverAttentionScorer()
        self.calibration = HeadCalibrationManager(duration_sec=2.5)

        # === ROS2 Publisher ===
        self.score_pub = self.create_publisher(
            UInt32,
            "/perception_int_focus_score",
            10
        )

        # === Timer (~30 FPS) ===
        self.timer = self.create_timer(0.033, self.loop)

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self.face_engine.process(rgb)

        score_to_publish = 0

        if landmarks:
            forward_head, face_x, face_y = self.head_pose.estimate(
                landmarks.landmark
            )

            # === Calibration ===
            if not self.calibration.is_calibrated():
                self.calibration.update(forward_head, face_x, face_y)
                cv2.putText(
                    frame,
                    "Calibration en cours...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
            else:
                yaw_rel, pitch_rel, forward_head = (
                    self.head_pose.estimate_relative_angles(
                        landmarks.landmark,
                        self.calibration.forward_ref,
                        self.calibration.face_x_ref,
                        self.calibration.face_y_ref
                    )
                )

                ear = self.eye_metrics.compute(landmarks.landmark)
                mar = self.mouth_metrics.compute_MAR(landmarks.landmark)

                head_score = self.distraction.classify(yaw_rel, pitch_rel)
                yawn_penalty = self.yawn_detector.update(mar)
                fatigue_penalty = self.fatigue.update_and_score(ear)

                total_score = self.scorer.compute_score(
                    initial_score=head_score,
                    penalty=yawn_penalty + fatigue_penalty
                )

                cv2.putText(
                    frame,
                    f"Attention: {total_score:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Normalisation pour ROS (0–100)
                score_to_publish = int(max(0, min(100, total_score * 100)))

        # === Publish ROS2 ===
        msg = UInt32()
        msg.data = score_to_publish
        self.score_pub.publish(msg)

        cv2.imshow("Driver Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            #self.get_logger().info("ESC pressed, shutting down.")
            rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DriverFocusNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
