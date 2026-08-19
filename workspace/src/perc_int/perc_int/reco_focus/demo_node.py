import cv2
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32, Int32MultiArray

import mediapipe as mp

from .face_landmarks import FaceLandmarkEngine
from .head_pose import HeadPoseEstimator
from .eye_metrics import EyeMetrics
from .mouth_metrics import MouthMetrics
from .baillement_module import YawnDetector
from .distraction_module import DistractionModule
from .fatigue_module import FatigueModule
from .driver_attention_score import DriverAttentionScorer
from .head_calibration import HeadCalibrationManager


class DriverMonitorNode(Node):

    def __init__(self):
        super().__init__('driver_monitor_node')

        # Publisher
        self.score_pub = self.create_publisher(UInt32,'/driver/attention_score',10)

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            #self.get_logger().error("Impossible d'ouvrir la caméra")
            raise RuntimeError("Camera error")

        # Processing modules
        self.face_engine = FaceLandmarkEngine()
        self.head_pose = HeadPoseEstimator()
        self.eye_metrics = EyeMetrics()
        self.mouth_metrics = MouthMetrics()
        self.yawn_detector = YawnDetector()
        self.distraction = DistractionModule()
        self.fatigue = FatigueModule()
        self.scorer = DriverAttentionScorer()
        self.calibration = HeadCalibrationManager(duration_sec=2.5)

        # Timer (~30 FPS)
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)

        self.prev_time = time.time()
        self.fps = 0.0

        #self.get_logger().info("DriverMonitorNode démarré")

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            #self.get_logger().warning("Frame caméra invalide")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self.face_engine.process(rgb)

        if landmarks:
            # Estimation tête
            forward_head, face_x, face_y = self.head_pose.estimate(
                landmarks.landmark
            )

            # Calibration
            if not self.calibration.is_calibrated():
                self.calibration.update(forward_head, face_x, face_y)

                cv2.putText(
                    frame,
                    "Calibration en cours...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                self.show(frame)
                return

            # Angles relatifs
            yaw_rel, pitch_rel, forward_head = (
                self.head_pose.estimate_relative_angles(
                    landmarks.landmark,
                    self.calibration.forward_ref,
                    self.calibration.face_x_ref,
                    self.calibration.face_y_ref
                )
            )

            # Metrics
            ear = self.eye_metrics.compute(landmarks.landmark)
            mar = self.mouth_metrics.compute_MAR(landmarks.landmark)

            # Inference
            head_score = self.distraction.classify(yaw_rel, pitch_rel)
            yawn_penalty = self.yawn_detector.update(mar)
            fatigue_penalty = self.fatigue.update_and_score(ear)

            total_score = self.scorer.compute_score(
                initial_score=head_score,
                penalty=yawn_penalty + fatigue_penalty
            )

            # Publish
            msg = UInt32()
            msg.data = total_score
            self.score_pub.publish(msg)

            # Debug display
            cv2.putText(
                frame,
                f"Attention: {total_score:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

        self.show(frame)

    def show(self, frame):
        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            #self.get_logger().info("Arrêt demandé (ESC)")
            rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DriverMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
