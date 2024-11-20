import cv2
import mediapipe as mp
from tqdm import tqdm

from activity.motion_detector import MotionDetector
from report.report_dataclass import ReportDataclass


class ActivityDetector:
    def __init__(self, video_path, output_path):
        self.__video_path = video_path
        self.__output_path = output_path
        self.__mp_pose = mp.solutions.pose
        self.__pose = self.__mp_pose.Pose()
        self.__motion_detector = MotionDetector()
        self.__mp_drawing = mp.solutions.drawing_utils
        self.report: ReportDataclass = ReportDataclass(
            title='Activity detection History',
            total_frames=0,
            anomalies_detected=0,
            summary={}
        )

    def run(self):
        """
        Process a video to detect activities, annotate movements, and save the output.
        """
        cap = self._initialize_video_capture(self.__video_path)
        if not cap:
            return

        output_writer, total_frames = self._initialize_video_writer(cap)
        self._process_frames(cap, output_writer, total_frames)

        output_writer.release()
        cv2.destroyAllWindows()
        self._generate_report(total_frames)

    def _initialize_video_capture(self, video_path):
        """
        Initialize video capture.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('Error: Unable to open video')
        return cap

    def _initialize_video_writer(self, cap):
        """
        Initialize video writer for output.
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_dimensions = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(
            self.__output_path, fourcc, fps, frame_dimensions
        )

        return output_writer, total_frames

    def _process_frames(self, cap, output_writer, total_frames):
        """
        Process each frame, annotate movements, and save the output.
        """
        for _ in tqdm(range(total_frames), desc="Processing video - activities detection"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.__pose.process(rgb_frame)
            if results.pose_landmarks:
                head_movement = self.__motion_detector.detect_head_movement(results.pose_landmarks)
                hand_movements = self.__motion_detector.detect_hand_movements(results.pose_landmarks)
                arm_positions = self.__motion_detector.detect_arm_positions(results.pose_landmarks)

                self.__mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.__mp_pose.POSE_CONNECTIONS
                )

                y_position = 30
                for text in [head_movement] + hand_movements + arm_positions:
                    cv2.putText(
                        frame,
                        text,
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )
                    y_position += 35

                self.__motion_detector.previous_landmarks = results.pose_landmarks

            output_writer.write(frame)

    def _generate_report(self, total_frames: int):
        """
        Generate and save the report after processing.
        """
        self.report.total_frames = total_frames
        self.report.summary = self._generate_activity_summary()

    def _generate_activity_summary(self):
        """
        Generate a summary of detected activities.
        """
        counters = self.__motion_detector.get_counters()
        summary = "Detected activities:\n"
        summary += "\n".join(f"{counters}: {count}" for counters, count in counters.items())
        return summary
