from collections import Counter
from collections import deque

import cv2
from deepface import DeepFace
from tqdm import tqdm

from report.report_dataclass import ReportDataclass


class FacialDetector:
    def __init__(self, video_path, output_path):
        self.__video_path: str = video_path
        self.__output_path: str = output_path
        self.__emotion_history = []
        self.report: ReportDataclass = ReportDataclass(
            title='Facial and emotion detection History',
            total_frames=0,
            anomalies_detected=0,
            summary={}
        )

    def run(self):
        """
        Process a video to detect faces, annotate emotions, and generate the output.
        """
        cap = self._initialize_video_capture(self.__video_path)
        if not cap:
            return

        output_writer, total_frames, emotion_buffer = self._initialize_video_writer(cap)
        self._process_frames(cap, output_writer, total_frames, emotion_buffer)

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
        emotion_buffer = deque(maxlen=10)

        return output_writer, total_frames, emotion_buffer

    def _process_frames(self, cap, output_writer, total_frames, emotion_buffer):
        """
        Process each frame, annotate movements, and save the output.
        """
        for _ in tqdm(range(total_frames), desc="Processing video - facial and emotion detection"):
            ret, frame = cap.read()
            if not ret:
                break

            result = DeepFace.analyze(frame, detector_backend='opencv', actions=['emotion'], enforce_detection=False)

            for face in result:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

                dominant_emotion = face['dominant_emotion']
                emotion_buffer.append(dominant_emotion)
                smoothed_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
                self.__emotion_history.append(smoothed_emotion)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    smoothed_emotion,
                    (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

            output_writer.write(frame)

    def _generate_report(self, total_frames: int):
        """
        Generate and save the report after processing
        """
        self.report.anomalies_detected = self._calculate_anomalies()
        self.report.total_frames = total_frames
        self.report.summary = self._generate_emotion_summary()

    def _calculate_anomalies(self):
        """
        Calculate the number of anomalies detected based on the emotions.
        """
        emotion_counts = Counter(self.__emotion_history)
        anomalies = sum(1 for emotion, count in emotion_counts.items() if count < 2)
        return anomalies

    def _generate_emotion_summary(self):
        """
        Generate a summary of detected emotions.
        """
        emotion_counts = Counter(self.__emotion_history)
        summary = "Detected emotions:\n"
        summary += "\n".join(f"{emotion}: {count}" for emotion, count in emotion_counts.items())
        return summary
