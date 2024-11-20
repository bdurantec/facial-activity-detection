from pathlib import Path

from activity.activity_detection import ActivityDetector
from facial.facial_recognition import FacialDetector
from report.report_generator import ReportGenerator


def main():
    base_path = Path(__file__).resolve().parent
    directories = {
        'video_path': base_path / "resources" / "Unlocking_Facial_Recognition_Diverse_Activities_Analysis.mp4",
        'report_path': base_path / "resources" / "Report.txt",
        'output_path_facial': base_path / "resources" / "Output_facial_detection.mp4",
        'output_path_activities': base_path / "resources" / "Output_activities_detection.mp4"
    }

    if not directories['video_path'].exists():
        print(f"Error: Video file not found at {directories['video_path']}")
        return

    report: ReportGenerator = ReportGenerator(directories['report_path'])

    activities = ActivityDetector(
        video_path=str(directories['video_path']),
        output_path=str(directories['output_path_activities'])
    )
    activities.run()
    report.add_report(activities.report)

    emotions: FacialDetector = FacialDetector(
        video_path=str(directories['video_path']),
        output_path=str(directories['output_path_facial'])
    )
    emotions.run()
    report.add_report(emotions.report)

    report.finish()


if __name__ == "__main__":
    main()
