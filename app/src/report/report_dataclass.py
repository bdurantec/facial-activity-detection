from dataclasses import dataclass


@dataclass
class ReportDataclass:
    title: str
    total_frames: int
    anomalies_detected: int
    summary: dict
