import math
from dataclasses import dataclass


@dataclass
class Head:
    threshold: float = 0.03
    default_movement: str = 'Head Neutral'
    right_movement: str = 'Head Right'
    left_movement: str = 'Head Left'
    up_movement: str = 'Head Up'
    down_movement: str = 'Head Down'
    tilted_movement: str = 'Head Tilted'
    counter: int = 0

    def calculates_nose_distance_between_frames(self, nose, prev_nose):
        return math.sqrt(
            (nose.x - prev_nose.x) ** 2 +
            (nose.y - prev_nose.y) ** 2
        )

    def calculate_ear_height_difference(self, left_ear, right_ear):
        return abs(left_ear.y - right_ear.y)


@dataclass
class Hand:
    threshold: float = 0.04
    default_movement: str = 'Hands Neutral'
    left_raised_movement: str = 'Left Hand Raised'
    right_raised_movement: str = 'Right Hand Raised'
    left_movement: str = 'Left Hand Moving'
    right_movement: str = 'Right Hand Moving'
    counter: int = 0

    def calculate_hands_distance_between_frames(self, current_hand, prev_hand):
        return math.sqrt(
            (current_hand.x - prev_hand.x) ** 2 +
            (current_hand.y - prev_hand.y) ** 2
        )


@dataclass
class Arm:
    threshold: float = 0.2
    default_movement: str = 'Arms Neutral'
    left_up_movement: str = 'Left Arm Up'
    right_up_movement: str = 'Right Arm Up'
    left_extended_movement: str = 'Left Arm Extended'
    right_extended_movement: str = 'Right Arm Extended'
    counter: int = 0

    def is_above(self, joint, reference_point):
        return joint.y < reference_point.y

    def is_extended(self, joint, reference_point, threshold=0.3):
        distance = math.sqrt(
            (joint.x - reference_point.x) ** 2 +
            (joint.y - reference_point.y) ** 2
        )
        return distance > threshold
