import mediapipe as mp

from activity.body_status_dataclass import Head, Hand, Arm


class MotionDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=2
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Previous frame landmarks for movement detection
        self.previous_landmarks = None

        self.head: Head = Head()
        self.hand: Hand = Hand()
        self.arm: Arm = Arm()

    def detect_head_movement(self, landmarks):
        """
        Detects head movement based on nose position changes and ear difference.
        """
        if self.previous_landmarks is None:
            return self.head.default_movement

        current_nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_ear = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        prev_nose = self.previous_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

        movement = self.head.calculates_nose_distance_between_frames(current_nose, prev_nose)

        # Detect head tilt
        ear_difference = self.head.calculate_ear_height_difference(left_ear, right_ear)

        # Check if general movement exceeds the threshold
        if movement > self.head.threshold:
            self.head.counter += 1
            # Horizontal movements
            if current_nose.x - prev_nose.x > self.head.threshold:
                return self.head.right_movement
            elif prev_nose.x - current_nose.x > self.head.threshold:
                return self.head.left_movement

            # Vertical movements
            elif current_nose.y - prev_nose.y > self.head.threshold:
                return self.head.down_movement
            elif prev_nose.y - current_nose.y > self.head.threshold:
                return self.head.up_movement

        # Check for head tilt based on ear difference
        if ear_difference > self.head.threshold:
            self.head.counter += 1
            return self.head.tilted_movement

        return self.head.default_movement

    def detect_hand_movements(self, landmarks):
        """
        Detects hand movements and positions relative to shoulders and previous positions.
        """

        movements = []

        # Extract current landmarks for hands and shoulders
        left_hand = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Detect if hands are raised above shoulders
        if left_hand.y < left_shoulder.y - self.arm.threshold:
            self.hand.counter += 1
            movements.append(self.hand.left_raised_movement)
        if right_hand.y < right_shoulder.y - self.arm.threshold:
            self.hand.counter += 1
            movements.append(self.hand.right_raised_movement)

        # Detect lateral hand movements if previous landmarks are available
        if self.previous_landmarks:
            prev_left_hand = self.previous_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            prev_right_hand = self.previous_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate movement distances for each hand
            left_movement = self.hand.calculate_hands_distance_between_frames(left_hand, prev_left_hand)
            right_movement = self.hand.calculate_hands_distance_between_frames(right_hand, prev_right_hand)

            # Check if movements exceed the threshold
            if left_movement > self.hand.threshold:
                self.hand.counter += 1
                movements.append(self.hand.left_movement)
            if right_movement > self.hand.threshold:
                self.hand.counter += 1
                movements.append(self.hand.right_movement)

        return movements if movements else [self.hand.default_movement]

    def detect_arm_positions(self, landmarks):
        """
        Detects arm positions based on joint landmarks.
        """
        positions = []

        # Extract current landmarks for shoulders, elbows, and wrists
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        # Detect raised arms
        if self.arm.is_above(left_elbow, left_shoulder):
            self.arm.counter += 1
            positions.append(self.arm.left_up_movement)
        if self.arm.is_above(right_elbow, right_shoulder):
            self.arm.counter += 1
            positions.append(self.arm.right_up_movement)

        # Detect extended arms
        if self.arm.is_extended(left_wrist, left_shoulder, threshold=0.3):
            self.arm.counter += 1
            positions.append(self.arm.left_extended_movement)
        if self.arm.is_extended(right_wrist, right_shoulder, threshold=0.3):
            self.arm.counter += 1
            positions.append(self.arm.right_extended_movement)

        return positions if positions else [self.arm.default_movement]

    def get_counters(self):
        return {
            'Head Movements': self.head.counter,
            'Hand Movements': self.hand.counter,
            'Arm Movements': self.arm.counter
        }
