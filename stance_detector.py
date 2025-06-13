import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import math
from collections import deque

class StanceDetector:
    def __init__(self, stability_threshold: float = 0.03, min_stability_duration: float = 0.1, confidence_threshold: float = 0.5, camera_perspective: str = "right", batsman_height: float = 5.5):
        """
        Initialize the stance detector.
        
        Args:
            stability_threshold: Maximum movement allowed for stable stance
            min_stability_duration: Minimum duration in seconds for stable stance
            confidence_threshold: Minimum confidence for pose detection
            camera_perspective: Camera perspective - "right" (bowler on right) or "left" (bowler on left)
            batsman_height: Height of batsman in feet
        """
        self.stability_threshold = stability_threshold
        self.min_stability_duration = min_stability_duration
        self.confidence_threshold = confidence_threshold
        self.camera_perspective = camera_perspective
        self.batsman_height = batsman_height
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Store pose history for stability analysis
        self.pose_history = deque(maxlen=30)  # Store last 30 frames
        self.movement_history = deque(maxlen=30)
        
        # Batting stance detection
        self.stance_window_duration = 0.3  # 300ms window
        self.last_stance_detected = None
        self.stance_cooldown = 1.0  # 1 second skip after detecting stance
        
        # Transition detection for balanced weight distribution
        self.previous_ankle_distance = None
        self.ankle_distance_threshold = 0.02  # 2% change threshold for transition detection
        
        # Key pose landmarks for stance detection
        self.key_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]
    
    def detect_stance(self, frame: np.ndarray, timestamp: float) -> Tuple[bool, Dict]:
        """
        Detect if the current frame shows a stable batting stance.
        
        Args:
            frame: Input frame
            timestamp: Current timestamp in seconds
            
        Returns:
            Tuple of (is_stable_stance, pose_data)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return False, {'confidence': 0, 'error': 'No pose detected'}
        
        # Extract pose landmarks
        landmarks = results.pose_landmarks.landmark
        pose_data = self._extract_pose_features(landmarks)
        
        # Check if pose meets confidence threshold
        if pose_data['confidence'] < self.confidence_threshold:
            return False, pose_data
        
        # Add to pose history
        self.pose_history.append({
            'timestamp': timestamp,
            'landmarks': pose_data['normalized_landmarks'],
            'features': pose_data
        })
        
        # Check stance criteria
        is_stance = self._check_stance_criteria(pose_data)
        
        # Check stability over time
        is_stable = self._check_stability(timestamp)
        
        return is_stance and is_stable, pose_data
    
    def _extract_pose_features(self, landmarks) -> Dict:
        """Extract relevant features from pose landmarks."""
        features = {}
        
        # Calculate confidence as average visibility of key landmarks
        confidences = []
        normalized_landmarks = {}
        
        for landmark_id in self.key_landmarks:
            landmark = landmarks[landmark_id.value]
            confidences.append(landmark.visibility)
            normalized_landmarks[landmark_id.name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        features['confidence'] = np.mean(confidences)
        features['normalized_landmarks'] = normalized_landmarks
        
        # Calculate specific stance features
        features.update(self._calculate_stance_features(landmarks))
        
        return features
    
    def _calculate_stance_features(self, landmarks) -> Dict:
        """Calculate specific features for stance detection."""
        features = {}
        
        # Get key landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # 1. Shoulder alignment (facing camera)
        shoulder_diff_z = abs(left_shoulder.z - right_shoulder.z)
        features['shoulder_alignment'] = shoulder_diff_z < 0.15  # Shoulders roughly at same depth
        
        # 2. Body orientation (chest facing camera)
        hip_center_z = (left_hip.z + right_hip.z) / 2
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
        features['body_facing_camera'] = shoulder_center_z < hip_center_z  # Upper body leaning forward
        
        # 3. Shoulder line angle with ground
        # Calculate the slope angle of the line connecting shoulders
        dx = right_shoulder.x - left_shoulder.x
        dy = right_shoulder.y - left_shoulder.y
        
        # Calculate angle in degrees (0 = horizontal, positive = right shoulder lower)
        shoulder_line_angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize to -90 to 90 degrees for shoulder tilt measurement
        # We want the acute angle relative to horizontal
        if shoulder_line_angle > 90:
            shoulder_line_angle = 180 - shoulder_line_angle
        elif shoulder_line_angle < -90:
            shoulder_line_angle = -180 - shoulder_line_angle
            
        features['shoulder_line_angle'] = shoulder_line_angle
        features['shoulder_line_good'] = -10 <= shoulder_line_angle <= 10
        
        # Calculate hip line angle with ground (same method as shoulder line)
        dx_hip = right_hip.x - left_hip.x
        dy_hip = right_hip.y - left_hip.y
        
        # Calculate angle in degrees (0 = horizontal, positive = right hip lower)
        hip_line_angle = math.degrees(math.atan2(dy_hip, dx_hip))
        
        # Normalize to -90 to 90 degrees for hip tilt measurement
        if hip_line_angle > 90:
            hip_line_angle = 180 - hip_line_angle
        elif hip_line_angle < -90:
            hip_line_angle = -180 - hip_line_angle
            
        features['hip_line_angle'] = hip_line_angle
        
        # Calculate shoulder twist relative to hip line (will be updated after twist calculations)
        shoulder_twist_hip = 0  # Placeholder, will be calculated after shoulder and hip twists
        features['shoulder_twist_hip'] = shoulder_twist_hip
        
        # Calculate head position (X-coordinate difference from right foot)
        head_x = nose.x
        right_foot_x = right_ankle.x
        head_position = head_x - right_foot_x
        features['head_position'] = head_position
        
        # Calculate sophisticated center of gravity using weighted body segments
        cog_result = self._calculate_weighted_center_of_gravity(landmarks)
        cog_x = cog_result['cog_x']
        cog_y = cog_result['cog_y']
        
        # Calculate distances from center of gravity to each foot
        left_foot_distance = ((cog_x - left_ankle.x)**2 + (cog_y - left_ankle.y)**2)**0.5
        right_foot_distance = ((cog_x - right_ankle.x)**2 + (cog_y - right_ankle.y)**2)**0.5
        
        # Calculate stance width and center for balanced classification
        stance_width = abs(left_ankle.x - right_ankle.x)
        stance_center_x = (left_ankle.x + right_ankle.x) / 2
        cog_distance_from_center = abs(cog_x - stance_center_x)
        
        # Define balanced threshold as a small percentage of stance width
        balanced_threshold = stance_width * 0.15  # 15% of stance width for balanced state
        
        # Determine weight distribution: 0 = Right Foot, 1 = Left Foot, 2 = Balanced, 3 = In transition
        if cog_distance_from_center <= balanced_threshold:
            weight_distribution = 2  # Balanced (CoG centered between feet)
        elif left_foot_distance < right_foot_distance:
            weight_distribution = 1  # Left Foot (CoG closer to left foot)
        else:
            weight_distribution = 0  # Right Foot (CoG closer to right foot)
        
        # Check for transition state if weight distribution is balanced
        current_ankle_distance = ((left_ankle.x - right_ankle.x)**2 + (left_ankle.y - right_ankle.y)**2)**0.5
        
        if weight_distribution == 2 and self.previous_ankle_distance is not None:
            # Calculate relative change in ankle distance
            distance_change = abs(current_ankle_distance - self.previous_ankle_distance)
            relative_change = distance_change / self.previous_ankle_distance if self.previous_ankle_distance > 0 else 0
            
            # If ankle distance changed significantly while balanced, mark as "In transition"
            if relative_change > self.ankle_distance_threshold:
                weight_distribution = 3  # In transition
        
        # Store current ankle distance for next frame comparison
        self.previous_ankle_distance = current_ankle_distance
        
        features['weight_distribution'] = weight_distribution
        features['cog_x'] = cog_x
        features['cog_y'] = cog_y
        features['left_foot_distance'] = left_foot_distance
        features['right_foot_distance'] = right_foot_distance
        features['stance_width'] = stance_width
        features['cog_distance_from_center'] = cog_distance_from_center
        features['balanced_threshold'] = balanced_threshold
        features['current_ankle_distance'] = current_ankle_distance
        features['cog_method'] = cog_result['method']
        
        # 4. Knee bend angle
        left_knee_angle = self._calculate_angle(
            (left_hip.x, left_hip.y), 
            (left_knee.x, left_knee.y), 
            (left_ankle.x, left_ankle.y)
        )
        right_knee_angle = self._calculate_angle(
            (right_hip.x, right_hip.y), 
            (right_knee.x, right_knee.y), 
            (right_ankle.x, right_ankle.y)
        )
        
        # Knees should be slightly bent (160-175 degrees) - either knee bent is sufficient
        features['knees_bent'] = (160 <= left_knee_angle <= 175) or (160 <= right_knee_angle <= 175)
        features['left_knee_angle'] = left_knee_angle
        features['right_knee_angle'] = right_knee_angle
        
        # 4. Feet alignment (parallel)
        feet_y_diff = abs(left_ankle.y - right_ankle.y)
        features['feet_parallel'] = feet_y_diff < 0.1  # Feet at similar height
        
        # 5. Head orientation (facing at least 30 degrees to the right of camera)
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Calculate head angle relative to shoulder center
        head_offset = nose.x - shoulder_center_x
        # Convert offset to approximate angle (rough estimation)
        # Assuming normalized coordinates, 0.1 offset ≈ 45 degrees
        head_angle_degrees = head_offset * 450  # Convert to degrees (rough approximation)
        features['head_angle'] = head_angle_degrees
        
        # Calculate head-to-shoulder line angle
        # Vector from shoulder center to nose
        dx = nose.x - shoulder_center_x
        dy = nose.y - shoulder_center_y
        # Calculate angle in degrees (0 degrees = straight up, positive = right lean)
        head_shoulder_angle = math.degrees(math.atan2(dx, -dy))  # -dy because y increases downward
        features['head_shoulder_angle'] = head_shoulder_angle
        
        # 30 degrees to the right means nose should be moderately right of center
        head_offset_threshold = 0.07  # Threshold for 30-degree turn (reduced from 45 degrees)
        features['head_facing_bowler'] = nose.x > (shoulder_center_x + head_offset_threshold)
        
        # 6. Stance width (feet should be shoulder-width apart)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        feet_width = abs(left_ankle.x - right_ankle.x)
        width_ratio = feet_width / shoulder_width if shoulder_width > 0 else 0
        
        # Adjust stance width criteria based on batsman height
        if self.batsman_height <= 3.1:
            # For shorter batsmen (typically children), allow wider stance
            features['stance_width_good'] = 0.8 <= width_ratio <= 2.0
        else:
            # Standard stance width for regular height batsmen
            features['stance_width_good'] = 0.8 <= width_ratio <= 1.5
        
        features['stance_width_ratio'] = width_ratio
        
        # 7. Hip line should be almost parallel to ground
        # Calculate the slope angle of the line connecting hips
        dx = right_hip.x - left_hip.x
        dy = right_hip.y - left_hip.y
        
        # Calculate angle in degrees (0 = horizontal, positive = right hip lower)
        hip_line_angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize to -90 to 90 degrees for hip tilt measurement
        # We want the acute angle relative to horizontal
        if hip_line_angle > 90:
            hip_line_angle = 180 - hip_line_angle
        elif hip_line_angle < -90:
            hip_line_angle = -180 - hip_line_angle
            
        features['hip_line_angle'] = hip_line_angle
        features['hip_line_parallel'] = -18 <= hip_line_angle <= 18  # Within 18 degrees of horizontal
        
        # 8. Calculate ankle-to-toe angle with camera straight view
        # Use foot index as toe approximation
        left_foot_index = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        right_foot_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        
        # Calculate ankle-to-toe angles (0° = pointing straight at camera)
        left_ankle_toe_angle = math.degrees(math.atan2(left_foot_index.y - left_ankle.y, left_foot_index.x - left_ankle.x))
        right_ankle_toe_angle = math.degrees(math.atan2(right_foot_index.y - right_ankle.y, right_foot_index.x - right_ankle.x))
        
        # Normalize angles to 0-360 degrees
        left_ankle_toe_angle = (left_ankle_toe_angle + 360) % 360
        right_ankle_toe_angle = (right_ankle_toe_angle + 360) % 360
        
        features['left_ankle_toe_angle'] = left_ankle_toe_angle
        features['right_ankle_toe_angle'] = right_ankle_toe_angle
        
        # 9. Toe Line Pointer (left: 60-105°, right: 90-130°)
        left_toe_good = 60 <= left_ankle_toe_angle <= 105
        right_toe_good = 90 <= right_ankle_toe_angle <= 130
        features['toe_line_pointer'] = left_toe_good and right_toe_good
        
        # 10. Head tilt (relative to vertical axis)
        # Calculate head tilt angle using nose and shoulder center
        head_tilt_angle = math.degrees(math.atan2(nose.x - shoulder_center_x, -(nose.y - shoulder_center_y)))
        # Normalize to -180 to 180 degrees (0 = straight up)
        if head_tilt_angle > 180:
            head_tilt_angle -= 360
        elif head_tilt_angle < -180:
            head_tilt_angle += 360
        features['head_tilt_angle'] = head_tilt_angle
        features['head_tilt_good'] = -20 <= head_tilt_angle <= 20
        

        
        # 11. Overall stance score
        stance_criteria = [
            features['shoulder_alignment'],
            features['body_facing_camera'],
            features['shoulder_line_good'],
            features['knees_bent'],
            features['feet_parallel'],
            features['stance_width_good'],
            features['hip_line_parallel'],
            features['toe_line_pointer'],
            features['head_tilt_good']
        ]
        
        # Include head facing criteria only for batsmen taller than 3.1 feet
        if self.batsman_height > 3.1:
            stance_criteria.append(features['head_facing_bowler'])
        features['stance_score'] = sum(stance_criteria) / len(stance_criteria)
        
        # Store ankle coordinates for movement tracking
        features['left_ankle_x'] = left_ankle.x
        features['left_ankle_y'] = left_ankle.y
        features['right_ankle_x'] = right_ankle.x
        features['right_ankle_y'] = right_ankle.y
        
        # 12. Shoulder line twist (rotation around vertical axis)
        # Calculate using shoulder depth perception (z-axis simulation)
        # Use shoulder-to-nose distance ratio to estimate twist
        left_shoulder_nose_dist = math.sqrt((left_shoulder.x - nose.x)**2 + (left_shoulder.y - nose.y)**2)
        right_shoulder_nose_dist = math.sqrt((right_shoulder.x - nose.x)**2 + (right_shoulder.y - nose.y)**2)
        
        # Calculate twist angle based on distance ratio (simulates depth)
        if right_shoulder_nose_dist > 0:
            distance_ratio = left_shoulder_nose_dist / right_shoulder_nose_dist
            # Convert ratio to approximate twist angle (empirical formula)
            shoulder_line_twist = math.degrees(math.atan((distance_ratio - 1) * 2))
            # Clamp to reasonable range
            shoulder_line_twist = max(-45, min(45, shoulder_line_twist))
        else:
            shoulder_line_twist = 0
        features['shoulder_line_twist'] = shoulder_line_twist
        
        # 13. Hip line twist (core rotation around vertical axis)
        # Calculate using hip-to-shoulder center distance ratio
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        left_hip_shoulder_dist = math.sqrt((left_hip.x - shoulder_center_x)**2 + (left_hip.y - shoulder_center_y)**2)
        right_hip_shoulder_dist = math.sqrt((right_hip.x - shoulder_center_x)**2 + (right_hip.y - shoulder_center_y)**2)
        
        if right_hip_shoulder_dist > 0:
            hip_distance_ratio = left_hip_shoulder_dist / right_hip_shoulder_dist
            # Convert to twist angle
            hip_line_twist = math.degrees(math.atan((hip_distance_ratio - 1) * 1.5))
            # Clamp to reasonable range
            hip_line_twist = max(-30, min(30, hip_line_twist))
        else:
            hip_line_twist = 0
        features['hip_line_twist'] = hip_line_twist
        
        # Update shoulder twist relative to hip twist (torso rotation)
        # This measures how much the shoulders are rotated relative to the hips
        shoulder_twist_hip = shoulder_line_twist - hip_line_twist
        features['shoulder_twist_hip'] = shoulder_twist_hip
        
        # 14. Knee-to-ankle line angles with ground
        # Left knee-to-ankle angle with horizontal ground
        left_knee_ankle_dx = left_ankle.x - left_knee.x
        left_knee_ankle_dy = left_ankle.y - left_knee.y
        left_knee_to_ankle_angle = math.degrees(math.atan2(left_knee_ankle_dy, left_knee_ankle_dx))
        # Normalize to angle with ground (0° = horizontal, 90° = vertical down)
        left_knee_to_ankle_angle = abs(left_knee_to_ankle_angle)
        if left_knee_to_ankle_angle > 90:
            left_knee_to_ankle_angle = 180 - left_knee_to_ankle_angle
        features['left_knee_to_ankle_angle'] = left_knee_to_ankle_angle
        
        # Right knee-to-ankle angle with horizontal ground
        right_knee_ankle_dx = right_ankle.x - right_knee.x
        right_knee_ankle_dy = right_ankle.y - right_knee.y
        right_knee_to_ankle_angle = math.degrees(math.atan2(right_knee_ankle_dy, right_knee_ankle_dx))
        # Normalize to angle with ground
        right_knee_to_ankle_angle = abs(right_knee_to_ankle_angle)
        if right_knee_to_ankle_angle > 90:
            right_knee_to_ankle_angle = 180 - right_knee_to_ankle_angle
        features['right_knee_to_ankle_angle'] = right_knee_to_ankle_angle
        
        # 15. Elbow-to-wrist line angles
        # Get elbow and wrist landmarks
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Left elbow-to-wrist angle with horizontal
        left_elbow_wrist_dx = left_wrist.x - left_elbow.x
        left_elbow_wrist_dy = left_wrist.y - left_elbow.y
        left_elbow_wrist_angle = math.degrees(math.atan2(left_elbow_wrist_dy, left_elbow_wrist_dx))
        # Normalize to 0-180 degrees (angle magnitude)
        left_elbow_wrist_angle = abs(left_elbow_wrist_angle)
        if left_elbow_wrist_angle > 90:
            left_elbow_wrist_angle = 180 - left_elbow_wrist_angle
        features['left_elbow_wrist_angle'] = left_elbow_wrist_angle
        
        # Right elbow-to-wrist angle with horizontal
        right_elbow_wrist_dx = right_wrist.x - right_elbow.x
        right_elbow_wrist_dy = right_wrist.y - right_elbow.y
        right_elbow_wrist_angle = math.degrees(math.atan2(right_elbow_wrist_dy, right_elbow_wrist_dx))
        # Normalize to 0-180 degrees (angle magnitude)
        right_elbow_wrist_angle = abs(right_elbow_wrist_angle)
        if right_elbow_wrist_angle > 90:
            right_elbow_wrist_angle = 180 - right_elbow_wrist_angle
        features['right_elbow_wrist_angle'] = right_elbow_wrist_angle
        
        # 16. Shoulder-to-elbow line angles
        # Left shoulder-to-elbow angle with horizontal
        left_shoulder_elbow_dx = left_elbow.x - left_shoulder.x
        left_shoulder_elbow_dy = left_elbow.y - left_shoulder.y
        left_shoulder_elbow_angle = math.degrees(math.atan2(left_shoulder_elbow_dy, left_shoulder_elbow_dx))
        # Normalize to 0-180 degrees (angle magnitude)
        left_shoulder_elbow_angle = abs(left_shoulder_elbow_angle)
        if left_shoulder_elbow_angle > 90:
            left_shoulder_elbow_angle = 180 - left_shoulder_elbow_angle
        features['left_shoulder_elbow_angle'] = left_shoulder_elbow_angle
        
        # Right shoulder-to-elbow angle with horizontal
        right_shoulder_elbow_dx = right_elbow.x - right_shoulder.x
        right_shoulder_elbow_dy = right_elbow.y - right_shoulder.y
        right_shoulder_elbow_angle = math.degrees(math.atan2(right_shoulder_elbow_dy, right_shoulder_elbow_dx))
        # Normalize to 0-180 degrees (angle magnitude)
        right_shoulder_elbow_angle = abs(right_shoulder_elbow_angle)
        if right_shoulder_elbow_angle > 90:
            right_shoulder_elbow_angle = 180 - right_shoulder_elbow_angle
        features['right_shoulder_elbow_angle'] = right_shoulder_elbow_angle
        
        return features
    
    def _analyze_ankle_movement(self, results: List[Dict], start_frame: int, end_frame: int) -> Dict:
        """Analyze ankle coordinate changes within a stable period."""
        ankle_info = {
            'has_movement': False,
            'movement_points': [],
            'max_displacement': 0.0,
            'total_frames_analyzed': 0
        }
        
        if not results or start_frame >= end_frame:
            return ankle_info
        
        # Get frames with valid ankle coordinates
        frames_with_coords = []
        for i in range(start_frame, min(end_frame + 1, len(results))):
            if (results[i].get('ankle_coords') is not None and 
                results[i].get('pose_confidence', 0) > 0.5):
                frames_with_coords.append(i)
        
        if len(frames_with_coords) < 2:
            return ankle_info
        
        ankle_info['total_frames_analyzed'] = len(frames_with_coords)
        
        # Get initial ankle positions from the first valid frame
        first_frame_idx = frames_with_coords[0]
        initial_coords = results[first_frame_idx]['ankle_coords']
        initial_left_x = initial_coords['left_ankle_x']
        initial_left_y = initial_coords['left_ankle_y']
        initial_right_x = initial_coords['right_ankle_x']
        initial_right_y = initial_coords['right_ankle_y']
        
        movement_threshold = 0.02  # 2% movement in normalized coordinates
        movement_detected_at = []
        max_displacement = 0.0
        
        # Check each frame for significant ankle movement
        for frame_idx in frames_with_coords[1:]:
            current_coords = results[frame_idx]['ankle_coords']
            
            # Calculate displacement for left ankle
            left_dx = current_coords['left_ankle_x'] - initial_left_x
            left_dy = current_coords['left_ankle_y'] - initial_left_y
            left_displacement = math.sqrt(left_dx * left_dx + left_dy * left_dy)
            
            # Calculate displacement for right ankle
            right_dx = current_coords['right_ankle_x'] - initial_right_x
            right_dy = current_coords['right_ankle_y'] - initial_right_y
            right_displacement = math.sqrt(right_dx * right_dx + right_dy * right_dy)
            
            # Use the maximum displacement of either ankle
            max_ankle_displacement = max(left_displacement, right_displacement)
            
            if max_ankle_displacement > movement_threshold:
                movement_detected_at.append({
                    'frame': frame_idx,
                    'timestamp': results[frame_idx]['timestamp'],
                    'displacement': max_ankle_displacement,
                    'left_displacement': left_displacement,
                    'right_displacement': right_displacement
                })
                max_displacement = max(max_displacement, max_ankle_displacement)
        
        ankle_info['has_movement'] = len(movement_detected_at) > 0
        ankle_info['movement_points'] = movement_detected_at
        ankle_info['max_displacement'] = max_displacement
        
        return ankle_info
    
    def _analyze_shot_triggers(self, results: List[Dict]) -> List[Dict]:
        """Analyze sudden, significant movements in biomechanical parameters to detect shot triggers."""
        if len(results) < 10:  # Need minimum frames for analysis
            return []
        
        shot_triggers = []
        fps = 30  # Assume 30 FPS
        min_duration_frames = int(0.2 * fps)  # 200ms = 6 frames at 30fps
        
        # Debug flag for 15.2-15.3 timeframe
        debug_timeframe = True
        
        # Parameters to track for sudden movement (excluding head tilt)
        tracked_params = [
            'shoulder_line_angle',
            'hip_line_angle',
            'shoulder_line_twist',
            'hip_line_twist',
            'knee_to_ankle_angle',
            'knee_angle',
            'elbow_wrist_line_angle',
            'shoulder_elbow_line_angle',
            'ankle_coordinates'
        ]
        
        # Extract parameter values for frames with valid pose data
        param_data = {}
        valid_frames = []
        
        for i, result in enumerate(results):
            if result.get('pose_confidence', 0) > 0.5 and result.get('biomech_data'):
                valid_frames.append(i)
                
                # Use actual stored biomechanical data
                biomech_data = result['biomech_data']
                param_data[i] = {
                    'shoulder_line_angle': biomech_data['shoulder_line_angle'],
                    'hip_line_angle': biomech_data['hip_line_angle'],
                    'shoulder_line_twist': biomech_data.get('shoulder_line_twist', 0),
                    'hip_line_twist': biomech_data.get('hip_line_twist', 0),
                    'knee_to_ankle_angle': max(
                        biomech_data.get('left_knee_to_ankle_angle', 0),
                        biomech_data.get('right_knee_to_ankle_angle', 0)
                    ),
                    'knee_angle': max(
                        biomech_data.get('left_knee_angle', 0),
                        biomech_data.get('right_knee_angle', 0)
                    ),
                    # Store individual ankle coordinates for movement calculation
                    'left_ankle_x': biomech_data.get('left_ankle_x', 0),
                    'left_ankle_y': biomech_data.get('left_ankle_y', 0),
                    'right_ankle_x': biomech_data.get('right_ankle_x', 0),
                    'right_ankle_y': biomech_data.get('right_ankle_y', 0),
                    # Store individual elbow-wrist angles for movement calculation
                    'left_elbow_wrist_angle': biomech_data.get('left_elbow_wrist_angle', 0),
                    'right_elbow_wrist_angle': biomech_data.get('right_elbow_wrist_angle', 0),
                    # Store individual shoulder-elbow angles for movement calculation
                    'left_shoulder_elbow_angle': biomech_data.get('left_shoulder_elbow_angle', 0),
                    'right_shoulder_elbow_angle': biomech_data.get('right_shoulder_elbow_angle', 0)
                }
        
        if len(valid_frames) < min_duration_frames * 2:
            return []
        
        # Sliding window analysis for sudden movements
        movement_threshold = {
            'shoulder_line_angle': 10,  # degrees
            'hip_line_angle': 2.5,
            'shoulder_line_twist': 20,  # degrees - rotation around vertical axis
            'hip_line_twist': 2,  # degrees - core rotation
            'knee_to_ankle_angle': 5,  # degrees - angle with ground (either leg)
            'knee_angle': 10,  # degrees - knee bend (either leg)
            'elbow_wrist_line_angle': 10,  # degrees - elbow-wrist line angle (either arm)
            'shoulder_elbow_line_angle': 5,  # degrees - shoulder-elbow line angle (either arm)
            'ankle_coordinates': 0.025  # normalized coordinates (any ankle movement)
        }
        
        frame_skip = 3  # Compare with 3rd frame before (n-3)
        cooldown_frames = int(1.0 * fps)  # 1 second cooldown = 30 frames at 30fps
        min_trigger_frames = 3  # At least 3 frames with 3+ parameters each
        last_trigger_idx = -cooldown_frames  # Track last trigger to enforce cooldown
        
        # Process each possible 200ms window
        for start_idx in range(frame_skip, len(valid_frames) - min_duration_frames + 1):
            # Skip if we're still in cooldown period from last trigger
            if start_idx < last_trigger_idx + cooldown_frames:
                continue
                
            # Define the 200ms window of frames to analyze
            window_start_idx = start_idx
            window_end_idx = min(start_idx + min_duration_frames, len(valid_frames))
            
            qualifying_frames = []  # Frames that have 3+ parameters exceeding thresholds
            
            # Check each frame in the 200ms window
            for window_idx in range(window_start_idx, window_end_idx):
                current_frame = valid_frames[window_idx]
                compare_frame_idx = window_idx - frame_skip
                
                # Make sure comparison frame exists
                if compare_frame_idx < 0 or compare_frame_idx >= len(valid_frames):
                    continue
                    
                compare_frame = valid_frames[compare_frame_idx]
                
                if current_frame not in param_data or compare_frame not in param_data:
                    continue
                
                # Check parameter changes for this specific frame pair (n vs n+3)
                frame_movements = []
                
                for param in tracked_params:
                    change = 0
                    
                    if param == 'ankle_coordinates':
                        # Calculate maximum ankle coordinate change (any ankle movement)
                        left_x_change = abs(param_data[compare_frame]['left_ankle_x'] - param_data[current_frame]['left_ankle_x'])
                        left_y_change = abs(param_data[compare_frame]['left_ankle_y'] - param_data[current_frame]['left_ankle_y'])
                        right_x_change = abs(param_data[compare_frame]['right_ankle_x'] - param_data[current_frame]['right_ankle_x'])
                        right_y_change = abs(param_data[compare_frame]['right_ankle_y'] - param_data[current_frame]['right_ankle_y'])
                        change = max(left_x_change, left_y_change, right_x_change, right_y_change)
                    
                    elif param == 'knee_angle':
                        # Use the pre-calculated maximum knee angle from param_data
                        current_val = param_data[current_frame][param]
                        compare_val = param_data[compare_frame][param]
                        change = abs(compare_val - current_val)
                    
                    elif param == 'knee_to_ankle_angle':
                        # Use the pre-calculated maximum knee-to-ankle angle from param_data
                        current_val = param_data[current_frame][param]
                        compare_val = param_data[compare_frame][param]
                        change = abs(compare_val - current_val)
                    
                    elif param == 'elbow_wrist_line_angle':
                        # Calculate maximum elbow-wrist angle change (either arm)
                        left_current = param_data[current_frame].get('left_elbow_wrist_angle', 0)
                        left_compare = param_data[compare_frame].get('left_elbow_wrist_angle', 0)
                        right_current = param_data[current_frame].get('right_elbow_wrist_angle', 0)
                        right_compare = param_data[compare_frame].get('right_elbow_wrist_angle', 0)
                        
                        left_change = abs(left_compare - left_current)
                        right_change = abs(right_compare - right_current)
                        change = max(left_change, right_change)
                    
                    elif param == 'shoulder_elbow_line_angle':
                        # Calculate maximum shoulder-elbow angle change (either arm)
                        left_current = param_data[current_frame].get('left_shoulder_elbow_angle', 0)
                        left_compare = param_data[compare_frame].get('left_shoulder_elbow_angle', 0)
                        right_current = param_data[current_frame].get('right_shoulder_elbow_angle', 0)
                        right_compare = param_data[compare_frame].get('right_shoulder_elbow_angle', 0)
                        
                        left_change = abs(left_compare - left_current)
                        right_change = abs(right_compare - right_current)
                        change = max(left_change, right_change)
                    
                    else:
                        # Standard single parameter change calculation
                        if param in param_data[current_frame] and param in param_data[compare_frame]:
                            current_val = param_data[current_frame][param]
                            compare_val = param_data[compare_frame][param]
                            change = abs(compare_val - current_val)
                    
                    if change > movement_threshold[param]:
                        # Calculate directional change (positive = increase, negative = decrease)
                        if param == 'ankle_coordinates':
                            # For ankle coordinates, use the maximum directional change
                            left_x_dir = param_data[compare_frame]['left_ankle_x'] - param_data[current_frame]['left_ankle_x']
                            left_y_dir = param_data[compare_frame]['left_ankle_y'] - param_data[current_frame]['left_ankle_y']
                            right_x_dir = param_data[compare_frame]['right_ankle_x'] - param_data[current_frame]['right_ankle_x']
                            right_y_dir = param_data[compare_frame]['right_ankle_y'] - param_data[current_frame]['right_ankle_y']
                            
                            # Use the direction with maximum absolute change
                            changes = [left_x_dir, left_y_dir, right_x_dir, right_y_dir]
                            max_change_idx = max(range(len(changes)), key=lambda i: abs(changes[i]))
                            directional_change = changes[max_change_idx]
                        
                        elif param in ['knee_angle', 'knee_to_ankle_angle']:
                            # For composite parameters, use the direction of the larger change
                            directional_change = param_data[compare_frame][param] - param_data[current_frame][param]
                        
                        elif param == 'elbow_wrist_line_angle':
                            # Use direction of arm with larger change
                            left_current = param_data[current_frame].get('left_elbow_wrist_angle', 0)
                            left_compare = param_data[compare_frame].get('left_elbow_wrist_angle', 0)
                            right_current = param_data[current_frame].get('right_elbow_wrist_angle', 0)
                            right_compare = param_data[compare_frame].get('right_elbow_wrist_angle', 0)
                            
                            left_change = left_compare - left_current
                            right_change = right_compare - right_current
                            directional_change = left_change if abs(left_change) > abs(right_change) else right_change
                        
                        elif param == 'shoulder_elbow_line_angle':
                            # Use direction of arm with larger change
                            left_current = param_data[current_frame].get('left_shoulder_elbow_angle', 0)
                            left_compare = param_data[compare_frame].get('left_shoulder_elbow_angle', 0)
                            right_current = param_data[current_frame].get('right_shoulder_elbow_angle', 0)
                            right_compare = param_data[compare_frame].get('right_shoulder_elbow_angle', 0)
                            
                            left_change = left_compare - left_current
                            right_change = right_compare - right_current
                            directional_change = left_change if abs(left_change) > abs(right_change) else right_change
                        
                        else:
                            # Standard directional change calculation
                            directional_change = param_data[compare_frame][param] - param_data[current_frame][param]
                        
                        frame_movements.append({
                            'parameter': param,
                            'change': change,
                            'directional_change': directional_change,
                            'direction': 'increase' if directional_change > 0 else 'decrease',
                            'threshold': movement_threshold[param]
                        })
                
                # Step 1: Check if this individual frame has 3+ parameters exceeding thresholds
                if len(frame_movements) >= 3:
                    qualifying_frames.append({
                        'frame': current_frame,
                        'compare_frame': compare_frame,
                        'timestamp': results[current_frame]['timestamp'],
                        'compare_timestamp': results[compare_frame]['timestamp'],
                        'movements': frame_movements,
                        'parameter_count': len(frame_movements)
                    })
            
            # Debug output for 15.2-15.3 timeframe
            window_start_time = results[valid_frames[window_start_idx]]['timestamp'] if window_start_idx < len(valid_frames) else 0
            window_end_time = results[valid_frames[min(window_end_idx-1, len(valid_frames)-1)]]['timestamp'] if window_end_idx <= len(valid_frames) else 0
            
            if debug_timeframe and 15.2 <= window_start_time <= 15.3:
                print(f"Debug: Window {window_start_time:.3f}-{window_end_time:.3f}s: {len(qualifying_frames)} qualifying frames (need {min_trigger_frames})")
                for qf in qualifying_frames:
                    print(f"  Frame {qf['frame']} at {qf['timestamp']:.3f}s: {qf['parameter_count']} parameters")
            
            # Step 2: Check if we have at least 4 qualifying frames in the 200ms window
            if len(qualifying_frames) >= min_trigger_frames:
                # Found a potential shot trigger
                trigger_start = qualifying_frames[0]['timestamp']
                trigger_end = qualifying_frames[-1]['timestamp']
                duration = trigger_end - trigger_start
                
                # Calculate average parameters moved across qualifying frames
                total_params = sum(frame['parameter_count'] for frame in qualifying_frames)
                avg_params_moved = total_params / len(qualifying_frames) if qualifying_frames else 0
                
                if debug_timeframe and 15.2 <= trigger_start <= 15.3:
                    print(f"Debug: SHOT TRIGGER DETECTED at {trigger_start:.3f}s with {len(qualifying_frames)} qualifying frames")
                
                shot_triggers.append({
                    'trigger_frame': qualifying_frames[0]['frame'],
                    'trigger_time': trigger_start,
                    'end_time': trigger_end,
                    'duration': duration,
                    'parameters_moved': int(avg_params_moved),
                    'trigger_frames_count': len(qualifying_frames),
                    'total_window_frames': window_end_idx - window_start_idx,
                    'trigger_ratio': len(qualifying_frames) / (window_end_idx - window_start_idx) if (window_end_idx - window_start_idx) > 0 else 0,
                    'movement_details': qualifying_frames[0]['movements'],
                    'qualifying_frame_details': qualifying_frames
                })
                
                # Update last trigger index and enforce cooldown
                last_trigger_idx = start_idx
        
        return shot_triggers
    
    def detect_batting_stance(self, results: List[Dict], fps: float) -> List[Dict]:
        """
        Detect "Batting stance taken" events using 300ms window with 6 stability criteria.
        
        Args:
            results: List of frame analysis results
            fps: Frames per second of the video
            
        Returns:
            List of batting stance detection events
        """
        batting_stances = []
        if not results or len(results) < 2:
            print(f"DEBUG: Not enough results for batting stance detection: {len(results) if results else 0}")
            return batting_stances
        
        # Calculate frames needed for 300ms window and n-7 comparison
        window_frames = max(1, int(fps * self.stance_window_duration))  # ~300ms worth of frames
        skip_frames = 7  # n-7 comparison
        
        # Cooldown tracking - 300ms skip after detection
        last_stance_frame_idx = -1
        skip_after_detection = int(fps * 0.3)  # 300ms skip
        
        print(f"DEBUG: Processing {len(results)} results with window_frames={window_frames}, skip_after_detection={skip_after_detection}")
        print(f"DEBUG: Majority voting - checking all 9 frames per window, requiring 5/9 to pass all criteria")
        
        # Process each potential window start
        for start_idx in range(len(results) - window_frames):
            # Check 300ms skip after detection
            if last_stance_frame_idx >= 0 and start_idx - last_stance_frame_idx < skip_after_detection:
                continue
            
            window_end_idx = start_idx + window_frames
            window_qualified = True
            criteria_details = []
            
            # Check all frames in the window against n-5 for majority voting
            passed_frames = 0
            
            for current_idx in range(start_idx, window_end_idx):
                compare_idx = current_idx - skip_frames
                if compare_idx < 0:
                    continue
                
                current_result = results[current_idx]
                compare_result = results[compare_idx]
                
                # Skip if either frame lacks pose data
                if (not current_result.get('pose_detected') or 
                    not compare_result.get('pose_detected') or
                    current_result.get('pose_confidence', 0) < self.confidence_threshold or
                    compare_result.get('pose_confidence', 0) < self.confidence_threshold):
                    continue
                
                current_features = current_result.get('pose_features', {})
                compare_features = compare_result.get('pose_features', {})
                
                # Check all 5 criteria for this frame pair
                frame_criteria = self._check_stance_criteria_frame(current_features, compare_features)
                criteria_details.append({
                    'frame_idx': current_idx,
                    'timestamp': current_result.get('timestamp', 0),
                    'criteria': frame_criteria
                })
                
                # Count frames that pass all criteria
                if all(frame_criteria.values()):
                    passed_frames += 1
                elif start_idx % 100 == 0:  # Debug every 100th window
                    print(f"DEBUG: Frame {current_idx} failed criteria: {frame_criteria}")
            
            # Require at least 5 out of 9 frames to pass all criteria (majority voting)
            window_qualified = passed_frames >= 5
            
            # If entire window passed all criteria, mark as batting stance
            if window_qualified and criteria_details:
                start_timestamp = results[start_idx].get('timestamp', 0)
                end_timestamp = results[window_end_idx - 1].get('timestamp', 0)
                
                batting_stances.append({
                    'start_frame': start_idx,
                    'end_frame': window_end_idx - 1,
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp,
                    'duration': end_timestamp - start_timestamp,
                    'criteria_details': criteria_details,
                    'window_frames': len(criteria_details)
                })
                
                last_stance_frame_idx = start_idx
                print(f"DEBUG: BATTING STANCE DETECTED at frame {start_idx}, timestamp {start_timestamp:.3f}s")
        
        print(f"DEBUG: Total batting stances detected: {len(batting_stances)}")
        return batting_stances
    
    def _check_stance_criteria_frame(self, current_features: Dict, compare_features: Dict) -> Dict:
        """
        Check all 6 batting stance criteria for a single frame comparison.
        
        Args:
            current_features: Current frame pose features
            compare_features: Compare frame (n-3) pose features
            
        Returns:
            Dictionary with boolean results for each criterion
        """
        criteria = {
            'ankle_stability': False,
            'hip_angle_stable': False,
            'shoulder_twist_stable': False,
            'shoulder_elbow_stable': False,
            'camera_perspective_ok': False
        }
        
        # Criterion 1: Ankles at same coordinates (both left and right)
        ankle_threshold = 0.01  # 1% movement in normalized coordinates
        left_ankle_stable = (abs(current_features.get('left_ankle_x', 0) - compare_features.get('left_ankle_x', 0)) <= ankle_threshold and
                           abs(current_features.get('left_ankle_y', 0) - compare_features.get('left_ankle_y', 0)) <= ankle_threshold)
        right_ankle_stable = (abs(current_features.get('right_ankle_x', 0) - compare_features.get('right_ankle_x', 0)) <= ankle_threshold and
                            abs(current_features.get('right_ankle_y', 0) - compare_features.get('right_ankle_y', 0)) <= ankle_threshold)
        criteria['ankle_stability'] = left_ankle_stable and right_ankle_stable
        
        # Criterion 2: Hip line angle unchanged or changed less than 2 degrees
        hip_angle_change = abs(current_features.get('hip_line_angle', 0) - compare_features.get('hip_line_angle', 0))
        criteria['hip_angle_stable'] = hip_angle_change < 2.0
        
        # Criterion 3: Shoulder line twist unchanged or changed less than 10 degrees
        shoulder_twist_change = abs(current_features.get('shoulder_line_twist', 0) - compare_features.get('shoulder_line_twist', 0))
        criteria['shoulder_twist_stable'] = shoulder_twist_change < 10.0
        
        # Criterion 4: Shoulder-elbow line angles (both left and right) unchanged or changed less than 2 degrees
        left_shoulder_elbow_change = abs(current_features.get('left_shoulder_elbow_angle', 0) - compare_features.get('left_shoulder_elbow_angle', 0))
        right_shoulder_elbow_change = abs(current_features.get('right_shoulder_elbow_angle', 0) - compare_features.get('right_shoulder_elbow_angle', 0))
        criteria['shoulder_elbow_stable'] = left_shoulder_elbow_change < 2.0 and right_shoulder_elbow_change < 2.0
        
        # Criterion 5: Batsman back not fully or partially towards camera
        # Check shoulder line twist - if > 45 degrees, back might be towards camera
        shoulder_twist = current_features.get('shoulder_line_twist', 0)
        criteria['camera_perspective_ok'] = abs(shoulder_twist) < 45.0
        
        return criteria
    
    def _calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points."""
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = abs(radians * 180.0 / math.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def _check_stance_criteria(self, pose_data: Dict) -> bool:
        """Check if current pose meets batting stance criteria."""
        # Require a minimum stance score of 75%
        stance_score = pose_data.get('stance_score', 0)
        return stance_score > 0.75  # More than 75% of criteria met
    
    def _check_stability(self, current_timestamp: float) -> bool:
        """Check if pose has been stable for minimum duration."""
        if len(self.pose_history) < 2:
            return False
        
        # Calculate movement between recent frames
        recent_movements = []
        
        for i in range(1, len(self.pose_history)):
            movement = self._calculate_movement(
                self.pose_history[i-1]['landmarks'],
                self.pose_history[i]['landmarks']
            )
            recent_movements.append(movement)
        
        # Store movement in history
        if recent_movements:
            self.movement_history.append({
                'timestamp': current_timestamp,
                'movement': recent_movements[-1]
            })
        
        # Check stability over required duration
        stable_start_time = None
        for movement_data in reversed(self.movement_history):
            if movement_data['movement'] <= self.stability_threshold:
                if stable_start_time is None:
                    stable_start_time = movement_data['timestamp']
            else:
                break
        
        if stable_start_time is not None:
            stable_duration = current_timestamp - stable_start_time
            return stable_duration >= self.min_stability_duration
        
        return False
    
    def _calculate_movement(self, landmarks1: Dict, landmarks2: Dict) -> float:
        """Calculate movement between two pose landmark sets."""
        movements = []
        
        # Focus on body landmarks (exclude wrists for bat tapping allowance)
        stable_landmarks = [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]
        
        for landmark_name in stable_landmarks:
            if landmark_name in landmarks1 and landmark_name in landmarks2:
                l1 = landmarks1[landmark_name]
                l2 = landmarks2[landmark_name]
                
                # Calculate 3D distance
                dx = l1['x'] - l2['x']
                dy = l1['y'] - l2['y']
                dz = l1['z'] - l2['z']
                
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                movements.append(distance)
        
        # Return average movement
        return float(np.mean(movements)) if movements else float('inf')
    
    def get_stable_periods(self, results: List[Dict]) -> Dict:
        """
        Post-process results to identify stable periods.
        New criteria: 20 continuous frames with >85% stance score, then skip 60 frames.
        
        Args:
            results: List of frame analysis results with stance_score
            
        Returns:
            Dictionary with stable periods and statistics
        """
        stable_periods = []
        i = 0
        
        while i < len(results):
            # Check if we have at least 20 frames left to analyze
            if i + 20 > len(results):
                break
            
            # Check 20 continuous frames for >85% stance score
            high_score_count = 0
            valid_period = True
            
            for j in range(i, min(i + 20, len(results))):
                # Get stance score from result (need to recalculate or store during analysis)
                if 'stance_score' in results[j]:
                    stance_score = results[j]['stance_score']
                else:
                    # If stance_score not stored, use is_stable_stance as fallback
                    stance_score = 0.85 if results[j]['is_stable_stance'] else 0.5
                
                if stance_score > 0.85:
                    high_score_count += 1
                else:
                    valid_period = False
                    break
            
            # If all 20 frames have >85% stance score, it's a stable period
            if valid_period and high_score_count == 20:
                start_frame = i
                start_time = results[i]['timestamp']
                
                # Find the end of this stable period (continue until score drops below 85%)
                end_frame = i + 19  # At least 20 frames
                for j in range(i + 20, len(results)):
                    if 'stance_score' in results[j]:
                        stance_score = results[j]['stance_score']
                    else:
                        stance_score = 0.85 if results[j]['is_stable_stance'] else 0.5
                    
                    if stance_score > 0.85:
                        end_frame = j
                    else:
                        break
                
                end_time = results[end_frame]['timestamp']
                duration = end_time - start_time
                
                # Calculate average confidence for this period
                period_confidences = [
                    results[j]['pose_confidence'] 
                    for j in range(start_frame, end_frame + 1)
                    if results[j]['pose_confidence'] > 0
                ]
                avg_confidence = np.mean(period_confidences) if period_confidences else 0
                
                # Calculate average stance score for this period
                period_scores = [
                    results[j].get('stance_score', 0.85 if results[j]['is_stable_stance'] else 0.5)
                    for j in range(start_frame, end_frame + 1)
                ]
                avg_stance_score = np.mean(period_scores) if period_scores else 0
                
                # Analyze ankle coordinate changes within this stable period
                ankle_changes = self._analyze_ankle_movement(results, start_frame, end_frame)
                
                stable_periods.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'avg_confidence': avg_confidence,
                    'avg_stance_score': avg_stance_score,
                    'frame_count': end_frame - start_frame + 1,
                    'ankle_movement': ankle_changes
                })
                
                # Skip 30 frames after finding a stable period
                i = end_frame + 30 + 1
            else:
                # Move to next frame
                i += 1
        
        # Analyze shot triggers after stable period detection
        shot_triggers = self._analyze_shot_triggers(results)
        
        return {
            'stable_periods': stable_periods,
            'shot_triggers': shot_triggers,
            'all_frames': results,
            'total_stable_time': sum(p['duration'] for p in stable_periods),
            'stability_percentage': len([r for r in results if r['is_stable_stance']]) / len(results) * 100 if results else 0
        }
    
    def _calculate_weighted_center_of_gravity(self, landmarks):
        """
        Calculate center of gravity using weighted body segments.
        
        Body segment weights and specifications:
        - Head/Neck (8%): Uses shoulder center point as reference, estimates head position 50 pixels above shoulders
        - Torso (50%): Averages shoulder and hip landmark positions, represents core body mass from shoulders to hips
        - Arms (12%): Uses elbow positions when available, falls back to wrists, averages both arms together
        - Upper Legs (20%): Averages hip and knee landmark positions, represents thigh mass from hips to knees
        - Lower Legs (10%): Averages knee and ankle landmark positions, represents calf/shin mass from knees to ankles
        
        Final CoG Formula:
        CoG_x = (Σ(segment_x × weight)) / total_weight
        CoG_y = (Σ(segment_y × weight)) / total_weight
        """
        try:
            import mediapipe as mp
            
            # Get core landmark positions (required)
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Check for arm landmarks availability
            try:
                left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
                right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
                has_elbows = True
            except:
                has_elbows = False
                
            try:
                left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
                has_wrists = True
            except:
                has_wrists = False
            
            # Initialize weighted segments list
            segments = []
            
            # 1. Head/Neck (8%) - Uses shoulder center point as reference, estimates head position 50 pixels above shoulders
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            # Estimate head position 50 pixels above shoulders (in normalized coordinates, approximately 0.05)
            head_x = shoulder_center_x
            head_y = shoulder_center_y - 0.05  # Move up in normalized coordinates
            segments.append({'x': head_x, 'y': head_y, 'weight': 0.08})
            
            # 2. Torso (50%) - Averages shoulder and hip landmark positions, represents core body mass
            torso_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
            torso_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
            segments.append({'x': torso_x, 'y': torso_y, 'weight': 0.50})
            
            # 3. Arms (12%) - Uses elbow positions when available, falls back to wrists, averages both arms together
            if has_elbows:
                # Use elbows for better arm positioning in batting stance
                arms_x = (left_elbow.x + right_elbow.x) / 2
                arms_y = (left_elbow.y + right_elbow.y) / 2
            elif has_wrists:
                # Fallback to wrist positions
                arms_x = (left_wrist.x + right_wrist.x) / 2
                arms_y = (left_wrist.y + right_wrist.y) / 2
            else:
                # Ultimate fallback to shoulder positions
                arms_x = shoulder_center_x
                arms_y = shoulder_center_y
            segments.append({'x': arms_x, 'y': arms_y, 'weight': 0.12})
            
            # 4. Upper Legs (20%) - Averages hip and knee landmark positions, represents thigh mass from hips to knees
            upper_legs_x = (left_hip.x + right_hip.x + left_knee.x + right_knee.x) / 4
            upper_legs_y = (left_hip.y + right_hip.y + left_knee.y + right_knee.y) / 4
            segments.append({'x': upper_legs_x, 'y': upper_legs_y, 'weight': 0.20})
            
            # 5. Lower Legs (10%) - Averages knee and ankle landmark positions, represents calf/shin mass from knees to ankles
            lower_legs_x = (left_knee.x + right_knee.x + left_ankle.x + right_ankle.x) / 4
            lower_legs_y = (left_knee.y + right_knee.y + left_ankle.y + right_ankle.y) / 4
            segments.append({'x': lower_legs_x, 'y': lower_legs_y, 'weight': 0.10})
            
            # Calculate weighted center of gravity using the formula: CoG = (Σ(segment × weight)) / total_weight
            total_weight = sum(segment['weight'] for segment in segments)
            cog_x = sum(segment['x'] * segment['weight'] for segment in segments) / total_weight
            cog_y = sum(segment['y'] * segment['weight'] for segment in segments) / total_weight
            
            return {
                'cog_x': cog_x,
                'cog_y': cog_y,
                'method': 'weighted_segments',
                'segments_used': len(segments),
                'has_elbows': has_elbows,
                'has_wrists': has_wrists
            }
            
        except Exception as e:
            # Fallback method - simple hip center calculation
            try:
                import mediapipe as mp
                left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
                
                cog_x = (left_hip.x + right_hip.x) / 2
                cog_y = (left_hip.y + right_hip.y) / 2
                
                return {
                    'cog_x': cog_x,
                    'cog_y': cog_y,
                    'method': 'hip_center_fallback',
                    'error': str(e)
                }
            except:
                # Ultimate fallback to default center
                return {
                    'cog_x': 0.5,
                    'cog_y': 0.5,
                    'method': 'default_center',
                    'error': 'No landmarks available'
                }
