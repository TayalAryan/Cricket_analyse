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
        
        return features
    
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
                
                stable_periods.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'avg_confidence': avg_confidence,
                    'avg_stance_score': avg_stance_score,
                    'frame_count': end_frame - start_frame + 1
                })
                
                # Skip 30 frames after finding a stable period
                i = end_frame + 30 + 1
            else:
                # Move to next frame
                i += 1
        
        return {
            'stable_periods': stable_periods,
            'all_frames': results,
            'total_stable_time': sum(p['duration'] for p in stable_periods),
            'stability_percentage': len([r for r in results if r['is_stable_stance']]) / len(results) * 100 if results else 0
        }
