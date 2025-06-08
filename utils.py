import cv2
import numpy as np
from typing import Tuple, List, Optional

def draw_rectangle(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], 
                  color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw a rectangle on an image.
    
    Args:
        image: Input image
        top_left: Top-left corner coordinates (x, y)
        bottom_right: Bottom-right corner coordinates (x, y)
        color: Rectangle color in BGR format
        thickness: Line thickness
        
    Returns:
        Image with rectangle drawn
    """
    result_image = image.copy()
    cv2.rectangle(result_image, top_left, bottom_right, color, thickness)
    return result_image

def calculate_rectangle_bounds(x1: int, y1: int, x2: int, y2: int, 
                             image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    """
    Calculate and validate rectangle bounds within image dimensions.
    
    Args:
        x1: Left x coordinate
        y1: Top y coordinate
        x2: Right x coordinate
        y2: Bottom y coordinate
        image_width: Image width
        image_height: Image height
        
    Returns:
        Validated rectangle coordinates (x1, y1, x2, y2)
    """
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))
    
    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x2 = min(x1 + 10, image_width - 1)
    if y2 <= y1:
        y2 = min(y1 + 10, image_height - 1)
    
    return (x1, y1, x2, y2)

def resize_frame_maintaining_aspect(frame: np.ndarray, target_width: int = 640) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Target width for resized frame
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_height = int(target_width / aspect_ratio)
    
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_frame

def crop_frame(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Crop frame to specified rectangle.
    
    Args:
        frame: Input frame
        x1: Left x coordinate
        y1: Top y coordinate
        x2: Right x coordinate
        y2: Bottom y coordinate
        
    Returns:
        Cropped frame
    """
    return frame[y1:y2, x1:x2]

def calculate_movement_score(landmarks1: dict, landmarks2: dict, 
                           key_points: List[str] = None) -> float:
    """
    Calculate movement score between two sets of landmarks.
    
    Args:
        landmarks1: First set of landmarks
        landmarks2: Second set of landmarks
        key_points: List of key landmark names to focus on
        
    Returns:
        Movement score (0 = no movement, higher = more movement)
    """
    if key_points is None:
        key_points = list(landmarks1.keys())
    
    movements = []
    
    for point_name in key_points:
        if point_name in landmarks1 and point_name in landmarks2:
            p1 = landmarks1[point_name]
            p2 = landmarks2[point_name]
            
            # Calculate 3D Euclidean distance
            dx = p1['x'] - p2['x']
            dy = p1['y'] - p2['y']
            dz = p1['z'] - p2['z']
            
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            movements.append(distance)
    
    return np.mean(movements) if movements else 0.0

def smooth_boolean_sequence(sequence: List[bool], window_size: int = 5) -> List[bool]:
    """
    Smooth a boolean sequence to reduce noise.
    
    Args:
        sequence: Input boolean sequence
        window_size: Size of smoothing window
        
    Returns:
        Smoothed boolean sequence
    """
    if len(sequence) < window_size:
        return sequence
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(sequence)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(sequence), i + half_window + 1)
        
        window_values = sequence[start_idx:end_idx]
        # Use majority vote
        smoothed_value = sum(window_values) > len(window_values) // 2
        smoothed.append(smoothed_value)
    
    return smoothed

def filter_short_periods(stable_periods: List[dict], min_duration: float = 0.5) -> List[dict]:
    """
    Filter out stable periods that are too short.
    
    Args:
        stable_periods: List of stable period dictionaries
        min_duration: Minimum duration in seconds
        
    Returns:
        Filtered list of stable periods
    """
    return [period for period in stable_periods if period['duration'] >= min_duration]

def merge_close_periods(stable_periods: List[dict], max_gap: float = 0.2) -> List[dict]:
    """
    Merge stable periods that are close together.
    
    Args:
        stable_periods: List of stable period dictionaries
        max_gap: Maximum gap in seconds to merge
        
    Returns:
        List with merged periods
    """
    if len(stable_periods) <= 1:
        return stable_periods
    
    merged = []
    current_period = stable_periods[0].copy()
    
    for next_period in stable_periods[1:]:
        gap = next_period['start_time'] - current_period['end_time']
        
        if gap <= max_gap:
            # Merge periods
            current_period['end_time'] = next_period['end_time']
            current_period['end_frame'] = next_period['end_frame']
            current_period['duration'] = current_period['end_time'] - current_period['start_time']
            
            # Update average confidence
            total_duration = current_period['duration'] + next_period['duration']
            if total_duration > 0:
                current_period['avg_confidence'] = (
                    (current_period['avg_confidence'] * current_period['duration'] + 
                     next_period['avg_confidence'] * next_period['duration']) / total_duration
                )
        else:
            # Add current period and start new one
            merged.append(current_period)
            current_period = next_period.copy()
    
    # Add the last period
    merged.append(current_period)
    
    return merged

def validate_video_format(file_path: str) -> bool:
    """
    Validate if the video file can be processed.
    
    Args:
        file_path: Path to video file
        
    Returns:
        True if video is valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    except Exception:
        return False

def convert_frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame from BGR to RGB format.
    
    Args:
        frame: Input frame in BGR format
        
    Returns:
        Frame in RGB format
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def annotate_frame_with_pose(frame: np.ndarray, landmarks, mp_pose, mp_drawing) -> np.ndarray:
    """
    Annotate frame with pose landmarks.
    
    Args:
        frame: Input frame
        landmarks: Pose landmarks
        mp_pose: MediaPipe pose object
        mp_drawing: MediaPipe drawing utilities
        
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    if landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
        )
    
    return annotated_frame
