import cv2
import numpy as np
from typing import Optional, Tuple

class VideoProcessor:
    def __init__(self, video_path: str):
        """
        Initialize video processor.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
    
    def get_first_frame(self) -> Optional[np.ndarray]:
        """Get the first frame of the video."""
        # Save current position
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Go to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        
        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        if ret:
            return frame
        return None
    
    def get_next_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from the video."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            return frame
        return None
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Get frame at specific timestamp."""
        frame_number = int(timestamp * self.fps)
        return self.get_frame_at_index(frame_number)
    
    def get_frame_at_index(self, frame_index: int) -> Optional[np.ndarray]:
        """Get frame at specific index."""
        if 0 <= frame_index < self.frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def reset(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
    
    def get_fps(self) -> float:
        """Get video FPS."""
        return self.fps
    
    def get_duration(self) -> float:
        """Get video duration in seconds."""
        return self.duration
    
    def get_frame_count(self) -> int:
        """Get total frame count."""
        return self.frame_count
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get video resolution (width, height)."""
        return (self.width, self.height)
    
    def get_current_timestamp(self) -> float:
        """Get current timestamp in seconds."""
        return self.current_frame / self.fps if self.fps > 0 else 0
    
    def seek_to_timestamp(self, timestamp: float):
        """Seek to specific timestamp."""
        frame_number = int(timestamp * self.fps)
        self.seek_to_frame(frame_number)
    
    def seek_to_frame(self, frame_number: int):
        """Seek to specific frame."""
        if 0 <= frame_number < self.frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
    
    def extract_frames_in_range(self, start_time: float, end_time: float, step: float = 1.0) -> list:
        """
        Extract frames in a time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            step: Time step between frames in seconds
            
        Returns:
            List of (timestamp, frame) tuples
        """
        frames = []
        current_time = start_time
        
        while current_time <= end_time and current_time <= self.duration:
            frame = self.get_frame_at_time(current_time)
            if frame is not None:
                frames.append((current_time, frame))
            current_time += step
        
        return frames
    
    def get_video_info(self) -> dict:
        """Get comprehensive video information."""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'duration': self.duration,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'resolution': f"{self.width}x{self.height}",
            'aspect_ratio': self.width / self.height if self.height > 0 else 0
        }
    
    def __del__(self):
        """Clean up video capture."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def close(self):
        """Explicitly close video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
