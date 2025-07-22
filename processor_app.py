import streamlit as st
import cv2
import numpy as np
import os
import math
import mediapipe as mp
import psycopg2 # For PostgreSQL connection
import json # To handle landmark data

# Assuming your other python files are in the same directory
from stance_detector import StanceDetector
from video_processor import VideoProcessor
from utils import annotate_frame_with_pose

# --- Page Configuration ---
st.set_page_config(
    page_title="Cricket Analysis Processor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Cricket Analysis Processor")
st.markdown("This application automatically finds videos to process from the database and saves the detailed results.")

# --- Database Helper Functions ---

@st.cache_resource
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using credentials from st.secrets."""
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.error("Please check your database credentials in the .streamlit/secrets.toml file.")
        return None

def get_video_to_process(conn):
    """Reads from 'splitvideo_dtls' to get the next video to be worked on."""
    with conn.cursor() as cur:
        # In a real system, you'd check a 'status' column (e.g., WHERE status = 'unprocessed')
        # For now, we just get the first video found.
        cur.execute('SELECT splitvideo_id, vid_path FROM "splitvideo_dtls" LIMIT 1;')
        result = cur.fetchone()
        return result # Returns (splitvideo_id, vid_path) or None

def delete_existing_analysis(conn, splitvideo_id):
    """Deletes any old analysis data for a given video ID to prevent duplicate errors."""
    with conn.cursor() as cur:
        cur.execute('DELETE FROM "stance_biomech_base" WHERE splitvideo_id = %s;', (splitvideo_id,))
        conn.commit()
    st.info(f"Cleared old analysis data for video ID: {splitvideo_id}")


def insert_stance_biomech_base(conn, splitvideo_id, results):
    """Inserts analysis results into the new 'stance_biomech_base' table."""
    
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
        'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
        'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
        'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
    ]
    
    with conn.cursor() as cur:
        for result in results:
            biomech_data = result.get('biomech_data') or {}
            
            values = {
                "splitvideo_id": int(splitvideo_id),
                "frame_id": int(result['frame']),
                "vid_timestamp": int(result['timestamp'] * 1000),
                "stage": biomech_data.get('stage', ''),
                "pose_confidence": float(result['pose_confidence']) if result.get('pose_confidence') is not None else None,
                "cog_x": float(biomech_data.get('cog_x')) if biomech_data.get('cog_x') is not None else None,
                "cog_y": float(biomech_data.get('cog_y')) if biomech_data.get('cog_y') is not None else None,
                "weight_distribution": biomech_data.get('weight_distribution_text')
            }

            raw_landmarks = biomech_data.get('raw_landmarks')
            if raw_landmarks and len(raw_landmarks) == 33:
                for i, name in enumerate(landmark_names):
                    lm = raw_landmarks[i]
                    values[f'{name}_x'] = float(lm.x) if hasattr(lm, 'x') else None
                    values[f'{name}_y'] = float(lm.y) if hasattr(lm, 'y') else None
                    values[f'{name}_z'] = float(lm.z) if hasattr(lm, 'z') else None
                    values[f'{name}_visibility'] = float(lm.visibility) if hasattr(lm, 'visibility') else None
            else:
                for name in landmark_names:
                    values[f'{name}_x'], values[f'{name}_y'], values[f'{name}_z'], values[f'{name}_visibility'] = None, None, None, None

            columns = ", ".join([f'"{k}"' for k in values.keys()])
            placeholders = ", ".join(["%s"] * len(values))
            sql = f'INSERT INTO "stance_biomech_base" ({columns}) VALUES ({placeholders})'
            
            cur.execute(sql, list(values.values()))
            
        conn.commit()

# --- Main App Logic ---
db_conn = get_db_connection()

if not db_conn:
    st.warning("Application cannot run without a database connection.")
else:
    video_info = get_video_to_process(db_conn)
    
    if not video_info:
        st.info("No videos found in the 'splitvideo_dtls' table to process.")
    else:
        splitvideo_id, video_path = video_info
        st.header("Video to Process")
        st.write(f"**Split Video ID:** `{splitvideo_id}`")
        st.write(f"**Path:** `{video_path}`")

        if not os.path.exists(video_path):
            st.error(f"Video file not found at the specified path. Please check the path in the 'splitvideo_dtls' table.")
        else:
            video_proc = VideoProcessor(video_path)
            
            st.header("Configuration")
            # --- MODIFICATION: Hardcode confidence threshold to 0.0 ---
            confidence_threshold = 0.0 
            st.info(f"Pose detection confidence threshold is fixed at > {confidence_threshold} to capture all detected poses.")
            # --- END OF MODIFICATION ---
            
            if st.button(f"▶️ Process Video #{splitvideo_id}", type="primary"):
                with st.spinner("Clearing any existing analysis data for this video..."):
                    delete_existing_analysis(db_conn, splitvideo_id)

                with st.spinner("Performing frame-by-frame analysis..."):
                    stance_detector = StanceDetector(confidence_threshold=confidence_threshold)
                    all_results = []
                    total_frames = video_proc.get_frame_count()
                    progress_bar = st.progress(0)
                    
                    video_proc.reset()
                    for i in range(total_frames):
                        frame = video_proc.get_next_frame()
                        if frame is None: continue
                        
                        timestamp = i / video_proc.get_fps()
                        
                        is_stable, pose_data = stance_detector.detect_stance(frame, timestamp)
                        
                        biomech_data = pose_data if pose_data and pose_data.get('confidence', 0) >= 0 else None
                        
                        all_results.append({
                            'frame': i, 'timestamp': timestamp, 'is_stable_stance': is_stable,
                            'pose_confidence': pose_data.get('confidence', 0) if pose_data else 0,
                            'biomech_data': biomech_data
                        })
                        progress_bar.progress((i + 1) / total_frames)

                with st.spinner("Saving detailed results to 'stance_biomech_base' table..."):
                    insert_stance_biomech_base(db_conn, splitvideo_id, all_results)
                
                st.success(f"Analysis for video ID {splitvideo_id} complete! Results have been saved to the database.")
