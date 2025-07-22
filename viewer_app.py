import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
import psycopg2
from psycopg2.extras import DictCursor

# Assuming your other python files are in the same directory
from video_processor import VideoProcessor
from utils import annotate_frame_with_pose
from stance_detector import StanceDetector # Needed for drawing utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Cricket Analysis Viewer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Cricket Analysis Viewer & Debug Tool")
st.markdown("This application reads processed data from the database and visualizes the results.")

# --- Database Helper Functions ---

@st.cache_resource
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using credentials from st.secrets."""
    try:
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def get_players(conn):
    """Gets a list of all players from the database."""
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute('SELECT * FROM "players" ORDER BY name;')
        results = cur.fetchall()
        return results

def get_processed_videos_for_player(conn, player_id):
    """Gets a list of processed videos for a specific player."""
    with conn.cursor(cursor_factory=DictCursor) as cur:
        # Find videos for the player that have entries in the analysis table
        cur.execute("""
            SELECT v.video_id, v.title, v.url, s.splitvideo_id
            FROM "videos" v
            JOIN "splitvideo_dtls" s ON v.video_id = s.video_id
            WHERE v.player_id = %s AND s.splitvideo_id IN (SELECT DISTINCT splitvideo_id FROM "stance_biomech_base")
            ORDER BY v.title;
        """, (player_id,))
        results = cur.fetchall()
        return results

def get_analysis_data(conn, splitvideo_id):
    """Gets all analysis data for a specific video ID from the database."""
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute('SELECT * FROM "stance_biomech_base" WHERE splitvideo_id = %s ORDER BY frame_id;', (splitvideo_id,))
        results = cur.fetchall()
        return results

# --- Main App Logic ---
db_conn = get_db_connection()

if not db_conn:
    st.warning("Application cannot run without a database connection.")
else:
    st.sidebar.header("Selection")
    players = get_players(db_conn)
    
    if not players:
        st.info("No players found in the database. Please add players and process their videos.")
    else:
        # --- Player Selection Dropdown ---
        player_options = {p['player_id']: p['name'] for p in players}
        selected_player_id = st.sidebar.selectbox("Choose a Player", options=player_options.keys(), format_func=lambda x: player_options[x])
        
        if selected_player_id:
            # --- Video Selection Dropdown (filtered by player) ---
            processed_videos = get_processed_videos_for_player(db_conn, selected_player_id)
            
            if not processed_videos:
                st.sidebar.warning("No processed videos found for this player.")
            else:
                video_options = {v['splitvideo_id']: v['title'] for v in processed_videos}
                selected_splitvideo_id = st.sidebar.selectbox("Choose a Video", options=video_options.keys(), format_func=lambda x: video_options[x])

                if selected_splitvideo_id:
                    # Find the full path for the selected video
                    video_path = next((v['url'] for v in processed_videos if v['splitvideo_id'] == selected_splitvideo_id), None)
                    
                    st.header(f"Viewing Analysis for: {video_options[selected_splitvideo_id]}")
                    st.write(f"**Player:** `{player_options[selected_player_id]}`")

                    if not video_path or not os.path.exists(video_path):
                        st.error("The video file for this analysis could not be found at the stored path.")
                    else:
                        analysis_data = get_analysis_data(db_conn, selected_splitvideo_id)
                        
                        if not analysis_data:
                            st.warning("No analysis data found for this video ID.")
                        else:
                            video_proc = VideoProcessor(video_path)
                            
                            st.subheader("🖼️ Frame-by-Frame Analysis Gallery")
                            st.markdown("A gallery of the processed frames with landmarks read from the database.")

                            cols_per_row = 4
                            temp_detector = StanceDetector()
                            
                            frames_to_show = analysis_data
                            st.info(f"Displaying all {len(analysis_data)} analyzed frames.")

                            landmark_names = [
                                'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
                                'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
                                'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
                                'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                                'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
                            ]

                            for i in range(0, len(frames_to_show), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j in range(cols_per_row):
                                    frame_index = i + j
                                    if frame_index < len(frames_to_show):
                                        with cols[j]:
                                            frame_data = frames_to_show[frame_index]
                                            frame_number = frame_data['frame_id']
                                            
                                            frame_image = video_proc.get_frame_at_index(frame_number)
                                            
                                            if frame_image is not None:
                                                from mediapipe.framework.formats import landmark_pb2
                                                drawable_landmarks = landmark_pb2.NormalizedLandmarkList()
                                                
                                                for base_name in landmark_names:
                                                    if frame_data[f'{base_name}_x'] is not None:
                                                        lm = drawable_landmarks.landmark.add()
                                                        lm.x = frame_data[f'{base_name}_x']
                                                        lm.y = frame_data[f'{base_name}_y']
                                                        lm.z = frame_data[f'{base_name}_z']
                                                        lm.visibility = frame_data[f'{base_name}_visibility']

                                                annotated_frame = annotate_frame_with_pose(
                                                    frame_image, drawable_landmarks,
                                                    temp_detector.mp_pose, temp_detector.mp_drawing
                                                )
                                                
                                                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_number}")
                                                
                                                confidence = frame_data.get('pose_confidence', 0)
                                                if confidence is not None:
                                                    if confidence > 0.7: st.success(f"Conf: {confidence:.2%}")
                                                    elif confidence > 0.4: st.warning(f"Conf: {confidence:.2%}")
                                                    else: st.error(f"Conf: {confidence:.2%}")
                                            else:
                                                st.error(f"Frame {frame_number} error")
