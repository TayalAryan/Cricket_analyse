import streamlit as st
import cv2
import numpy as np
import os
import math
import mediapipe as mp
from PIL import Image
import plotly.graph_objects as go

# Assuming your other python files are in the same directory
from stance_detector import StanceDetector
from video_processor import VideoProcessor
from utils import annotate_frame_with_pose

# --- Page Configuration ---
st.set_page_config(
    page_title="Admin - Cricket Analysis",
    page_icon="👑",
    layout="wide"
)

st.title("👑 Admin - Cricket Stance Analysis")
st.markdown("Select a customer and video to perform a detailed frame-by-frame analysis.")

# --- Data Source Setup ---
# We assume a directory named 'customer_data' exists.
# Each sub-folder in 'customer_data' is a unique customer.
DATA_DIR = "customer_data"

def get_customers():
    """Scans the data directory for customer folders."""
    if not os.path.exists(DATA_DIR) or not os.path.isdir(DATA_DIR):
        return []
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

def get_videos_for_customer(customer_name):
    """Gets all video files for a given customer."""
    customer_path = os.path.join(DATA_DIR, customer_name)
    if not os.path.exists(customer_path):
        return []
    return [f for f in os.listdir(customer_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# --- Session State Initialization ---
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_customer' not in st.session_state:
    st.session_state.selected_customer = None
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None

# --- Sidebar for Selection and Configuration ---
with st.sidebar:
    st.header("Selection")

    # 1. Customer Selection
    customers = get_customers()
    if not customers:
        st.error(f"No customer folders found. Please create a '{DATA_DIR}' directory and add customer folders with videos inside.")
    else:
        selected_customer = st.selectbox("Select Customer", customers, index=0)
        if selected_customer != st.session_state.selected_customer:
            # Reset video selection if customer changes
            st.session_state.selected_customer = selected_customer
            st.session_state.selected_video = None
            st.session_state.video_processor = None
            st.session_state.analysis_complete = False
            st.rerun()

    # 2. Video Selection
    if st.session_state.selected_customer:
        videos = get_videos_for_customer(st.session_state.selected_customer)
        if not videos:
            st.warning(f"No videos found for {st.session_state.selected_customer}.")
        else:
            selected_video = st.selectbox("Select Video for Analysis", videos)
            if selected_video != st.session_state.selected_video:
                st.session_state.selected_video = selected_video
                video_path = os.path.join(DATA_DIR, st.session_state.selected_customer, selected_video)
                try:
                    with st.spinner(f"Loading {selected_video}..."):
                        st.session_state.video_processor = VideoProcessor(video_path)
                        st.session_state.analysis_complete = False
                    st.success(f"Loaded: {selected_video}")
                except Exception as e:
                    st.error(f"Failed to load video: {e}")
                    st.session_state.video_processor = None
                st.rerun()

    st.markdown("---")
    st.header("Analysis Configuration")

    if st.session_state.video_processor:
        # Re-using the configuration options from your original app
        camera_perspective = st.radio(
            "Camera Perspective",
            options=["front", "right", "left"],
            format_func=lambda x: {"front": "Front View", "right": "Bowler on Right", "left": "Bowler on Left"}[x],
            index=0
        )
        confidence_threshold = st.slider("Pose Detection Confidence", 0.3, 0.9, 0.5, 0.1)
        # Add other parameters as needed
    else:
        st.info("Load a video to see configuration options.")

# --- Main App Body ---
if not st.session_state.video_processor:
    st.info("Please select a customer and a video from the sidebar to begin.")

elif st.session_state.video_processor and not st.session_state.analysis_complete:
    # This section handles the analysis setup and execution
    st.header("Analysis Setup")
    
    video_proc = st.session_state.video_processor
    first_frame = video_proc.get_first_frame()
    height, width, _ = first_frame.shape
    
    st.write(f"**Video:** `{st.session_state.selected_video}` | **Customer:** `{st.session_state.selected_customer}`")
    
    # Bounding Box selection logic
    if camera_perspective == 'front':
        st.info("Using fixed bounding box for 'Front View'.")
        x1 = int(width * 0.15)
        y1 = int(height * 0.10)
        x2 = int(width * 0.85)
        y2 = int(height * 0.80)
        st.session_state.rectangle_coords = (x1, y1, x2, y2)
        
        frame_with_rect = first_frame.copy()
        cv2.rectangle(frame_with_rect, (x1, y1), (x2, y2), (0, 255, 0), 3)
        st.image(cv2.cvtColor(frame_with_rect, cv2.COLOR_BGR2RGB), caption="Fixed Analysis Area (Front View)")
    else:
        st.info("Define the analysis area for side views.")
        x1 = st.number_input("X1 (left)", 0, width-1, int(width*0.25))
        y1 = st.number_input("Y1 (top)", 0, height-1, int(height*0.25))
        x2 = st.number_input("X2 (right)", 0, width-1, int(width*0.75))
        y2 = st.number_input("Y2 (bottom)", 0, height-1, int(height*0.75))
        st.session_state.rectangle_coords = (x1, y1, x2, y2)

        frame_with_rect = first_frame.copy()
        cv2.rectangle(frame_with_rect, (x1, y1), (x2, y2), (255, 0, 0), 3)
        st.image(cv2.cvtColor(frame_with_rect, cv2.COLOR_BGR2RGB), caption="Manual Analysis Area")

    if st.button("▶️ Start Full Analysis", type="primary"):
        with st.spinner("Performing frame-by-frame analysis... This may take a while."):
            # Initialize Detector
            stance_detector = StanceDetector(confidence_threshold=confidence_threshold, camera_perspective=camera_perspective)
            
            # Get ROI
            x1, y1, x2, y2 = st.session_state.rectangle_coords
            
            # Full video analysis loop
            all_results = []
            total_frames = video_proc.get_frame_count()
            progress_bar = st.progress(0)
            
            video_proc.reset() # Ensure we start from the beginning
            for i in range(total_frames):
                frame = video_proc.get_next_frame()
                if frame is None:
                    continue
                
                timestamp = i / video_proc.get_fps()
                cropped_frame = frame[y1:y2, x1:x2]
                
                # --- MODIFICATION: Capture all landmarks ---
                # We call detect_stance and store results regardless of confidence
                is_stable, pose_data = stance_detector.detect_stance(cropped_frame, timestamp)
                
                biomech_data = None
                if pose_data and pose_data.get('confidence', 0) > 0: # Store if any pose is detected
                    biomech_data = pose_data
                
                all_results.append({
                    'frame': i,
                    'timestamp': timestamp,
                    'is_stable_stance': is_stable,
                    'pose_confidence': pose_data.get('confidence', 0) if pose_data else 0,
                    'biomech_data': biomech_data
                })
                progress_bar.progress((i + 1) / total_frames)

            st.session_state.stance_results = {'all_frames': all_results}
            st.session_state.analysis_complete = True
            st.success("Analysis Complete!")
            st.rerun()

elif st.session_state.analysis_complete:
    # --- Analysis Results and Debugging UI ---
    st.header("Analysis & Debugging")
    st.write(f"**Showing results for:** `{st.session_state.selected_video}` from `{st.session_state.selected_customer}`")

    # Interactive Frame Debugger
    st.subheader("🔬 Interactive Frame Debugger")
    st.markdown("Use the slider to select a specific frame to see the detected pose and its confidence score.")

    all_results = st.session_state.stance_results['all_frames']
    
    selected_frame_index = st.slider(
        "Select a frame to inspect", 
        min_value=0, 
        max_value=len(all_results) - 1, 
        value=0, 
        step=1
    )

    if all_results and selected_frame_index < len(all_results):
        selected_data = all_results[selected_frame_index]
        frame_number = selected_data['frame']
        timestamp = selected_data['timestamp']
        confidence = selected_data['pose_confidence']
        
        frame_image = st.session_state.video_processor.get_frame_at_index(frame_number)

        if frame_image is not None:
            x1, y1, x2, y2 = st.session_state.rectangle_coords
            cropped_frame = frame_image[y1:y2, x1:x2]

            # --- FIX FOR ATTRIBUTE ERROR ---
            # The 'raw_landmarks' is a list-like object. The drawing utility
            # expects the parent container object. We reconstruct it here before drawing.
            from mediapipe.framework.formats import landmark_pb2
            
            landmark_list = selected_data.get('biomech_data', {}).get('raw_landmarks')
            drawable_landmarks = None
            if landmark_list:
                drawable_landmarks = landmark_pb2.NormalizedLandmarkList()
                drawable_landmarks.landmark.extend(landmark_list)
            # --- END OF FIX ---

            # Use a temporary detector to get drawing utils
            temp_detector = StanceDetector()
            annotated_frame = annotate_frame_with_pose(
                cropped_frame, 
                drawable_landmarks, # Pass the corrected object
                temp_detector.mp_pose, 
                temp_detector.mp_drawing
            )
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Frame #{frame_number} at {timestamp:.2f}s", use_column_width=True)
            with col2:
                st.metric("Pose Confidence", f"{confidence:.2%}")
                if confidence > 0.7:
                    st.success("High confidence detection.")
                elif confidence > 0.4:
                    st.warning("Medium confidence detection.")
                else:
                    st.error("Low confidence detection. Landmarks may be inaccurate.")
        else:
            st.error(f"Could not retrieve frame #{frame_number}.")

    # Placeholder for other charts from the original app
    st.markdown("---")
    st.subheader("Additional Analytics")
    st.info("The other charts and detailed analysis from the main application can be added here, using the `st.session_state.stance_results` data.")
    # You can copy/paste the timeline, cover drive, and other chart sections from your original app.py here.

