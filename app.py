import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import math
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from stance_detector import StanceDetector
from video_processor import VideoProcessor
from utils import draw_rectangle, calculate_rectangle_bounds

# Configure page
st.set_page_config(
    page_title="Cricket Stance Detection",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 Cricket Batsman Stance Detection")
st.markdown("Upload a cricket video to detect stable batting stances. Configure the camera perspective based on which side the bowler is positioned.")

# Add reset button at the top
if st.button("🔄 Reset Application", help="Clear all data and start fresh"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Initialize session state
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None
if 'rectangle_coords' not in st.session_state:
    st.session_state.rectangle_coords = None
if 'first_frame' not in st.session_state:
    st.session_state.first_frame = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'stance_results' not in st.session_state:
    st.session_state.stance_results = None
if 'cricket_events' not in st.session_state:
    st.session_state.cricket_events = {
        'trigger': None,
        'swing_start': None,
        'bat_ball_connect': None
    }



# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    
    # Cricket Events Timing - Show if video is loaded
    if st.session_state.video_processor is not None:
        st.subheader("Cricket Events Timing")
        
        # Get video duration for validation
        video_duration = st.session_state.video_processor.get_duration()
        
        # Trigger Point
        trigger_time = st.number_input(
            "Trigger (seconds)",
            min_value=0.0,
            max_value=video_duration,
            value=st.session_state.cricket_events.get('trigger', 0.0) or 0.0,
            step=0.01,
            format="%.2f",
            key="sidebar_trigger_input",
            help="When the batsman initiates the shot trigger movement"
        )
        st.session_state.cricket_events['trigger'] = trigger_time if trigger_time > 0.0 else None
        
        # Swing Start
        swing_time = st.number_input(
            "Swing Start (seconds)",
            min_value=0.0,
            max_value=video_duration,
            value=st.session_state.cricket_events.get('swing_start', 0.0) or 0.0,
            step=0.01,
            format="%.2f",
            key="sidebar_swing_input",
            help="When the batsman begins the forward swing motion"
        )
        st.session_state.cricket_events['swing_start'] = swing_time if swing_time > 0.0 else None
        
        # Bat-Ball Connect
        contact_time = st.number_input(
            "Bat-Ball Connect (seconds)",
            min_value=0.0,
            max_value=video_duration,
            value=st.session_state.cricket_events.get('bat_ball_connect', 0.0) or 0.0,
            step=0.01,
            format="%.2f",
            key="sidebar_contact_input",
            help="When the bat makes contact with the ball"
        )
        st.session_state.cricket_events['bat_ball_connect'] = contact_time if contact_time > 0.0 else None
        
        # Show current settings
        events_count = sum(1 for v in st.session_state.cricket_events.values() if v is not None)
        if events_count > 0:
            st.success(f"{events_count} event(s) set")
        else:
            st.info("No events specified")
        
        st.markdown("---")
    
    # Camera perspective configuration
    st.subheader("Camera Perspective")
    camera_perspective = st.radio(
        "Bowler Position",
        options=["right", "left"],
        format_func=lambda x: "Bowler on Right" if x == "right" else "Bowler on Left",
        index=0,
        help="Select which side of the camera the bowler is positioned"
    )
    
    # Stance detection parameters
    st.subheader("Detection Parameters")
    stability_threshold = st.slider("Movement Threshold", 0.01, 0.1, 0.03, 0.01, 
                                   help="Lower values require less movement for stable stance")
    min_stability_duration = st.slider("Minimum Stance Duration (ms)", 50, 1000, 100, 25,
                                      help="Minimum time to maintain stable stance")
    confidence_threshold = st.slider("Pose Detection Confidence", 0.3, 0.9, 0.5, 0.1,
                                    help="Minimum confidence for pose detection")
    
    # Batsman height configuration
    st.subheader("Batsman Details")
    batsman_height = st.number_input(
        "Batsman Height (feet)",
        min_value=2.0,
        max_value=7.0,
        value=5.5,
        step=0.1,
        help="Height of the batsman in feet. Used to adjust stance width criteria for younger players."
    )
    
    # Fixed body segment weights for CoG calculation
    cog_weights = {
        'head': 0.08,      # 8%
        'torso': 0.50,     # 50%
        'arms': 0.12,      # 12%
        'upper_legs': 0.20, # 20%
        'lower_legs': 0.10  # 10%
    }

# Enhanced file upload with multiple methods
st.markdown("### Upload Cricket Video")

# Method 1: Direct drag-and-drop area
st.markdown("**Method 1: Direct Upload**")
uploaded_file = st.file_uploader(
    "Drag and drop your cricket video file here", 
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Supports files up to 1000MB",
    key="main_uploader"
)

# Method 2: URL-based upload (for large files)
st.markdown("**Method 2: URL Upload** (for files >100MB)")
video_url = st.text_input("Enter video URL (Google Drive, Dropbox, etc.):", placeholder="https://...")
if video_url and st.button("Download from URL"):
    try:
        import urllib.request
        import uuid
        
        st.info("Downloading video from URL...")
        unique_id = str(uuid.uuid4())[:8]
        temp_filename = f"cricket_video_{unique_id}.mp4"
        video_path = os.path.join('/tmp', temp_filename)
        
        urllib.request.urlretrieve(video_url, video_path)
        
        file_size = os.path.getsize(video_path) / (1024*1024)
        st.session_state.temp_video_path = video_path
        st.session_state.video_processor = None
        st.success(f"Downloaded successfully: {file_size:.1f} MB")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to download from URL: {str(e)}")

# Method 3: Existing files fallback
with st.expander("Method 3: Use Previously Uploaded Files"):
    existing_files = []
    try:
        import glob
        existing_files = glob.glob('/tmp/cricket_video_*.mp4') + glob.glob('/tmp/*.mp4')
        existing_files = [f for f in existing_files if os.path.getsize(f) > 1024*1024]  # Only files >1MB
    except:
        pass
    
    if existing_files:
        st.markdown(f"Found {len(existing_files)} video file(s):")
        for idx, file_path in enumerate(existing_files):
            file_size = os.path.getsize(file_path) / (1024*1024)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{os.path.basename(file_path)} ({file_size:.1f} MB)")
            with col2:
                if st.button("Use", key=f"use_file_{idx}"):
                    st.session_state.temp_video_path = file_path
                    st.session_state.video_processor = None
                    st.success(f"Using: {os.path.basename(file_path)}")
                    st.rerun()
    else:
        st.info("No video files found")

# Process uploaded file if present
if uploaded_file is not None:
    # Clean up any existing temporary files first
    if hasattr(st.session_state, 'temp_video_path') and st.session_state.temp_video_path:
        try:
            if os.path.exists(st.session_state.temp_video_path):
                os.unlink(st.session_state.temp_video_path)
        except:
            pass
    
    # Enhanced file processing with chunked writing for large files
    try:
        file_size_mb = uploaded_file.size / (1024*1024)
        
        # Reject files that are too large for memory processing
        if file_size_mb > 500:
            st.error(f"File too large ({file_size_mb:.1f} MB). Please use Method 2 (URL Upload) for files over 500MB.")
            st.stop()
        
        st.info(f"Processing video file: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Create a unique filename
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        file_extension = os.path.splitext(uploaded_file.name)[1] or '.mp4'
        temp_filename = f"cricket_video_{unique_id}{file_extension}"
        video_path = os.path.join('/tmp', temp_filename)
        
        # Write file in chunks to handle large files better
        chunk_size = 8192  # 8KB chunks
        total_size = uploaded_file.size
        written = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with open(video_path, 'wb') as f:
            uploaded_file.seek(0)
            while written < total_size:
                chunk = uploaded_file.read(min(chunk_size, total_size - written))
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                progress = written / total_size
                progress_bar.progress(progress)
                status_text.text(f"Uploading: {progress*100:.1f}% ({written/(1024*1024):.1f}/{file_size_mb:.1f} MB)")
        
        progress_bar.empty()
        status_text.empty()
        
        # Verify file was written correctly
        if os.path.exists(video_path) and os.path.getsize(video_path) == total_size:
            st.session_state.temp_video_path = video_path
            st.success(f"File uploaded successfully: {file_size_mb:.1f} MB")
        else:
            raise Exception("File verification failed - incomplete upload")
            
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        st.markdown("**Alternative solutions:**")
        st.markdown("- Use Method 2 (URL Upload) for reliable large file handling")
        st.markdown("- Use Method 3 if file was partially uploaded")
        st.markdown("- Try smaller file chunks or compress the video")
        st.stop()
    
    # Initialize video processor
    if st.session_state.video_processor is None:
        with st.spinner("Loading video..."):
            try:
                st.session_state.video_processor = VideoProcessor(video_path)
                st.session_state.first_frame = st.session_state.video_processor.get_first_frame()
                st.success(f"Video loaded successfully! Duration: {st.session_state.video_processor.get_duration():.1f}s, FPS: {st.session_state.video_processor.get_fps():.1f}")
            except Exception as e:
                st.error(f"Error loading video: {str(e)}")
                st.stop()
        


# Continue processing if we have a video loaded (from any method)
if st.session_state.get('temp_video_path') and st.session_state.get('video_processor'):
    
    # Rectangle selection on first frame
    if st.session_state.first_frame is not None and st.session_state.rectangle_coords is None:
        st.subheader("Step 1: Select Analysis Area")
        st.markdown("Click and drag to select the rectangular area where the batsman should be analyzed.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display first frame for rectangle selection
            frame_rgb = cv2.cvtColor(st.session_state.first_frame, cv2.COLOR_BGR2RGB)
            
            # Create a simple rectangle selection interface
            st.markdown("**Instructions:**")
            st.markdown("1. Enter the coordinates for the analysis rectangle")
            st.markdown("2. The rectangle should encompass the batting area near the stumps")
            st.markdown("3. Adjust coordinates based on the frame preview below")
            
            # Show frame dimensions
            height, width = frame_rgb.shape[:2]
            st.info(f"Frame dimensions: {width} x {height} pixels")
            
            # Rectangle coordinate inputs
            col_x1, col_y1, col_x2, col_y2 = st.columns(4)
            with col_x1:
                x1 = st.number_input("X1 (left)", min_value=0, max_value=width-1, value=width//4)
            with col_y1:
                y1 = st.number_input("Y1 (top)", min_value=0, max_value=height-1, value=height//4)
            with col_x2:
                x2 = st.number_input("X2 (right)", min_value=0, max_value=width-1, value=3*width//4)
            with col_y2:
                y2 = st.number_input("Y2 (bottom)", min_value=0, max_value=height-1, value=3*height//4)
            
            # Draw rectangle on frame
            frame_with_rect = frame_rgb.copy()
            cv2.rectangle(frame_with_rect, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            
            # Display frame with rectangle
            st.image(frame_with_rect, caption="First frame with analysis area", use_container_width=True)
            
            if st.button("Confirm Selection", type="primary"):
                if x2 > x1 and y2 > y1:
                    st.session_state.rectangle_coords = (int(x1), int(y1), int(x2), int(y2))
                    st.success("Analysis area selected successfully!")
                    st.rerun()
                else:
                    st.error("Invalid rectangle coordinates. X2 must be greater than X1, and Y2 must be greater than Y1.")
        
        with col2:
            st.markdown("**Rectangle Selection Tips:**")
            st.markdown("- Focus on the batting crease area")
            st.markdown("- Include space for batsman movement")
            st.markdown("- Exclude spectators and background")
            st.markdown("- Leave some margin around the expected batting position")
    
    # Stance detection
    elif st.session_state.rectangle_coords is not None:
        st.subheader("Step 3: Stance Detection Analysis")
        
        st.info("Analyzing entire video for stance detection")
        
        if not st.session_state.analysis_complete:
            if st.button("Start Analysis", type="primary"):
                x1, y1, x2, y2 = st.session_state.rectangle_coords
                
                # Initialize stance detector
                stance_detector = StanceDetector(
                    stability_threshold=stability_threshold,
                    min_stability_duration=min_stability_duration,
                    confidence_threshold=confidence_threshold,
                    camera_perspective=camera_perspective,
                    batsman_height=batsman_height,
                    cog_weights=cog_weights
                )
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Full video analysis
                    fps = st.session_state.video_processor.get_fps()
                    video_duration = st.session_state.video_processor.get_duration()
                    frames_to_analyze = list(range(0, int(video_duration * fps)))
                    
                    # Process all frames
                    results = []
                    total_frames = len(frames_to_analyze)
                    
                    status_text.text(f"Analyzing {total_frames} frames...")
                    
                    for i, frame_idx in enumerate(frames_to_analyze):
                        # Get frame at specific index
                        frame = st.session_state.video_processor.get_frame_at_index(frame_idx)
                        if frame is None:
                            continue
                        
                        # Update progress
                        progress = i / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {i + 1}/{total_frames} (Frame #{frame_idx})")
                        
                        # Crop frame to analysis area
                        cropped_frame = frame[y1:y2, x1:x2]
                        
                        # Detect stance
                        timestamp = frame_idx / fps
                        is_stable_stance, pose_data = stance_detector.detect_stance(cropped_frame, timestamp)
                        
                        # Store comprehensive biomechanical data for shot trigger analysis
                        biomech_data = None
                        if pose_data and pose_data.get('confidence', 0) > 0.5:
                            # Pass through all stance detector features directly
                            biomech_data = pose_data.copy()
                            
                            # Calculate center of gravity from key body landmarks
                            # Use weighted average of major body points
                            left_hip_x = pose_data.get('left_hip_x', 0.5)
                            left_hip_y = pose_data.get('left_hip_y', 0.5)
                            right_hip_x = pose_data.get('right_hip_x', 0.5)
                            right_hip_y = pose_data.get('right_hip_y', 0.5)
                            left_shoulder_x = pose_data.get('left_shoulder_x', 0.5)
                            left_shoulder_y = pose_data.get('left_shoulder_y', 0.3)
                            right_shoulder_x = pose_data.get('right_shoulder_x', 0.5)
                            right_shoulder_y = pose_data.get('right_shoulder_y', 0.3)
                            left_elbow_x = pose_data.get('left_elbow_x', 0)
                            left_elbow_y = pose_data.get('left_elbow_y', 0)
                            left_wrist_x = pose_data.get('left_wrist_x', 0)
                            left_wrist_y = pose_data.get('left_wrist_y', 0)
                            
                            # Center of gravity approximation using torso center
                            # Weighted more towards hips (lower body mass)
                            cog_x = (left_hip_x + right_hip_x + left_shoulder_x + right_shoulder_x) / 4
                            cog_y = (left_hip_y * 0.6 + right_hip_y * 0.6 + left_shoulder_y * 0.4 + right_shoulder_y * 0.4) / 2
                            
                            # Calculate distance from center of gravity to right foot
                            right_ankle_x = pose_data.get('right_ankle_x', 0)
                            right_ankle_y = pose_data.get('right_ankle_y', 0)
                            cog_to_right_foot = ((cog_x - right_ankle_x)**2 + (cog_y - right_ankle_y)**2)**0.5
                            
                            # Calculate left foot-head gap (X coordinate distance)
                            left_ankle_x = pose_data.get('left_ankle_x', 0)
                            head_x = (left_shoulder_x + right_shoulder_x) / 2  # Use shoulder center as head X position
                            left_foot_head_gap = abs(left_ankle_x - head_x)
                            
                            biomech_data = {
                                'left_ankle_x': pose_data.get('left_ankle_x', 0),
                                'left_ankle_y': pose_data.get('left_ankle_y', 0),
                                'right_ankle_x': right_ankle_x,
                                'right_ankle_y': right_ankle_y,
                                'left_hip_x': left_hip_x,
                                'left_hip_y': left_hip_y,
                                'right_hip_x': right_hip_x,
                                'right_hip_y': right_hip_y,
                                'left_shoulder_x': left_shoulder_x,
                                'left_shoulder_y': left_shoulder_y,
                                'right_shoulder_x': right_shoulder_x,
                                'right_shoulder_y': right_shoulder_y,
                                'left_elbow_x': left_elbow_x,
                                'left_elbow_y': left_elbow_y,
                                'left_wrist_x': left_wrist_x,
                                'left_wrist_y': left_wrist_y,
                                'shoulder_line_angle': pose_data.get('shoulder_line_angle', 0),
                                'hip_line_angle': pose_data.get('hip_line_angle', 0),
                                'shoulder_twist_hip': pose_data.get('shoulder_twist_hip', 0),
                                'head_position': pose_data.get('head_position', 0),
                                'head_tilt_angle': pose_data.get('head_tilt_angle', 0),
                                'left_knee_angle': pose_data.get('left_knee_angle', 170),
                                'right_knee_angle': pose_data.get('right_knee_angle', 170),
                                'center_of_gravity_x': cog_x,
                                'center_of_gravity_y': cog_y,
                                'cog_to_right_foot': cog_to_right_foot,
                                'weight_distribution': pose_data.get('weight_distribution', 0),
                                'weight_distribution_text': pose_data.get('weight_distribution_text', 'Unknown'),
                                'cog_x': pose_data.get('cog_x', cog_x),
                                'cog_y': pose_data.get('cog_y', cog_y),
                                'stance_width': pose_data.get('stance_width', 0),
                                'stance_center_x': pose_data.get('stance_center_x', 0),
                                'cog_offset_from_center': pose_data.get('cog_offset_from_center', 0),
                                'cog_distance_from_center': pose_data.get('cog_distance_from_center', 0),
                                'balanced_threshold': pose_data.get('balanced_threshold', 0),
                                'left_foot_distance': pose_data.get('left_foot_distance', 0),
                                'right_foot_distance': pose_data.get('right_foot_distance', 0),
                                'cog_method': pose_data.get('cog_method', 'fallback'),
                                'left_foot_head_gap': left_foot_head_gap
                            }
                        
                        results.append({
                            'frame': frame_idx,
                            'timestamp': timestamp,
                            'is_stable_stance': is_stable_stance,
                            'pose_confidence': pose_data.get('confidence', 0) if pose_data else 0,
                            'stance_score': pose_data.get('stance_score', 0) if pose_data else 0,
                            'ankle_coords': biomech_data,  # Keep same name for compatibility
                            'biomech_data': biomech_data
                        })
                        
                        frame_idx += 1
                    
                    # Post-process results to find stable periods
                    st.session_state.stance_results = stance_detector.get_stable_periods(results)
                    
                    # Analyze shot triggers
                    st.session_state.shot_triggers = stance_detector._analyze_shot_triggers(results)
                    
                    # Detect batting stance taken events
                    video_fps = st.session_state.video_processor.get_fps()
                    st.session_state.batting_stances = stance_detector.detect_batting_stance(results, video_fps)
                    
                    st.session_state.analysis_complete = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("Analysis complete!")
                    st.success("Stance detection analysis completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
        
        else:
            # Display results
            st.subheader("Analysis Results")
            
            # Add re-analyze button to clear cached data
            if st.button("🔄 Re-analyze Video", help="Force re-analysis with updated calculations"):
                # Clear all cached results
                st.session_state.analysis_complete = False
                st.session_state.stance_results = None
                st.rerun()
            
            if st.session_state.stance_results:
                stable_periods = st.session_state.stance_results['stable_periods']
                shot_triggers = st.session_state.stance_results.get('shot_triggers', [])
                all_results = st.session_state.stance_results['all_frames']
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Stable Periods Found", len(stable_periods))
                
                with col2:
                    total_stable_time = sum(period['duration'] for period in stable_periods)
                    st.metric("Total Stable Time", f"{total_stable_time:.1f}s")
                
                with col3:
                    video_duration = st.session_state.video_processor.get_duration()
                    stability_percentage = (total_stable_time / video_duration) * 100 if video_duration > 0 else 0
                    st.metric("Stability %", f"{stability_percentage:.1f}%")
                
                with col4:
                    st.metric("Shot Triggers Found", len(shot_triggers))
                
                # Timeline visualization
                st.subheader("Stance Detection Timeline")
                
                # Prepare data for timeline
                timestamps = [r['timestamp'] for r in all_results]
                is_stable = [1 if r['is_stable_stance'] else 0 for r in all_results]
                confidences = [r['pose_confidence'] for r in all_results]
                
                # Use pre-calculated stance scores or estimate from existing data
                # For now, we'll estimate stance scores based on stable stance detection
                # This is much faster than re-analyzing every frame
                stance_scores = []
                for result in all_results:
                    if result['is_stable_stance']:
                        # If stance is stable, assign a good score based on confidence
                        stance_scores.append(min(0.8 + result['pose_confidence'] * 0.2, 1.0))
                    else:
                        # If not stable, assign lower score based on confidence
                        stance_scores.append(result['pose_confidence'] * 0.5)
                
                # Create timeline plot
                fig = go.Figure()
                
                # Add stability timeline
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=is_stable,
                    mode='lines',
                    name='Stable Stance',
                    line=dict(color='green', width=2),
                    fill='tonexty',
                    fillcolor='rgba(0,255,0,0.3)'
                ))
                
                # Add stance score timeline on secondary y-axis
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=stance_scores,
                    mode='lines',
                    name='Stance Score',
                    line=dict(color='red', width=2),
                    yaxis='y2',
                    opacity=0.8
                ))
                
                # Add confidence timeline on secondary y-axis
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=confidences,
                    mode='lines',
                    name='Pose Confidence',
                    line=dict(color='blue', width=1),
                    yaxis='y2',
                    opacity=0.7
                ))
                
                # Mark stable period start points with vertical lines
                for i, period in enumerate(stable_periods):
                    fig.add_vline(
                        x=period['start_time'],
                        line=dict(color="red", width=3, dash="solid"),
                        annotation_text=f"Stable Period {i+1}",
                        annotation_position="top",
                        annotation=dict(
                            font=dict(color="red", size=10),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="red",
                            borderwidth=1
                        )
                    )

                # Mark batting stance detection points
                batting_stances = st.session_state.get('batting_stances', [])
                st.write(f"DEBUG: Found {len(batting_stances)} batting stances to display")
                if batting_stances:
                    st.write("DEBUG: Batting stance timestamps:", [s.get('start_timestamp', 'No timestamp') for s in batting_stances])
                
                for i, stance in enumerate(batting_stances):
                    fig.add_vline(
                        x=stance['start_timestamp'],
                        line=dict(color="purple", width=3, dash="dot"),
                        annotation_text=f"Batting Stance {i+1}",
                        annotation_position="bottom",
                        annotation=dict(
                            font=dict(color="purple", size=10),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="purple",
                            borderwidth=1
                        )
                    )

                # Mark shot trigger points
                shot_triggers = st.session_state.get('shot_triggers', [])
                for i, trigger in enumerate(shot_triggers):
                    fig.add_vline(
                        x=trigger['trigger_time'],
                        line=dict(color="orange", width=2, dash="dash"),
                        annotation_text=f"Shot Trigger {i+1}",
                        annotation_position="top right",
                        annotation=dict(
                            font=dict(color="orange", size=9),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="orange",
                            borderwidth=1
                        )
                    )
                
                # Highlight stable periods with light background
                for period in stable_periods:
                    fig.add_vrect(
                        x0=period['start_time'],
                        x1=period['end_time'],
                        fillcolor="rgba(255,0,0,0.1)",
                        layer="below",
                        line_width=0,
                    )
                
                # Configure layout and axes
                fig.update_layout(
                    title="Cricket Stance Detection Timeline",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Stable Stance (0/1)",
                    yaxis2=dict(
                        title="Stance Score / Pose Confidence",
                        overlaying='y',
                        side='right',
                        range=[0, 1]
                    ),
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed stable periods table
                if stable_periods:
                    st.subheader("Detected Stable Periods")
                    
                    periods_data = []
                    for i, period in enumerate(stable_periods):
                        ankle_movement = period.get('ankle_movement', {})
                        
                        # Format movement points for display
                        movement_display = "No movement"
                        if ankle_movement.get('has_movement', False):
                            movement_points = ankle_movement.get('movement_points', [])
                            if movement_points:
                                timestamps = [f"{mp['timestamp']:.1f}s" for mp in movement_points[:3]]  # Show first 3
                                movement_display = f"At: {', '.join(timestamps)}"
                                if len(movement_points) > 3:
                                    movement_display += f" (+{len(movement_points)-3} more)"
                        
                        periods_data.append({
                            'Period': i + 1,
                            'Start Time (s)': f"{period['start_time']:.2f}",
                            'End Time (s)': f"{period['end_time']:.2f}",
                            'Duration (s)': f"{period['duration']:.2f}",
                            'Frame Count': period.get('frame_count', 0),
                            'Avg Stance Score': f"{period.get('avg_stance_score', 0):.1%}",
                            'Avg Confidence': f"{period['avg_confidence']:.3f}",
                            'Ankle Movement': movement_display,
                            'Max Displacement': f"{ankle_movement.get('max_displacement', 0):.3f}"
                        })
                    
                    st.table(periods_data)
                
                # Batting Stance Detection Results
                batting_stances = st.session_state.get('batting_stances', [])
                if batting_stances:
                    st.subheader("Batting Stance Taken")
                    st.markdown("**Detected moments when batsman achieved stable batting stance with all 5 criteria**")
                    
                    stance_data = []
                    for i, stance in enumerate(batting_stances):
                        # Count frames that passed all criteria
                        criteria_frames = len(stance.get('criteria_details', []))
                        
                        # Extract sample criteria details from first frame
                        first_criteria = {}
                        if stance.get('criteria_details'):
                            first_criteria = stance['criteria_details'][0].get('criteria', {})
                        
                        # Create summary of passed criteria
                        passed_criteria = []
                        if first_criteria.get('ankle_stability'): passed_criteria.append('Ankle Stable')
                        if first_criteria.get('hip_angle_stable'): passed_criteria.append('Hip Angle')
                        if first_criteria.get('shoulder_twist_stable'): passed_criteria.append('Shoulder Twist')
                        if first_criteria.get('shoulder_elbow_stable'): passed_criteria.append('Shoulder-Elbow')
                        if first_criteria.get('camera_perspective_ok'): passed_criteria.append('Camera View')

                        
                        criteria_summary = ', '.join(passed_criteria) if passed_criteria else 'None'
                        
                        stance_data.append({
                            'Stance #': i + 1,
                            'Start Time (s)': f"{stance['start_timestamp']:.2f}",
                            'End Time (s)': f"{stance['end_timestamp']:.2f}",
                            'Duration (ms)': f"{stance['duration']*1000:.0f}",
                            'Window Frames': criteria_frames,
                            'Criteria Met': f"{len(passed_criteria)}/6",
                            'Details': criteria_summary
                        })
                    
                    st.table(stance_data)
                    
                    # Show detailed criteria for each batting stance
                    with st.expander("Detailed Criteria Analysis"):
                        for i, stance in enumerate(batting_stances):
                            st.markdown(f"**Batting Stance #{i+1} at {stance['start_timestamp']:.2f}s**")
                            
                            # Show criteria details for each frame in the window
                            criteria_details = stance.get('criteria_details', [])
                            if criteria_details:
                                criteria_table = []
                                for detail in criteria_details[:5]:  # Show first 5 frames
                                    criteria = detail.get('criteria', {})
                                    criteria_table.append({
                                        'Frame': detail.get('frame_idx', 0),
                                        'Time (s)': f"{detail.get('timestamp', 0):.3f}",
                                        'Ankle Stable': '✅' if criteria.get('ankle_stability') else '❌',
                                        'Hip Angle': '✅' if criteria.get('hip_angle_stable') else '❌',
                                        'Shoulder Twist': '✅' if criteria.get('shoulder_twist_stable') else '❌',
                                        'Shoulder-Elbow': '✅' if criteria.get('shoulder_elbow_stable') else '❌',
                                        'Camera View': '✅' if criteria.get('camera_perspective_ok') else '❌',

                                    })
                                
                                st.table(criteria_table)
                                if len(criteria_details) > 5:
                                    st.caption(f"Showing first 5 frames of {len(criteria_details)} total frames")
                            
                            st.divider()

                # Shot Trigger Analysis Results
                shot_triggers = st.session_state.get('shot_triggers', [])
                if shot_triggers:
                    st.subheader("Shot Trigger Analysis")
                    st.markdown("**Detected moments when batsman initiated shot-making movements**")
                    
                    trigger_data = []
                    for i, trigger in enumerate(shot_triggers):
                        movement_params = [m['parameter'] for m in trigger['movement_details']]
                        param_summary = ', '.join(movement_params[:3])
                        if len(movement_params) > 3:
                            param_summary += f" (+{len(movement_params)-3} more)"
                        
                        trigger_data.append({
                            'Trigger #': i + 1,
                            'Time (s)': f"{trigger['trigger_time']:.2f}",
                            'Duration (ms)': f"{trigger['duration']*1000:.0f}",
                            'Parameters Moved': trigger['parameters_moved'],
                            'Trigger Frames': trigger.get('trigger_frames_count', trigger.get('sustained_frames', 0)),
                            'Key Movements': param_summary
                        })
                    
                    st.table(trigger_data)
                    
                    # Detailed movement analysis
                    with st.expander("Detailed Movement Analysis"):
                        for i, trigger in enumerate(shot_triggers):
                            st.markdown(f"**Trigger {i+1} at {trigger['trigger_time']:.2f}s:**")
                            for movement in trigger['movement_details']:
                                param_name = movement['parameter'].replace('_', ' ').title()
                                change_val = movement['change']
                                threshold = movement['threshold']
                                st.markdown(f"- {param_name}: {change_val:.3f} (threshold: {threshold})")
                else:
                    st.subheader("Shot Trigger Analysis")
                    st.info("No shot triggers detected in this video. This indicates the batsman maintained a stable stance throughout without initiating shot movements.")
                
                # Show sample frames with pose markers every 3 seconds
                st.subheader("Sample Frames with Biomechanical Markers")
                st.markdown("Red dots show body landmarks, green lines show skeletal connections")
                
                # Get frames at 3-second intervals
                sample_interval = 3.0  # seconds
                video_duration = st.session_state.video_processor.get_duration()
                sample_timestamps = list(np.arange(0, video_duration, sample_interval))
                
                if sample_timestamps:
                    # Create a pose detector for sample frame visualization
                    sample_detector = StanceDetector(
                        stability_threshold=stability_threshold,
                        min_stability_duration=min_stability_duration,
                        confidence_threshold=confidence_threshold,
                        camera_perspective=camera_perspective,
                        batsman_height=batsman_height
                    )
                    
                    # Show progress for sample frame processing
                    with st.spinner("Processing sample frames with pose detection..."):
                        # Create columns for frame display
                        cols_per_row = 3
                        for i in range(0, len(sample_timestamps), cols_per_row):
                            cols = st.columns(cols_per_row)
                            
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(sample_timestamps):
                                    timestamp = sample_timestamps[idx]
                                    
                                    with cols[j]:
                                        try:
                                            # Get frame at timestamp
                                            frame = st.session_state.video_processor.get_frame_at_time(timestamp)
                                            if frame is not None:
                                                # Crop to analysis area
                                                x1, y1, x2, y2 = st.session_state.rectangle_coords
                                                cropped_frame = frame[y1:y2, x1:x2]
                                                
                                                # Process with pose detection
                                                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                                                pose_results = sample_detector.pose.process(rgb_frame)
                                                
                                                # Draw pose landmarks if detected
                                                annotated_frame = rgb_frame.copy()
                                                pose_detected = False
                                                
                                                if hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
                                                    try:
                                                        # Draw pose landmarks
                                                        sample_detector.mp_drawing.draw_landmarks(
                                                            annotated_frame,
                                                            pose_results.pose_landmarks,
                                                            sample_detector.mp_pose.POSE_CONNECTIONS,
                                                            landmark_drawing_spec=sample_detector.mp_drawing.DrawingSpec(
                                                                color=(255, 0, 0), thickness=2, circle_radius=3
                                                            ),
                                                            connection_drawing_spec=sample_detector.mp_drawing.DrawingSpec(
                                                                color=(0, 255, 0), thickness=2
                                                            )
                                                        )
                                                        

                                                        
                                                        pose_detected = True
                                                    except Exception as e:
                                                        st.warning(f"Error drawing pose: {str(e)}")
                                                
                                                # Display frame
                                                st.image(
                                                    annotated_frame, 
                                                    caption=f"Frame at {timestamp:.1f}s", 
                                                    use_container_width=True
                                                )
                                                
                                                # Analyze stance requirements for this frame
                                                if pose_results.pose_landmarks:
                                                    # Get detailed stance analysis
                                                    is_stable_stance, pose_data = sample_detector.detect_stance(cropped_frame, timestamp)
                                                    
                                                    # Calculate movement compared to frame 3 positions earlier
                                                    frame_number = int(timestamp * st.session_state.video_processor.get_fps())
                                                    skip_frames = 3
                                                    earlier_frame_number = frame_number - skip_frames
                                                    
                                                    movement_data = None
                                                    if earlier_frame_number >= 0:
                                                        # Get earlier frame for comparison
                                                        earlier_timestamp = earlier_frame_number / st.session_state.video_processor.get_fps()
                                                        earlier_frame = st.session_state.video_processor.get_frame_at_time(earlier_timestamp)
                                                        
                                                        if earlier_frame is not None:
                                                            # Crop earlier frame
                                                            earlier_cropped = earlier_frame[y1:y2, x1:x2]
                                                            earlier_rgb = cv2.cvtColor(earlier_cropped, cv2.COLOR_BGR2RGB)
                                                            earlier_pose_results = sample_detector.pose.process(earlier_rgb)
                                                            
                                                            if hasattr(earlier_pose_results, 'pose_landmarks') and earlier_pose_results.pose_landmarks:
                                                                # Get biomechanical data for both frames
                                                                _, earlier_pose_data = sample_detector.detect_stance(earlier_cropped, earlier_timestamp)
                                                                
                                                                # Calculate movement parameters
                                                                movement_data = {
                                                                    'shoulder_line_angle': abs(pose_data.get('shoulder_line_angle', 0) - earlier_pose_data.get('shoulder_line_angle', 0)),
                                                                    'hip_line_angle': abs(pose_data.get('hip_line_angle', 0) - earlier_pose_data.get('hip_line_angle', 0)),
                                                                    'shoulder_line_twist': abs(pose_data.get('shoulder_line_twist', 0) - earlier_pose_data.get('shoulder_line_twist', 0)),
                                                                    'hip_line_twist': abs(pose_data.get('hip_line_twist', 0) - earlier_pose_data.get('hip_line_twist', 0)),
                                                                    'knee_to_ankle_angle': max(
                                                                        abs(pose_data.get('left_knee_to_ankle_angle', 0) - earlier_pose_data.get('left_knee_to_ankle_angle', 0)),
                                                                        abs(pose_data.get('right_knee_to_ankle_angle', 0) - earlier_pose_data.get('right_knee_to_ankle_angle', 0))
                                                                    ),
                                                                    'knee_angle': max(
                                                                        abs(pose_data.get('left_knee_angle', 0) - earlier_pose_data.get('left_knee_angle', 0)),
                                                                        abs(pose_data.get('right_knee_angle', 0) - earlier_pose_data.get('right_knee_angle', 0))
                                                                    ),
                                                                    'elbow_wrist_line_angle': max(
                                                                        abs(pose_data.get('left_elbow_wrist_angle', 0) - earlier_pose_data.get('left_elbow_wrist_angle', 0)),
                                                                        abs(pose_data.get('right_elbow_wrist_angle', 0) - earlier_pose_data.get('right_elbow_wrist_angle', 0))
                                                                    ),
                                                                    'ankle_coordinates': max(
                                                                        abs(pose_data.get('left_ankle_x', 0) - earlier_pose_data.get('left_ankle_x', 0)),
                                                                        abs(pose_data.get('left_ankle_y', 0) - earlier_pose_data.get('left_ankle_y', 0)),
                                                                        abs(pose_data.get('right_ankle_x', 0) - earlier_pose_data.get('right_ankle_x', 0)),
                                                                        abs(pose_data.get('right_ankle_y', 0) - earlier_pose_data.get('right_ankle_y', 0))
                                                                    )
                                                                }
                                                    
                                                    # Show overall status
                                                    col_status1, col_status2 = st.columns(2)
                                                    with col_status1:
                                                        if is_stable_stance:
                                                            st.success("✓ Stable Stance")
                                                        else:
                                                            st.info("○ Not Stable")
                                                    
                                                    with col_status2:
                                                        if pose_detected:
                                                            st.success("✓ Pose Detected")
                                                        else:
                                                            st.warning("⚠ No Pose")
                                                    
                                                    # Show detailed stance requirements
                                                    st.markdown("**Stance Requirements:**")
                                                    
                                                    # Create two columns for requirements
                                                    req_col1, req_col2 = st.columns(2)
                                                    
                                                    with req_col1:
                                                        # Shoulder alignment
                                                        if pose_data.get('shoulder_alignment', False):
                                                            st.markdown("✅ Shoulders facing camera")
                                                        else:
                                                            st.markdown("❌ Shoulders not aligned")
                                                        
                                                        # Knees bent
                                                        if pose_data.get('knees_bent', False):
                                                            st.markdown("✅ Knees slightly bent")
                                                        else:
                                                            st.markdown("❌ Knees not properly bent")
                                                        
                                                        # Feet parallel
                                                        if pose_data.get('feet_parallel', False):
                                                            st.markdown("✅ Feet parallel")
                                                        else:
                                                            st.markdown("❌ Feet not parallel")
                                                    
                                                    with req_col2:
                                                        # Body facing camera
                                                        if pose_data.get('body_facing_camera', False):
                                                            st.markdown("✅ Body facing camera")
                                                        else:
                                                            st.markdown("❌ Body not facing camera")
                                                        
                                                        # Head facing bowler
                                                        if pose_data.get('head_facing_bowler', False):
                                                            st.markdown("✅ Head facing 30° right")
                                                        else:
                                                            st.markdown("❌ Head not facing 30° right")
                                                        
                                                        # Stance width
                                                        if pose_data.get('stance_width_good', False):
                                                            st.markdown("✅ Good stance width")
                                                        else:
                                                            st.markdown("❌ Poor stance width")
                                                        
                                                        # Hip line parallel
                                                        if pose_data.get('hip_line_parallel', False):
                                                            st.markdown("✅ Hip line parallel")
                                                        else:
                                                            st.markdown("❌ Hip line not parallel")
                                                        
                                                        # Toe line pointer
                                                        if pose_data.get('toe_line_pointer', False):
                                                            st.markdown("✅ Toe line pointer")
                                                        else:
                                                            st.markdown("❌ Toe line pointer")
                                                        
                                                        # Shoulder line
                                                        if pose_data.get('shoulder_line_good', False):
                                                            st.markdown("✅ Shoulder line level")
                                                        else:
                                                            st.markdown("❌ Shoulder line not level")
                                                        
                                                        # Head tilt
                                                        if pose_data.get('head_tilt_good', False):
                                                            st.markdown("✅ Head tilt good")
                                                        else:
                                                            st.markdown("❌ Head tilt excessive")
                                                        

                                                    
                                                    # Show movement parameters (3-frame skip comparison)
                                                    if movement_data:
                                                        st.markdown("**Movement Parameters (vs 3 frames earlier):**")
                                                        
                                                        # Movement thresholds for reference
                                                        thresholds = {
                                                            'shoulder_line_angle': 10,
                                                            'hip_line_angle': 2.5, 
                                                            'shoulder_line_twist': 20,
                                                            'hip_line_twist': 2,
                                                            'knee_to_ankle_angle': 5,
                                                            'knee_angle': 10,
                                                            'elbow_wrist_line_angle': 15,
                                                            'ankle_coordinates': 0.025
                                                        }
                                                        
                                                        # Display movement in table format
                                                        movement_table = []
                                                        time_span = skip_frames / st.session_state.video_processor.get_fps()  # Time between compared frames
                                                        
                                                        for param, change in movement_data.items():
                                                            threshold = thresholds[param]
                                                            exceeds_threshold = change > threshold
                                                            
                                                            # Calculate velocity (change per second)
                                                            velocity = change / time_span
                                                            
                                                            # Format display name
                                                            display_name = param.replace('_', ' ').title()
                                                            if param == 'ankle_coordinates':
                                                                unit = 'norm'
                                                                change_str = f"{change:.3f}"
                                                                threshold_str = f"{threshold:.3f}"
                                                                velocity_str = f"{velocity:.3f}/s"
                                                            else:
                                                                unit = '°'
                                                                change_str = f"{change:.1f}°"
                                                                threshold_str = f"{threshold}°"
                                                                velocity_str = f"{velocity:.1f}°/s"
                                                            
                                                            status = "🔴 TRIGGER" if exceeds_threshold else "🟢 Normal"
                                                            
                                                            movement_table.append({
                                                                'Parameter': display_name,
                                                                'Change': change_str,
                                                                'Velocity': velocity_str,
                                                                'Threshold': threshold_str,
                                                                'Status': status
                                                            })
                                                        
                                                        # Display as table
                                                        st.table(movement_table)
                                                        
                                                        # Count triggered parameters
                                                        triggered_count = sum(1 for param, change in movement_data.items() 
                                                                             if change > thresholds[param])
                                                        
                                                        if triggered_count >= 3:
                                                            st.error(f"⚠️ {triggered_count}/7 parameters exceed thresholds - Potential shot trigger detected!")
                                                        elif triggered_count > 0:
                                                            st.warning(f"ℹ️ {triggered_count}/7 parameters exceed thresholds")
                                                        else:
                                                            st.success("✅ All parameters within normal range")
                                                    
                                                    # Show stance score
                                                    stance_score = pose_data.get('stance_score', 0)
                                                    st.metric("Stance Score", f"{stance_score:.1%}", 
                                                             help="Percentage of stance criteria met")
                                                    
                                                    # Show confidence and measurements
                                                    col_metrics1, col_metrics2 = st.columns(2)
                                                    with col_metrics1:
                                                        confidence = pose_data.get('confidence', 0)
                                                        st.metric("Pose Confidence", f"{confidence:.2f}")
                                                    
                                                    with col_metrics2:
                                                        stance_width_ratio = pose_data.get('stance_width_ratio', 0)
                                                        width_help = "Feet width relative to shoulder width (0.8-1.5x ideal for adults, up to 2.0x for children under 3.1ft)"
                                                        st.metric("Stance Width", f"{stance_width_ratio:.1f}x", 
                                                                 help=width_help)
                                                    
                                                    # Show angles in compact format
                                                    col_angles1, col_angles2, col_angles3, col_angles4, col_angles5 = st.columns(5)
                                                    with col_angles1:
                                                        head_angle = pose_data.get('head_angle', 0)
                                                        st.markdown(f"<p style='font-size:11px; margin:0;'><b>Head Turn:</b> {head_angle:.0f}°</p>", 
                                                                   unsafe_allow_html=True)
                                                    
                                                    with col_angles2:
                                                        head_tilt_angle = pose_data.get('head_tilt_angle', 0)
                                                        st.markdown(f"<p style='font-size:11px; margin:0;'><b>Head Tilt:</b> {head_tilt_angle:.0f}°</p>", 
                                                                   unsafe_allow_html=True)
                                                    
                                                    with col_angles3:
                                                        shoulder_line_angle = pose_data.get('shoulder_line_angle', 0)
                                                        st.markdown(f"<p style='font-size:11px; margin:0;'><b>Shoulder Line:</b> {shoulder_line_angle:.0f}°</p>", 
                                                                   unsafe_allow_html=True)
                                                    
                                                    with col_angles4:
                                                        hip_line_angle = pose_data.get('hip_line_angle', 0)
                                                        st.markdown(f"<p style='font-size:11px; margin:0;'><b>Hip Line:</b> {hip_line_angle:.0f}°</p>", 
                                                                   unsafe_allow_html=True)
                                                    
                                                    with col_angles5:
                                                        left_ankle_toe_angle = pose_data.get('left_ankle_toe_angle', 0)
                                                        right_ankle_toe_angle = pose_data.get('right_ankle_toe_angle', 0)
                                                        st.markdown(f"<p style='font-size:10px; margin:0;'><b>L/R Toe:</b> {left_ankle_toe_angle:.0f}°/{right_ankle_toe_angle:.0f}°</p>", 
                                                                   unsafe_allow_html=True)
                                                    

                                                
                                                else:
                                                    # No pose detected
                                                    st.warning("⚠ No pose detected")
                                                    st.markdown("**Cannot analyze stance requirements without pose detection**")
                                            
                                            else:
                                                st.error(f"Could not load frame at {timestamp:.1f}s")
                                        
                                        except Exception as e:
                                            st.error(f"Error processing frame at {timestamp:.1f}s: {str(e)}")
                
                else:
                    st.warning("No stable batting stances detected in the video. Try adjusting the detection parameters.")
                
                # Cover Drive Profile section
                st.subheader("Cover Drive Profile")
                st.markdown("**Normalized biomechanical parameters over time**")
                
                # Calculate Cover Drive Profile data
                cover_drive_data = []
                timestamps = []
                
                # Store previous frame wrist coordinates for speed calculation
                prev_left_wrist_x = None
                prev_left_wrist_y = None
                
                # Debug information
                total_results = len(all_results)
                results_with_biomech = sum(1 for r in all_results if r.get('biomech_data'))
                results_with_confidence = sum(1 for r in all_results if r.get('pose_confidence', 0) > 0.5)
                
                st.info(f"Debug: Total frames: {total_results} | With biomech data: {results_with_biomech} | With confidence >0.5: {results_with_confidence}")
                
                # Initialize previous frame coordinates for speed calculations
                prev_left_wrist_x = None
                prev_left_wrist_y = None
                debug_wrist_speeds = []
                
                for result in all_results:
                    if result.get('biomech_data') and result['pose_confidence'] > 0.5:
                        biomech_data = result['biomech_data']
                        timestamp = result['timestamp']
                        
                        # 1. Shoulder line angle (with ground)
                        shoulder_angle = biomech_data.get('shoulder_line_angle', 0)
                        
                        # 1.5. Shoulder twist relative to hip line
                        shoulder_twist_hip = biomech_data.get('shoulder_twist_hip', 0)
                        
                        # 1.6. Head X from right foot (head X - right foot X)
                        head_position = biomech_data.get('head_position', 0)
                        
                        # 2. Left foot extension (X-coordinate difference from right ankle)
                        left_ankle_x = biomech_data.get('left_ankle_x', 0)
                        left_ankle_y = biomech_data.get('left_ankle_y', 0)
                        right_ankle_x = biomech_data.get('right_ankle_x', 0)
                        right_ankle_y = biomech_data.get('right_ankle_y', 0)
                        
                        # Calculate X-coordinate difference from right ankle
                        foot_extension = left_ankle_x - right_ankle_x
                        
                        # 3. Left wrist position (X coordinate difference from right foot)
                        left_wrist_x = biomech_data.get('left_wrist_x', 0)
                        left_wrist_y = biomech_data.get('left_wrist_y', 0)
                        left_wrist_position = left_wrist_x - right_ankle_x
                        
                        # 4. Left wrist speed (Euclidean distance from previous frame)
                        left_wrist_speed = 0
                        if prev_left_wrist_x is not None and prev_left_wrist_y is not None:
                            # Calculate Euclidean distance between current and previous wrist position
                            dx = left_wrist_x - prev_left_wrist_x
                            dy = left_wrist_y - prev_left_wrist_y
                            left_wrist_speed = (dx**2 + dy**2)**0.5
                        
                        # Debug: Track wrist speed values
                        debug_wrist_speeds.append({
                            'timestamp': timestamp,
                            'left_wrist_x': left_wrist_x,
                            'left_wrist_y': left_wrist_y,
                            'prev_x': prev_left_wrist_x,
                            'prev_y': prev_left_wrist_y,
                            'speed': left_wrist_speed
                        })
                        
                        # Update previous frame coordinates for next iteration (allow 0,0 coordinates)
                        prev_left_wrist_x = left_wrist_x
                        prev_left_wrist_y = left_wrist_y
                        
                        # 5. Center of gravity distance from right foot
                        cog_to_right_foot = biomech_data.get('cog_to_right_foot', 0)
                        
                        # 6. Left foot-head gap (X coordinate distance)
                        left_foot_head_gap = biomech_data.get('left_foot_head_gap', 0)
                        
                        cover_drive_data.append({
                            'timestamp': timestamp,
                            'shoulder_angle': shoulder_angle,
                            'shoulder_twist_hip': shoulder_twist_hip,
                            'head_position': head_position,
                            'foot_extension': foot_extension,
                            'left_wrist_position': left_wrist_position,
                            'left_wrist_speed': left_wrist_speed,
                            'cog_to_right_foot': cog_to_right_foot,
                            'left_foot_head_gap': left_foot_head_gap
                        })
                        timestamps.append(timestamp)
                
                if cover_drive_data:
                    # Extract data for normalization
                    timestamps = [d['timestamp'] for d in cover_drive_data]
                    shoulder_angles = [d['shoulder_angle'] for d in cover_drive_data]
                    absolute_shoulder_angles = [abs(d['shoulder_angle']) for d in cover_drive_data]
                    shoulder_twist_hip = [d['shoulder_twist_hip'] for d in cover_drive_data]
                    head_positions = [d['head_position'] for d in cover_drive_data]
                    foot_extensions = [d['foot_extension'] for d in cover_drive_data]
                    left_wrist_positions = [d['left_wrist_position'] for d in cover_drive_data]
                    left_wrist_speeds = [d['left_wrist_speed'] for d in cover_drive_data]
                    cog_distances = [d['cog_to_right_foot'] for d in cover_drive_data]
                    left_foot_head_gaps = [d['left_foot_head_gap'] for d in cover_drive_data]
                    
                    # Normalize values to 0-100 scale for visualization
                    def normalize_to_scale(values, target_min=0, target_max=100):
                        if not values or max(values) == min(values):
                            return values
                        min_val, max_val = min(values), max(values)
                        return [(v - min_val) / (max_val - min_val) * (target_max - target_min) + target_min for v in values]
                    
                    def normalize_shoulder_angles_relative_to_first(angles, center=0, scale_range=40):
                        """Calculate shoulder angles relative to first frame, with first frame at 0"""
                        if not angles:
                            return angles
                        
                        # Use first frame as reference (0 degrees)
                        reference_angle = angles[0]
                        relative_angles = [angle - reference_angle for angle in angles]
                        
                        # First frame will be exactly 0, others scaled relative to max change
                        max_abs = max(abs(angle) for angle in relative_angles) if relative_angles else 0
                        if max_abs == 0:
                            return relative_angles  # All zeros
                        
                        # Scale factor to fit within reasonable range while preserving direction
                        scale_factor = scale_range / max_abs
                        return [angle * scale_factor for angle in relative_angles]
                    
                    # Calculate shoulder angles relative to first frame, other parameters normally
                    normalized_shoulder = normalize_shoulder_angles_relative_to_first(shoulder_angles)
                    normalized_shoulder_twist_hip = normalize_shoulder_angles_relative_to_first(shoulder_twist_hip)
                    normalized_head_position = normalize_shoulder_angles_relative_to_first(head_positions)
                    normalized_abs_shoulder = normalize_to_scale(absolute_shoulder_angles)
                    normalized_foot_ext = normalize_to_scale(foot_extensions)
                    normalized_left_wrist = normalize_to_scale(left_wrist_positions)
                    normalized_left_wrist_speed = normalize_to_scale(left_wrist_speeds)
                    normalized_cog = normalize_to_scale(cog_distances)
                    normalized_left_foot_head_gap = normalize_to_scale(left_foot_head_gaps)
                    
                    # Calculate relative shoulder angles and shoulder twist-hip values for CSV
                    relative_shoulder_angles = [shoulder_angles[i] - shoulder_angles[0] if shoulder_angles else 0 for i in range(len(shoulder_angles))]
                    relative_shoulder_twist_hip = [shoulder_twist_hip[i] - shoulder_twist_hip[0] if shoulder_twist_hip else 0 for i in range(len(shoulder_twist_hip))]
                    relative_head_positions = [head_positions[i] - head_positions[0] if head_positions else 0 for i in range(len(head_positions))]
                    
                    # Create the line chart
                    fig = go.Figure()
                    
                    # Add shoulder line angle (relative to first frame)
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_shoulder,
                        mode='lines+markers',
                        name='Shoulder Line Angle',
                        line=dict(color='red', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add shoulder twist-hip
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_shoulder_twist_hip,
                        mode='lines+markers',
                        name='Shoulder Twist-Hip',
                        line=dict(color='darkblue', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add head X from right foot
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_head_position,
                        mode='lines+markers',
                        name='Head X from Right Foot',
                        line=dict(color='green', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add foot extension
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_foot_ext,
                        mode='lines+markers',
                        name='Left Foot Extension',
                        line=dict(color='orange', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add left wrist position (X coordinate difference from right foot)
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_left_wrist,
                        mode='lines+markers',
                        name='Left Wrist Position',
                        line=dict(color='cyan', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add left wrist speed (Euclidean distance between frames)
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_left_wrist_speed,
                        mode='lines+markers',
                        name='Left Wrist Speed',
                        line=dict(color='magenta', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add absolute shoulder line angle
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_abs_shoulder,
                        mode='lines+markers',
                        name='Absolute Shoulder Line Angle',
                        line=dict(color='brown', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add center of gravity distance from right foot
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_cog,
                        mode='lines+markers',
                        name='Center of Gravity Distance',
                        line=dict(color='purple', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add left foot-head gap (8th parameter)
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized_left_foot_head_gap,
                        mode='lines+markers',
                        name='Left Foot-Head Gap',
                        line=dict(color='orange', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add vertical lines for cricket events if specified
                    if st.session_state.cricket_events['trigger'] is not None:
                        fig.add_vline(
                            x=st.session_state.cricket_events['trigger'],
                            line=dict(color="red", width=3, dash="dash"),
                            annotation_text="Trigger",
                            annotation_position="top",
                            annotation=dict(
                                font=dict(color="red", size=12),
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="red",
                                borderwidth=2
                            )
                        )
                    
                    if st.session_state.cricket_events['swing_start'] is not None:
                        fig.add_vline(
                            x=st.session_state.cricket_events['swing_start'],
                            line=dict(color="blue", width=3, dash="dash"),
                            annotation_text="Swing Start",
                            annotation_position="top",
                            annotation=dict(
                                font=dict(color="blue", size=12),
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="blue",
                                borderwidth=2
                            )
                        )
                    
                    if st.session_state.cricket_events['bat_ball_connect'] is not None:
                        fig.add_vline(
                            x=st.session_state.cricket_events['bat_ball_connect'],
                            line=dict(color="green", width=3, dash="dash"),
                            annotation_text="Bat-Ball Connect",
                            annotation_position="top",
                            annotation=dict(
                                font=dict(color="green", size=12),
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="green",
                                borderwidth=2
                            )
                        )
                    
                    fig.update_layout(
                        title="Cover Drive Profile - 8 Key Biomechanical Parameters",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Normalized Scale",
                        hovermode='x unified',
                        showlegend=True,
                        height=500,
                        annotations=[
                            dict(
                                x=0.02, y=0.98,
                                xref="paper", yref="paper",
                                text="Shoulder Angle: 0=starting position, +values=rightward tilt, -values=leftward tilt",
                                showarrow=False,
                                font=dict(size=10, color="gray"),
                                align="left"
                            )
                        ]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation of relative measurement
                    st.info("""
                    **Shoulder Line Relative Measurement:**
                    - **0 (baseline)**: Starting shoulder position (first frame reference)
                    - **Positive values**: More rightward tilt relative to starting position
                    - **Negative values**: More leftward tilt relative to starting position
                    
                    **Shoulder Twist-Hip Analysis:**
                    - Shows shoulder rotation relative to hip line orientation
                    - First frame serves as reference baseline (0)
                    - Indicates torso twist and body positioning changes
                    
                    These relative measurements clearly show biomechanical changes from the initial stance, helping identify preparation and execution phases.
                    """)
                    
                    # CSV Download section
                    st.subheader("Download Cover Drive Data")
                    
                    # Calculate relative shoulder angles for CSV
                    reference_angle = shoulder_angles[0] if shoulder_angles else 0
                    relative_shoulder_angles = [angle - reference_angle for angle in shoulder_angles]
                    
                    # Prepare CSV data with original and relative values
                    csv_data = []
                    for i, data in enumerate(cover_drive_data):
                        csv_data.append({
                            'Frame': i + 1,
                            'Timestamp (s)': f"{data['timestamp']:.3f}",
                            'Shoulder Line Angle (degrees)': f"{data['shoulder_angle']:.2f}",
                            'Shoulder Angle Relative to First Frame (degrees)': f"{relative_shoulder_angles[i]:.2f}",
                            'Absolute Shoulder Line Angle (degrees)': f"{absolute_shoulder_angles[i]:.2f}",
                            'Shoulder Twist-Hip (degrees)': f"{shoulder_twist_hip[i]:.2f}",
                            'Shoulder Twist-Hip Relative to First Frame (degrees)': f"{relative_shoulder_twist_hip[i]:.2f}",
                            'Head X from Right Foot (X-coordinate difference from right foot)': f"{head_positions[i]:.4f}",
                            'Head X from Right Foot Relative to First Frame': f"{relative_head_positions[i]:.4f}",
                            'Left Foot Extension (X-coordinate difference from right foot)': f"{data['foot_extension']:.4f}",
                            'Left Wrist Position (X difference from right foot)': f"{data['left_wrist_position']:.4f}",
                            'Left Wrist Speed (pixels per frame)': f"{data['left_wrist_speed']:.4f}",
                            'Center of Gravity Distance from Right Foot': f"{data['cog_to_right_foot']:.4f}",
                            'Left Foot-Head Gap (X-coordinate distance)': f"{data['left_foot_head_gap']:.4f}",
                            'Normalized Shoulder Angle (Chart Scale)': f"{normalized_shoulder[i]:.2f}",
                            'Normalized Foot Extension (0-100)': f"{normalized_foot_ext[i]:.2f}",
                            'Normalized Left Wrist Position (0-100)': f"{normalized_left_wrist[i]:.2f}",
                            'Normalized Left Wrist Speed (0-100)': f"{normalized_left_wrist_speed[i]:.2f}",
                            'Normalized CoG Distance (0-100)': f"{normalized_cog[i]:.2f}",
                            'Normalized Left Foot-Head Gap (0-100)': f"{normalized_left_foot_head_gap[i]:.2f}"
                        })
                    
                    # Convert to CSV string
                    import io
                    import csv
                    
                    output = io.StringIO()
                    if csv_data:
                        fieldnames = csv_data[0].keys()
                        writer = csv.DictWriter(output, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(csv_data)
                        csv_string = output.getvalue()
                        
                        # Create download button
                        st.download_button(
                            label="📊 Download Cover Drive Profile CSV",
                            data=csv_string,
                            file_name=f"cover_drive_profile_{len(csv_data)}_frames.csv",
                            mime="text/csv",
                            help="Download detailed frame-by-frame cover drive analysis data"
                        )
                        
                        # Show summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_shoulder = sum(shoulder_angles) / len(shoulder_angles)
                            st.metric("Avg Shoulder Angle", f"{avg_shoulder:.1f}°")
                        
                        with col2:
                            avg_extension = sum(foot_extensions) / len(foot_extensions)
                            st.metric("Avg Foot Extension", f"{avg_extension:.3f}")
                        
                        with col3:
                            avg_left_wrist_speed = sum(left_wrist_speeds) / len(left_wrist_speeds) if left_wrist_speeds else 0
                            st.metric("Avg Left Wrist Speed", f"{avg_left_wrist_speed:.2f}")
                        
                        with col4:
                            avg_cog_distance = sum(cog_distances) / len(cog_distances)
                            st.metric("Avg CoG Distance", f"{avg_cog_distance:.3f}")
                    
                    # Debug: Show wrist speed calculation details
                    if debug_wrist_speeds:
                        st.subheader("🔧 Left Wrist Speed Debug Info")
                        st.write(f"Total frames processed: {len(debug_wrist_speeds)}")
                        
                        # Show first 5 frames for debugging
                        debug_df = debug_wrist_speeds[:5]
                        for i, frame_debug in enumerate(debug_df):
                            st.write(f"Frame {i+1}: X={frame_debug['left_wrist_x']:.2f}, Y={frame_debug['left_wrist_y']:.2f}, PrevX={frame_debug['prev_x']}, PrevY={frame_debug['prev_y']}, Speed={frame_debug['speed']:.2f}")
                        
                        # Check if all speeds are 0
                        non_zero_speeds = [d['speed'] for d in debug_wrist_speeds if d['speed'] > 0]
                        st.write(f"Non-zero speeds found: {len(non_zero_speeds)} out of {len(debug_wrist_speeds)}")
                        if non_zero_speeds:
                            st.write(f"Max speed: {max(non_zero_speeds):.2f}, Min speed: {min(non_zero_speeds):.2f}")
                
                else:
                    st.warning("No pose data available for Cover Drive Profile analysis")
                
                # Stance Stability Check section
                st.subheader("Stance Stability Check")
                st.markdown("**Comprehensive biomechanical parameter analysis over time**")
                
                if all_results:
                    # Calculate stance stability data
                    stability_data = []
                    stability_timestamps = []
                    
                    # Get ROI coordinates for pitch end calculation
                    x1, y1, x2, y2 = st.session_state.rectangle_coords
                    pitch_end_x = x1  # Left edge of ROI rectangle
                    pitch_end_y = (y1 + y2) / 2  # Middle height of ROI rectangle
                    
                    debug_frame_count = 0  # Counter for debug output
                    for result in all_results:
                        if result.get('biomech_data') and result['pose_confidence'] > 0.5:
                            biomech_data = result['biomech_data']
                            timestamp = result['timestamp']
                            
                            # Extract existing parameters from biomech_data (same as Cover Drive Profile)
                            shoulder_angle = biomech_data.get('shoulder_line_angle', 0)
                            abs_shoulder_angle = abs(shoulder_angle)
                            shoulder_twist_hip = biomech_data.get('shoulder_twist_hip', 0)
                            
                            # Calculate hip center distance from pitch end
                            left_hip_x = biomech_data.get('left_hip_x', 0)
                            right_hip_x = biomech_data.get('right_hip_x', 0)
                            left_hip_y = biomech_data.get('left_hip_y', 0)
                            right_hip_y = biomech_data.get('right_hip_y', 0)
                            
                            hip_center_x = (left_hip_x + right_hip_x) / 2
                            hip_center_y = (left_hip_y + right_hip_y) / 2
                            hip_distance_from_pitch = ((hip_center_x - pitch_end_x)**2 + (hip_center_y - pitch_end_y)**2)**0.5
                            
                            # Calculate hip line twist from camera (angle connecting left-right hips)
                            # If batsman is perfectly facing camera, this should be zero
                            if debug_frame_count < 3:
                                print(f"DEBUG Frame {debug_frame_count}: left_hip=({left_hip_x:.3f},{left_hip_y:.3f}), right_hip=({right_hip_x:.3f},{right_hip_y:.3f})")
                            
                            if left_hip_x != right_hip_x or left_hip_y != right_hip_y:
                                # Calculate angle of hip line relative to horizontal (camera plane)
                                hip_line_twist = math.degrees(math.atan2(right_hip_y - left_hip_y, right_hip_x - left_hip_x))
                                # Normalize to show deviation from horizontal (camera-facing = 0)
                                hip_line_twist = abs(hip_line_twist)
                                if hip_line_twist > 90:
                                    hip_line_twist = 180 - hip_line_twist
                                if debug_frame_count < 3:
                                    print(f"DEBUG Frame {debug_frame_count}: calculated hip_twist={hip_line_twist:.3f} degrees")
                            else:
                                hip_line_twist = 0
                                if debug_frame_count < 3:
                                    print(f"DEBUG Frame {debug_frame_count}: hip coordinates identical, twist=0")
                            
                            # Calculate head position from pitch end
                            left_shoulder_x = biomech_data.get('left_shoulder_x', 0)
                            right_shoulder_x = biomech_data.get('right_shoulder_x', 0)
                            left_shoulder_y = biomech_data.get('left_shoulder_y', 0)
                            right_shoulder_y = biomech_data.get('right_shoulder_y', 0)
                            
                            head_x = (left_shoulder_x + right_shoulder_x) / 2
                            head_y = (left_shoulder_y + right_shoulder_y) / 2
                            head_distance_from_pitch = ((head_x - pitch_end_x)**2 + (head_y - pitch_end_y)**2)**0.5
                            
                            # Calculate head tilt - angle neck-head line makes with hip-shoulder center line
                            # Shoulder center coordinates
                            shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
                            shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2
                            
                            # Head position (estimate above shoulders)
                            head_x = shoulder_center_x
                            head_y = shoulder_center_y - 0.1  # Estimate head position above shoulders
                            
                            # Body vertical line from hip center to shoulder center
                            body_line_angle = math.degrees(math.atan2(shoulder_center_y - hip_center_y, shoulder_center_x - hip_center_x))
                            
                            # Neck-head line angle
                            neck_head_angle = math.degrees(math.atan2(head_y - shoulder_center_y, head_x - shoulder_center_x))
                            
                            # Head tilt is the angle difference
                            head_tilt = abs(neck_head_angle - body_line_angle)
                            if head_tilt > 90:
                                head_tilt = 180 - head_tilt
                            
                            # Calculate Left Shoulder-elbow line angle (angle with ground)
                            left_shoulder_x = biomech_data.get('left_shoulder_x', 0)
                            left_shoulder_y = biomech_data.get('left_shoulder_y', 0)
                            left_elbow_x = biomech_data.get('left_elbow_x', 0)
                            left_elbow_y = biomech_data.get('left_elbow_y', 0)
                            
                            if (left_shoulder_x != 0 or left_shoulder_y != 0) and (left_elbow_x != 0 or left_elbow_y != 0):
                                left_shoulder_elbow_angle = math.degrees(math.atan2(left_elbow_y - left_shoulder_y, left_elbow_x - left_shoulder_x))
                                # Normalize to 0-180 degrees
                                left_shoulder_elbow_angle = abs(left_shoulder_elbow_angle)
                                if left_shoulder_elbow_angle > 90:
                                    left_shoulder_elbow_angle = 180 - left_shoulder_elbow_angle
                            else:
                                left_shoulder_elbow_angle = 0
                            
                            # Calculate Left Elbow-wrist line angle (angle with ground)
                            left_wrist_x = biomech_data.get('left_wrist_x', 0)
                            left_wrist_y = biomech_data.get('left_wrist_y', 0)
                            
                            if (left_elbow_x != 0 or left_elbow_y != 0) and (left_wrist_x != 0 or left_wrist_y != 0):
                                left_elbow_wrist_angle = math.degrees(math.atan2(left_wrist_y - left_elbow_y, left_wrist_x - left_elbow_x))
                                # Normalize to 0-180 degrees
                                left_elbow_wrist_angle = abs(left_elbow_wrist_angle)
                                if left_elbow_wrist_angle > 90:
                                    left_elbow_wrist_angle = 180 - left_elbow_wrist_angle
                            else:
                                left_elbow_wrist_angle = 0
                            
                            # Extract wrist coordinates for distance calculation
                            left_wrist_x = biomech_data.get('left_wrist_x', 0)
                            left_wrist_y = biomech_data.get('left_wrist_y', 0)
                            
                            # Left wrist distance from pitch end (only if landmark is valid)
                            if left_wrist_x != 0 or left_wrist_y != 0:
                                left_wrist_distance_from_pitch = ((left_wrist_x - pitch_end_x)**2 + (left_wrist_y - pitch_end_y)**2)**0.5
                            else:
                                left_wrist_distance_from_pitch = 0  # Invalid landmark
                            
                            # Ankle distances from pitch end
                            left_ankle_x = biomech_data.get('left_ankle_x', 0)
                            left_ankle_y = biomech_data.get('left_ankle_y', 0)
                            right_ankle_x = biomech_data.get('right_ankle_x', 0)
                            right_ankle_y = biomech_data.get('right_ankle_y', 0)
                            
                            left_ankle_distance_from_pitch = abs(left_ankle_x - pitch_end_x)  # X coordinate distance only
                            right_ankle_distance_from_pitch = abs(right_ankle_x - pitch_end_x)  # X coordinate distance only
                            
                            # Debug ankle calculations
                            if debug_frame_count < 3:
                                print(f"DEBUG Frame {debug_frame_count}: left_ankle_x={left_ankle_x:.3f}, right_ankle_x={right_ankle_x:.3f}, pitch_end_x={pitch_end_x:.3f}")
                                print(f"DEBUG Frame {debug_frame_count}: left_ankle_dist={left_ankle_distance_from_pitch:.3f}, right_ankle_dist={right_ankle_distance_from_pitch:.3f}")
                            
                            # Extract knee angles directly from stance detector calculations
                            left_knee_angle = biomech_data.get('left_knee_angle', 0)
                            right_knee_angle = biomech_data.get('right_knee_angle', 0)
                            
                            # Extract coordinates for debug output
                            left_knee_x = biomech_data.get('left_knee_x', 0)
                            left_knee_y = biomech_data.get('left_knee_y', 0)
                            right_knee_x = biomech_data.get('right_knee_x', 0)
                            right_knee_y = biomech_data.get('right_knee_y', 0)
                            left_elbow_x = biomech_data.get('left_elbow_x', 0)
                            left_elbow_y = biomech_data.get('left_elbow_y', 0)
                            
                            # Debug output for first 3 frames to check data capture
                            if debug_frame_count < 3:
                                print(f"DEBUG Frame {debug_frame_count}: hip_distance={hip_distance_from_pitch:.3f}, hip_twist={hip_line_twist:.3f}, head_distance={head_distance_from_pitch:.3f}, head_tilt={head_tilt:.3f}")
                                print(f"DEBUG Frame {debug_frame_count}: left_shoulder_elbow={left_shoulder_elbow_angle:.3f}, left_elbow_wrist={left_elbow_wrist_angle:.3f}, left_wrist_distance={left_wrist_distance_from_pitch:.3f}")
                                print(f"DEBUG Frame {debug_frame_count}: left_knee={left_knee_angle:.3f}, right_knee={right_knee_angle:.3f}")
                                print(f"DEBUG Frame {debug_frame_count}: left_knee_pos=({left_knee_x:.1f},{left_knee_y:.1f}), right_knee_pos=({right_knee_x:.1f},{right_knee_y:.1f})")
                                print(f"DEBUG Frame {debug_frame_count}: left_elbow_pos=({left_elbow_x:.1f},{left_elbow_y:.1f}), left_wrist_pos=({left_wrist_x:.1f},{left_wrist_y:.1f})")
                            
                            debug_frame_count += 1
                            
                            stability_data.append({
                                'timestamp': timestamp,
                                'shoulder_line_angle': shoulder_angle,
                                'abs_shoulder_line_angle': abs_shoulder_angle,
                                'shoulder_twist_hip': shoulder_twist_hip,
                                'hip_distance_from_pitch': hip_distance_from_pitch,
                                'hip_line_twist': hip_line_twist,
                                'head_distance_from_pitch': head_distance_from_pitch,
                                'head_tilt': head_tilt,
                                'left_shoulder_elbow_angle': left_shoulder_elbow_angle,
                                'left_elbow_wrist_angle': left_elbow_wrist_angle,
                                'left_wrist_distance_from_pitch': left_wrist_distance_from_pitch,
                                'right_ankle_distance_from_pitch': right_ankle_distance_from_pitch,
                                'left_ankle_distance_from_pitch': left_ankle_distance_from_pitch,
                                'left_knee_angle': left_knee_angle,
                                'right_knee_angle': right_knee_angle
                            })
                            stability_timestamps.append(timestamp)
                    
                    if stability_data:
                        st.info("Stance stability data has been processed successfully.")
                    else:
                        st.warning("No pose data available for Stance Stability Check analysis")
                
                # Debug Section - Show processed frames between 1st to 2nd second
                st.markdown("---")
                st.subheader("Debug: Processed Frames (1-2 seconds)")
                
                # Find frames between 1 and 2 seconds
                debug_frames = []
                fps = st.session_state.video_processor.get_fps()
                for result in all_results:
                    timestamp = result['timestamp']
                    if 1.0 <= timestamp <= 2.0:
                        debug_frames.append(result)
                
                if debug_frames:
                    st.info(f"Showing {len(debug_frames)} frames between 1.0 and 2.0 seconds")
                    
                    # Process debug frames with pose landmarks
                    import mediapipe as mp
                    from utils import crop_frame
                    mp_pose = mp.solutions.pose
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    
                    # Initialize pose detector for debug visualization
                    with mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5
                    ) as pose:
                        
                        cols = st.columns(3)  # Show 3 frames per row
                        col_idx = 0
                        
                        for i, result in enumerate(debug_frames[:9]):  # Limit to 9 frames
                            frame_idx = result['frame']
                            timestamp = result['timestamp']
                            
                            # Get frame from video
                            processor = st.session_state.video_processor
                            processor.seek_to_frame(frame_idx)
                            frame = processor.get_next_frame()
                            
                            if frame is not None:
                                # Get ROI coordinates from session state
                                roi = st.session_state.get('roi_coordinates', {})
                                x1, y1, x2, y2 = roi.get('x1', 0), roi.get('y1', 0), roi.get('x2', frame.shape[1]), roi.get('y2', frame.shape[0])
                                
                                # Crop frame to ROI
                                cropped_frame = crop_frame(frame, x1, y1, x2, y2)
                                
                                # Convert to RGB for MediaPipe
                                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                                
                                # Process with MediaPipe pose  
                                pose_results = pose.process(rgb_frame)
                                
                                # Draw landmarks if detected
                                annotated_frame = rgb_frame.copy()
                                if pose_results.pose_landmarks:
                                    # Draw landmarks and connections
                                    mp_drawing.draw_landmarks(
                                        annotated_frame,
                                        pose_results.pose_landmarks,
                                        mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                                            color=(255, 0, 0), thickness=2, circle_radius=2
                                        ),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(
                                            color=(0, 255, 0), thickness=2
                                        )
                                    )
                                
                                # Display frame in column
                                with cols[col_idx]:
                                    st.image(annotated_frame, 
                                           caption=f"Frame {frame_idx} ({timestamp:.2f}s)",
                                           use_container_width=True)
                                    
                                    # Show key measurements and pose detection status
                                    if result.get('biomech_data'):
                                        biomech_data = result['biomech_data']
                                        st.caption(f"Confidence: {result.get('pose_confidence', 0):.2f}")
                                        st.caption(f"Hip Distance: {biomech_data.get('hip_distance', 0):.1f}")
                                        st.caption(f"Hip Twist: {biomech_data.get('hip_twist', 0):.1f}°")
                                    
                                    # Show pose detection status
                                    if pose_results.pose_landmarks:
                                        st.caption("✅ Pose detected")
                                    else:
                                        st.caption("❌ No pose detected")
                                
                                col_idx = (col_idx + 1) % 3
                else:
                    st.warning("No frames found between 1.0 and 2.0 seconds")
                
                # Angles Time Series Chart section
                st.subheader("Angles")
                st.markdown("**Time series analysis of 13 key biomechanical angle measurements**")
                
                if all_results:
                    # Get cricket events timing from session state
                    trigger_time = st.session_state.cricket_events.get('trigger', 0) or 0
                    swing_time = st.session_state.cricket_events.get('swing_start', 0) or 0
                    contact_time = st.session_state.cricket_events.get('bat_ball_connect', 0) or 0
                    
                    # Calculate angles data with trigger point differences
                    angles_data = []
                    angles_timestamps = []
                    
                    # Find trigger point frame for difference calculations
                    trigger_frame_data = None
                    for result in all_results:
                        if result.get('biomech_data') and result['pose_confidence'] > 0.5:
                            if abs(result['timestamp'] - trigger_time) < 0.1:  # Within 0.1s of trigger time
                                trigger_frame_data = result['biomech_data']
                                break
                    
                    # Extract trigger point reference values (if trigger frame found)
                    trigger_left_knee_angle = trigger_frame_data.get('left_knee_angle', 0) if trigger_frame_data else 0
                    trigger_right_knee_angle = trigger_frame_data.get('right_knee_angle', 0) if trigger_frame_data else 0
                    trigger_shoulder_line_angle = trigger_frame_data.get('shoulder_line_angle', 0) if trigger_frame_data else 0
                    trigger_hip_line_angle = trigger_frame_data.get('hip_line_angle', 0) if trigger_frame_data else 0
                    
                    for result in all_results:
                        if result.get('biomech_data') and result['pose_confidence'] > 0.5:
                            biomech_data = result['biomech_data']
                            timestamp = result['timestamp']
                            
                            # 1. Left knee angle change (difference from trigger point)
                            current_left_knee = biomech_data.get('left_knee_angle', 0)
                            left_knee_change = current_left_knee - trigger_left_knee_angle
                            
                            # 2. Right knee angle change (difference from trigger point)  
                            current_right_knee = biomech_data.get('right_knee_angle', 0)
                            right_knee_change = current_right_knee - trigger_right_knee_angle
                            
                            # 3. Shoulder angle change (difference from trigger point)
                            current_shoulder_angle = biomech_data.get('shoulder_line_angle', 0)
                            shoulder_angle_change = current_shoulder_angle - trigger_shoulder_line_angle
                            
                            # 4. Shoulder line tilt with respect to ground
                            shoulder_line_tilt = biomech_data.get('shoulder_line_tilt_with_ground', 0)
                            
                            # 5. Shoulder twist-hip (same as Cover Drive Profile)
                            shoulder_twist_hip = biomech_data.get('shoulder_twist_hip', 0)
                            
                            # 6. Hip angle change (difference from trigger point)
                            current_hip_angle = biomech_data.get('hip_line_angle', 0)
                            hip_angle_change = current_hip_angle - trigger_hip_line_angle
                            
                            # 7. Right upper arm to body angle
                            right_upper_arm_to_body = biomech_data.get('right_upper_arm_to_body_angle', 0)
                            
                            # 8. Right elbow angle
                            right_elbow_angle = biomech_data.get('right_elbow_angle', 0)
                            
                            # 9. Left upper arm to body angle
                            left_upper_arm_to_body = biomech_data.get('left_upper_arm_to_body_angle', 0)
                            
                            # 10. Left elbow angle
                            left_elbow_angle = biomech_data.get('left_elbow_angle', 0)
                            
                            # 11. Body tilt wrt right upper leg
                            body_tilt_right_leg = biomech_data.get('body_tilt_wrt_right_upper_leg', 0)
                            
                            # 12. Body tilt wrt ground
                            body_tilt_ground = biomech_data.get('body_tilt_wrt_ground', 0)
                            
                            # 13. Left forearm angle with ground
                            left_forearm_ground = biomech_data.get('left_forearm_angle_with_ground', 0)
                            
                            angles_data.append({
                                'left_knee_change': left_knee_change,
                                'right_knee_change': right_knee_change,
                                'shoulder_angle_change': shoulder_angle_change,
                                'shoulder_line_tilt': shoulder_line_tilt,
                                'shoulder_twist_hip': shoulder_twist_hip,
                                'hip_angle_change': hip_angle_change,
                                'right_upper_arm_to_body': right_upper_arm_to_body,
                                'right_elbow_angle': right_elbow_angle,
                                'left_upper_arm_to_body': left_upper_arm_to_body,
                                'left_elbow_angle': left_elbow_angle,
                                'body_tilt_right_leg': body_tilt_right_leg,
                                'body_tilt_ground': body_tilt_ground,
                                'left_forearm_ground': left_forearm_ground
                            })
                            angles_timestamps.append(timestamp)
                    
                    if angles_data:
                        # Create the Angles time series chart
                        import plotly.graph_objects as go
                        
                        fig_angles = go.Figure()
                        
                        # Extract data arrays for plotting
                        left_knee_changes = [d['left_knee_change'] for d in angles_data]
                        right_knee_changes = [d['right_knee_change'] for d in angles_data]
                        shoulder_angle_changes = [d['shoulder_angle_change'] for d in angles_data]
                        shoulder_line_tilts = [d['shoulder_line_tilt'] for d in angles_data]
                        shoulder_twist_hips = [d['shoulder_twist_hip'] for d in angles_data]
                        hip_angle_changes = [d['hip_angle_change'] for d in angles_data]
                        right_upper_arm_to_bodies = [d['right_upper_arm_to_body'] for d in angles_data]
                        right_elbow_angles = [d['right_elbow_angle'] for d in angles_data]
                        left_upper_arm_to_bodies = [d['left_upper_arm_to_body'] for d in angles_data]
                        left_elbow_angles = [d['left_elbow_angle'] for d in angles_data]
                        body_tilt_right_legs = [d['body_tilt_right_leg'] for d in angles_data]
                        body_tilt_grounds = [d['body_tilt_ground'] for d in angles_data]
                        left_forearm_grounds = [d['left_forearm_ground'] for d in angles_data]
                        
                        # Add all 13 angle measurements as traces
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=left_knee_changes,
                            mode='lines+markers', name='1. Left Knee Angle Change',
                            line=dict(color='#1f77b4', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=right_knee_changes,
                            mode='lines+markers', name='2. Right Knee Angle Change',
                            line=dict(color='#ff7f0e', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=shoulder_angle_changes,
                            mode='lines+markers', name='3. Shoulder Angle Change',
                            line=dict(color='#2ca02c', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=shoulder_line_tilts,
                            mode='lines+markers', name='4. Shoulder Line Tilt with Ground',
                            line=dict(color='#d62728', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=shoulder_twist_hips,
                            mode='lines+markers', name='5. Shoulder Twist-Hip',
                            line=dict(color='#9467bd', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=hip_angle_changes,
                            mode='lines+markers', name='6. Hip Angle Change',
                            line=dict(color='#8c564b', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=right_upper_arm_to_bodies,
                            mode='lines+markers', name='7. Right Upper Arm to Body Angle',
                            line=dict(color='#e377c2', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=right_elbow_angles,
                            mode='lines+markers', name='8. Right Elbow Angle',
                            line=dict(color='#7f7f7f', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=left_upper_arm_to_bodies,
                            mode='lines+markers', name='9. Left Upper Arm to Body Angle',
                            line=dict(color='#bcbd22', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=left_elbow_angles,
                            mode='lines+markers', name='10. Left Elbow Angle',
                            line=dict(color='#17becf', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=body_tilt_right_legs,
                            mode='lines+markers', name='11. Body Tilt wrt Right Upper Leg',
                            line=dict(color='#ff9896', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=body_tilt_grounds,
                            mode='lines+markers', name='12. Body Tilt wrt Ground',
                            line=dict(color='#c5b0d5', width=2), marker=dict(size=3)
                        ))
                        
                        fig_angles.add_trace(go.Scatter(
                            x=angles_timestamps, y=left_forearm_grounds,
                            mode='lines+markers', name='13. Left Forearm Angle with Ground',
                            line=dict(color='#c49c94', width=2), marker=dict(size=3)
                        ))
                        
                        # Add cricket event points as scatter overlays (same as Cover Drive Profile)
                        if trigger_time > 0:
                            # Find closest data point to trigger time
                            trigger_idx = min(range(len(angles_timestamps)), 
                                             key=lambda i: abs(angles_timestamps[i] - trigger_time))
                            if trigger_idx < len(angles_data):
                                # Add scatter points for all 13 measurements at trigger time
                                trigger_data = angles_data[trigger_idx]
                                fig_angles.add_trace(go.Scatter(
                                    x=[angles_timestamps[trigger_idx]] * 13,
                                    y=[trigger_data['left_knee_change'], trigger_data['right_knee_change'],
                                       trigger_data['shoulder_angle_change'], trigger_data['shoulder_line_tilt'],
                                       trigger_data['shoulder_twist_hip'], trigger_data['hip_angle_change'],
                                       trigger_data['right_upper_arm_to_body'], trigger_data['right_elbow_angle'],
                                       trigger_data['left_upper_arm_to_body'], trigger_data['left_elbow_angle'],
                                       trigger_data['body_tilt_right_leg'], trigger_data['body_tilt_ground'],
                                       trigger_data['left_forearm_ground']],
                                    mode='markers',
                                    name='Trigger Point',
                                    marker=dict(color='red', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        if swing_time > 0:
                            # Find closest data point to swing time
                            swing_idx = min(range(len(angles_timestamps)), 
                                          key=lambda i: abs(angles_timestamps[i] - swing_time))
                            if swing_idx < len(angles_data):
                                swing_data = angles_data[swing_idx]
                                fig_angles.add_trace(go.Scatter(
                                    x=[angles_timestamps[swing_idx]] * 13,
                                    y=[swing_data['left_knee_change'], swing_data['right_knee_change'],
                                       swing_data['shoulder_angle_change'], swing_data['shoulder_line_tilt'],
                                       swing_data['shoulder_twist_hip'], swing_data['hip_angle_change'],
                                       swing_data['right_upper_arm_to_body'], swing_data['right_elbow_angle'],
                                       swing_data['left_upper_arm_to_body'], swing_data['left_elbow_angle'],
                                       swing_data['body_tilt_right_leg'], swing_data['body_tilt_ground'],
                                       swing_data['left_forearm_ground']],
                                    mode='markers',
                                    name='Swing Start',
                                    marker=dict(color='blue', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        if contact_time > 0:
                            # Find closest data point to contact time
                            contact_idx = min(range(len(angles_timestamps)), 
                                            key=lambda i: abs(angles_timestamps[i] - contact_time))
                            if contact_idx < len(angles_data):
                                contact_data = angles_data[contact_idx]
                                fig_angles.add_trace(go.Scatter(
                                    x=[angles_timestamps[contact_idx]] * 13,
                                    y=[contact_data['left_knee_change'], contact_data['right_knee_change'],
                                       contact_data['shoulder_angle_change'], contact_data['shoulder_line_tilt'],
                                       contact_data['shoulder_twist_hip'], contact_data['hip_angle_change'],
                                       contact_data['right_upper_arm_to_body'], contact_data['right_elbow_angle'],
                                       contact_data['left_upper_arm_to_body'], contact_data['left_elbow_angle'],
                                       contact_data['body_tilt_right_leg'], contact_data['body_tilt_ground'],
                                       contact_data['left_forearm_ground']],
                                    mode='markers',
                                    name='Bat-Ball Connect',
                                    marker=dict(color='green', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        fig_angles.update_layout(
                            title="Angles - Time Series Analysis of 13 Biomechanical Measurements",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Angle (degrees)",
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02
                            )
                        )
                        
                        st.plotly_chart(fig_angles, use_container_width=True)
                        
                        # Show summary information
                        st.info(f"""
                        **Angles Chart Information:**
                        - Total frames analyzed: {len(angles_data)}
                        - Time range: {min(angles_timestamps):.2f}s to {max(angles_timestamps):.2f}s
                        - Trigger point reference frame: {trigger_time:.2f}s
                        - Measurements 1-3 and 6 show changes from trigger point
                        - Measurements 4-5 and 7-13 show absolute angle values
                        - All angle measurements in degrees
                        """)
                    
                    else:
                        st.warning("No pose data available for Angles analysis")
                else:
                    st.warning("No video analysis results available for Angles chart")
                
                # Distances Time Series Chart section
                st.subheader("Distances")
                st.markdown("**Time series analysis of 9 key distance measurements**")
                
                if all_results:
                    # Get cricket events timing from session state
                    trigger_time = st.session_state.cricket_events.get('trigger', 0) or 0
                    swing_time = st.session_state.cricket_events.get('swing_start', 0) or 0
                    contact_time = st.session_state.cricket_events.get('bat_ball_connect', 0) or 0
                    
                    # Calculate distances data
                    distances_data = []
                    distances_timestamps = []
                    
                    # Get ball position (assume right edge of ROI at middle height)
                    roi_coords = st.session_state.get('roi_coords', [0, 0, 640, 480])
                    ball_x = roi_coords[2] / 640.0  # Right edge normalized
                    ball_y = (roi_coords[1] + roi_coords[3]) / 2 / 480.0  # Middle height normalized
                    
                    # Get left pitch end (left edge of ROI at middle height)
                    left_pitch_end_x = roi_coords[0] / 640.0  # Left edge normalized
                    
                    for result in all_results:
                        if result.get('biomech_data') and result['pose_confidence'] > 0.5:
                            biomech_data = result['biomech_data']
                            timestamp = result['timestamp']
                            
                            # Extract coordinates
                            left_ankle_x = biomech_data.get('left_ankle_x', 0)
                            left_ankle_y = biomech_data.get('left_ankle_y', 0)
                            right_ankle_x = biomech_data.get('right_ankle_x', 0)
                            right_ankle_y = biomech_data.get('right_ankle_y', 0)
                            left_elbow_x = biomech_data.get('left_elbow_x', 0)
                            left_elbow_y = biomech_data.get('left_elbow_y', 0)
                            
                            # Calculate head position (shoulder center + offset)
                            left_shoulder_x = biomech_data.get('left_shoulder_x', 0)
                            left_shoulder_y = biomech_data.get('left_shoulder_y', 0)
                            right_shoulder_x = biomech_data.get('right_shoulder_x', 0)
                            right_shoulder_y = biomech_data.get('right_shoulder_y', 0)
                            head_x = (left_shoulder_x + right_shoulder_x) / 2
                            head_y = (left_shoulder_y + right_shoulder_y) / 2 - 0.05  # Head above shoulders
                            
                            # 1. Left-right ankle gap (euclidean distance)
                            ankle_gap = ((left_ankle_x - right_ankle_x) ** 2 + (left_ankle_y - right_ankle_y) ** 2) ** 0.5
                            
                            # 2. Left ankle from left-end (X coordinate distance)
                            left_ankle_from_left_end = abs(left_ankle_x - left_pitch_end_x)
                            
                            # 3. Right ankle from left-end (X coordinate distance)
                            right_ankle_from_left_end = abs(right_ankle_x - left_pitch_end_x)
                            
                            # 4. Head X position wrt ball (X coordinate distance)
                            head_x_wrt_ball = abs(head_x - ball_x)
                            
                            # 5. Head Y position wrt ball (Y coordinate distance)
                            head_y_wrt_ball = abs(head_y - ball_y)
                            
                            # 6. Left ankle X position wrt ball (X coordinate distance)
                            left_ankle_x_wrt_ball = abs(left_ankle_x - ball_x)
                            
                            # 7. Left ankle Y position wrt ball (Y coordinate distance)
                            left_ankle_y_wrt_ball = abs(left_ankle_y - ball_y)
                            
                            # 8. Left elbow X position wrt head (signed integer difference)
                            left_elbow_x_wrt_head = int((left_elbow_x - head_x) * 1000)  # Convert to integer (millipixels)
                            
                            # 9. Left elbow Y position wrt head (signed integer difference)
                            left_elbow_y_wrt_head = int((left_elbow_y - head_y) * 1000)  # Convert to integer (millipixels)
                            
                            distances_data.append({
                                'ankle_gap': ankle_gap,
                                'left_ankle_from_left_end': left_ankle_from_left_end,
                                'right_ankle_from_left_end': right_ankle_from_left_end,
                                'head_x_wrt_ball': head_x_wrt_ball,
                                'head_y_wrt_ball': head_y_wrt_ball,
                                'left_ankle_x_wrt_ball': left_ankle_x_wrt_ball,
                                'left_ankle_y_wrt_ball': left_ankle_y_wrt_ball,
                                'left_elbow_x_wrt_head': left_elbow_x_wrt_head,
                                'left_elbow_y_wrt_head': left_elbow_y_wrt_head
                            })
                            distances_timestamps.append(timestamp)
                    
                    if distances_data:
                        # Create the Distances time series chart
                        import plotly.graph_objects as go
                        
                        fig_distances = go.Figure()
                        
                        # Extract data arrays for plotting
                        ankle_gaps = [d['ankle_gap'] for d in distances_data]
                        left_ankle_from_left_ends = [d['left_ankle_from_left_end'] for d in distances_data]
                        right_ankle_from_left_ends = [d['right_ankle_from_left_end'] for d in distances_data]
                        head_x_wrt_balls = [d['head_x_wrt_ball'] for d in distances_data]
                        head_y_wrt_balls = [d['head_y_wrt_ball'] for d in distances_data]
                        left_ankle_x_wrt_balls = [d['left_ankle_x_wrt_ball'] for d in distances_data]
                        left_ankle_y_wrt_balls = [d['left_ankle_y_wrt_ball'] for d in distances_data]
                        left_elbow_x_wrt_heads = [d['left_elbow_x_wrt_head'] for d in distances_data]
                        left_elbow_y_wrt_heads = [d['left_elbow_y_wrt_head'] for d in distances_data]
                        
                        # Add all 9 distance measurements as traces
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=ankle_gaps,
                            mode='lines+markers', name='1. Left-Right Ankle Gap',
                            line=dict(color='#1f77b4', width=2), marker=dict(size=3)
                        ))
                        
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=left_ankle_from_left_ends,
                            mode='lines+markers', name='2. Left Ankle from Left-End',
                            line=dict(color='#ff7f0e', width=2), marker=dict(size=3)
                        ))
                        
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=right_ankle_from_left_ends,
                            mode='lines+markers', name='3. Right Ankle from Left-End',
                            line=dict(color='#2ca02c', width=2), marker=dict(size=3)
                        ))
                        
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=head_x_wrt_balls,
                            mode='lines+markers', name='4. Head X Position wrt Ball',
                            line=dict(color='#d62728', width=2), marker=dict(size=3)
                        ))
                        
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=head_y_wrt_balls,
                            mode='lines+markers', name='5. Head Y Position wrt Ball',
                            line=dict(color='#9467bd', width=2), marker=dict(size=3)
                        ))
                        
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=left_ankle_x_wrt_balls,
                            mode='lines+markers', name='6. Left Ankle X Position wrt Ball',
                            line=dict(color='#8c564b', width=2), marker=dict(size=3)
                        ))
                        
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=left_ankle_y_wrt_balls,
                            mode='lines+markers', name='7. Left Ankle Y Position wrt Ball',
                            line=dict(color='#e377c2', width=2), marker=dict(size=3)
                        ))
                        
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=left_elbow_x_wrt_heads,
                            mode='lines+markers', name='8. Left Elbow X Position wrt Head',
                            line=dict(color='#7f7f7f', width=2), marker=dict(size=3)
                        ))
                        
                        fig_distances.add_trace(go.Scatter(
                            x=distances_timestamps, y=left_elbow_y_wrt_heads,
                            mode='lines+markers', name='9. Left Elbow Y Position wrt Head',
                            line=dict(color='#bcbd22', width=2), marker=dict(size=3)
                        ))
                        
                        # Add cricket event points as scatter overlays
                        if trigger_time > 0:
                            trigger_idx = min(range(len(distances_timestamps)), 
                                             key=lambda i: abs(distances_timestamps[i] - trigger_time))
                            if trigger_idx < len(distances_data):
                                trigger_data = distances_data[trigger_idx]
                                fig_distances.add_trace(go.Scatter(
                                    x=[distances_timestamps[trigger_idx]] * 9,
                                    y=[trigger_data['ankle_gap'], trigger_data['left_ankle_from_left_end'],
                                       trigger_data['right_ankle_from_left_end'], trigger_data['head_x_wrt_ball'],
                                       trigger_data['head_y_wrt_ball'], trigger_data['left_ankle_x_wrt_ball'],
                                       trigger_data['left_ankle_y_wrt_ball'], trigger_data['left_elbow_x_wrt_head'],
                                       trigger_data['left_elbow_y_wrt_head']],
                                    mode='markers',
                                    name='Trigger Point',
                                    marker=dict(color='red', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        if swing_time > 0:
                            swing_idx = min(range(len(distances_timestamps)), 
                                          key=lambda i: abs(distances_timestamps[i] - swing_time))
                            if swing_idx < len(distances_data):
                                swing_data = distances_data[swing_idx]
                                fig_distances.add_trace(go.Scatter(
                                    x=[distances_timestamps[swing_idx]] * 9,
                                    y=[swing_data['ankle_gap'], swing_data['left_ankle_from_left_end'],
                                       swing_data['right_ankle_from_left_end'], swing_data['head_x_wrt_ball'],
                                       swing_data['head_y_wrt_ball'], swing_data['left_ankle_x_wrt_ball'],
                                       swing_data['left_ankle_y_wrt_ball'], swing_data['left_elbow_x_wrt_head'],
                                       swing_data['left_elbow_y_wrt_head']],
                                    mode='markers',
                                    name='Swing Start',
                                    marker=dict(color='blue', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        if contact_time > 0:
                            contact_idx = min(range(len(distances_timestamps)), 
                                            key=lambda i: abs(distances_timestamps[i] - contact_time))
                            if contact_idx < len(distances_data):
                                contact_data = distances_data[contact_idx]
                                fig_distances.add_trace(go.Scatter(
                                    x=[distances_timestamps[contact_idx]] * 9,
                                    y=[contact_data['ankle_gap'], contact_data['left_ankle_from_left_end'],
                                       contact_data['right_ankle_from_left_end'], contact_data['head_x_wrt_ball'],
                                       contact_data['head_y_wrt_ball'], contact_data['left_ankle_x_wrt_ball'],
                                       contact_data['left_ankle_y_wrt_ball'], contact_data['left_elbow_x_wrt_head'],
                                       contact_data['left_elbow_y_wrt_head']],
                                    mode='markers',
                                    name='Bat-Ball Connect',
                                    marker=dict(color='green', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        fig_distances.update_layout(
                            title="Distances - Time Series Analysis of 9 Key Distance Measurements",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Distance (normalized coordinates)",
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02
                            )
                        )
                        
                        st.plotly_chart(fig_distances, use_container_width=True)
                        
                        # Show summary information
                        st.info(f"""
                        **Distances Chart Information:**
                        - Total frames analyzed: {len(distances_data)}
                        - Time range: {min(distances_timestamps):.2f}s to {max(distances_timestamps):.2f}s
                        - Ball position: Right edge of ROI at middle height
                        - Left pitch end: Left edge of ROI at middle height
                        - Head position: Calculated as shoulder center with upward offset
                        - All distance measurements in normalized coordinates (0-1 scale)
                        """)
                    
                    else:
                        st.warning("No pose data available for Distances analysis")
                else:
                    st.warning("No video analysis results available for Distances chart")
                
                # Right Wrist Swing Coordinates Section
                st.subheader("🎯 Right Wrist Swing Coordinates (from 1.8s onwards)")
                st.markdown("**X-Y coordinate trajectory of right wrist movement from 1.8 seconds onwards**")
                
                if all_results:
                    # Extract right wrist coordinates from 1.8 seconds onwards
                    wrist_x_coords = []
                    wrist_y_coords = []
                    frame_numbers = []
                    timestamps = []
                    
                    fps = st.session_state.video_processor.get_fps()
                    start_time = 1.8  # Start from 1.8 seconds
                    start_frame = int(start_time * fps)
                    
                    for i, result in enumerate(all_results):
                        if i >= start_frame and result.get('biomech_data') and result['biomech_data'].get('right_wrist_x') is not None:
                            wrist_x_coords.append(result['biomech_data']['right_wrist_x'])
                            wrist_y_coords.append(result['biomech_data']['right_wrist_y'])
                            frame_numbers.append(i)
                            timestamps.append(i / fps)
                    
                    if len(wrist_x_coords) > 0:
                        # Create X-Y coordinate plot
                        fig_wrist_xy = go.Figure()
                        
                        # Add wrist trajectory line
                        fig_wrist_xy.add_trace(go.Scatter(
                            x=wrist_x_coords,
                            y=wrist_y_coords,
                            mode='lines+markers',
                            name='Right Wrist Path',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4, color='blue')
                        ))
                        
                        # Mark start and end points
                        if len(wrist_x_coords) > 1:
                            fig_wrist_xy.add_trace(go.Scatter(
                                x=[wrist_x_coords[0]],
                                y=[wrist_y_coords[0]],
                                mode='markers',
                                name='Start Position',
                                marker=dict(size=10, color='green', symbol='circle')
                            ))
                            
                            fig_wrist_xy.add_trace(go.Scatter(
                                x=[wrist_x_coords[-1]],
                                y=[wrist_y_coords[-1]],
                                mode='markers',
                                name='End Position',
                                marker=dict(size=10, color='red', symbol='circle')
                            ))
                        
                        # Add cricket event markers if they exist
                        trigger_time = st.session_state.cricket_events.get('trigger', 0) or 0
                        swing_time = st.session_state.cricket_events.get('swing_start', 0) or 0
                        contact_time = st.session_state.cricket_events.get('bat_ball_connect', 0) or 0
                        video_duration = st.session_state.video_processor.get_duration()
                        fps = st.session_state.video_processor.get_fps()
                        
                        events_to_plot = []
                        if trigger_time > 0:
                            events_to_plot.append(('Trigger', trigger_time, 'red'))
                        if swing_time > 0:
                            events_to_plot.append(('Swing Start', swing_time, 'blue'))
                        if contact_time > 0:
                            events_to_plot.append(('Bat-Ball Connect', contact_time, 'green'))
                        
                        for event_name, event_time, event_color in events_to_plot:
                            if event_time is not None and 0 <= event_time <= video_duration:
                                event_frame = int(event_time * fps)
                                if event_frame < len(all_results) and all_results[event_frame].get('biomech_data'):
                                    event_data = all_results[event_frame]['biomech_data']
                                    if event_data.get('right_wrist_x') is not None:
                                        fig_wrist_xy.add_trace(go.Scatter(
                                            x=[event_data['right_wrist_x']],
                                            y=[event_data['right_wrist_y']],
                                            mode='markers',
                                            name=event_name,
                                            marker=dict(size=12, color=event_color, symbol='diamond')
                                        ))
                        
                        fig_wrist_xy.update_layout(
                            title="Right Wrist Movement Pattern (X-Y Coordinates)",
                            xaxis_title="X Coordinate (normalized)",
                            yaxis_title="Y Coordinate (normalized)",
                            height=500,
                            showlegend=True,
                            hovermode='closest'
                        )
                        
                        # Invert Y-axis to match video coordinates (0,0 at top-left)
                        fig_wrist_xy.update_yaxes(autorange="reversed")
                        
                        st.plotly_chart(fig_wrist_xy, use_container_width=True)
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_range = max(wrist_x_coords) - min(wrist_x_coords)
                            st.metric("X-axis Range", f"{x_range:.3f}")
                        with col2:
                            y_range = max(wrist_y_coords) - min(wrist_y_coords)
                            st.metric("Y-axis Range", f"{y_range:.3f}")
                        with col3:
                            total_distance = 0
                            for i in range(1, len(wrist_x_coords)):
                                dx = wrist_x_coords[i] - wrist_x_coords[i-1]
                                dy = wrist_y_coords[i] - wrist_y_coords[i-1]
                                total_distance += (dx**2 + dy**2)**0.5
                            st.metric("Total Path Distance", f"{total_distance:.3f}")
                        
                        # Show time range info
                        if timestamps:
                            st.info(f"Analysis covers {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s ({len(wrist_x_coords)} data points)")
                        else:
                            st.info("Starting analysis from 1.8 seconds onwards")
                    else:
                        st.info("No right wrist coordinate data available for plotting.")
                else:
                    st.warning("No video analysis results available for right wrist coordinates chart")
                
                # Body Landmarks CSV Download section
                st.subheader("📊 Body Landmarks Data Export")
                st.markdown("**Download complete X-Y coordinates for all body landmarks across all frames**")
                
                if all_results:
                    # Prepare CSV data with all body landmarks
                    landmark_csv_data = []
                    
                    for i, result in enumerate(all_results):
                        biomech_data = result.get('biomech_data')
                        timestamp = result['timestamp']
                        
                        # Create row with only basic info - no calculated values
                        row = {
                            'Frame': i,
                            'Timestamp (s)': f"{timestamp:.3f}",
                            'Pose_Confidence': f"{result['pose_confidence']:.3f}"
                        }
                        
                        # Add all 33 MediaPipe landmark coordinates - only raw coordinates, no calculated values
                        if biomech_data:
                            landmarks = biomech_data.get('raw_landmarks')
                            if landmarks:
                                landmark_names = [
                                    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
                                    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                                    'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
                                    'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                                    'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
                                ]
                                
                                for idx, name in enumerate(landmark_names):
                                    if idx < len(landmarks) and landmarks[idx] is not None:
                                        row[f'{name}_X'] = landmarks[idx].x if hasattr(landmarks[idx], 'x') else 0
                                        row[f'{name}_Y'] = landmarks[idx].y if hasattr(landmarks[idx], 'y') else 0
                                        row[f'{name}_Z'] = landmarks[idx].z if hasattr(landmarks[idx], 'z') else 0
                                        row[f'{name}_VISIBILITY'] = landmarks[idx].visibility if hasattr(landmarks[idx], 'visibility') else 0
                                    else:
                                        row[f'{name}_X'] = 0
                                        row[f'{name}_Y'] = 0
                                        row[f'{name}_Z'] = 0
                                        row[f'{name}_VISIBILITY'] = 0
                            else:
                                # No landmarks available - fill with zeros
                                landmark_names = [
                                    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
                                    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                                    'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
                                    'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                                    'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
                                ]
                                for name in landmark_names:
                                    row[f'{name}_X'] = 0
                                    row[f'{name}_Y'] = 0
                                    row[f'{name}_Z'] = 0
                                    row[f'{name}_VISIBILITY'] = 0
                        else:
                            # No biomech data - fill with zeros
                            landmark_names = [
                                'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
                                'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
                                'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                                'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
                            ]
                            for name in landmark_names:
                                row[f'{name}_X'] = 0
                                row[f'{name}_Y'] = 0
                                row[f'{name}_Z'] = 0
                                row[f'{name}_VISIBILITY'] = 0
                        
                        landmark_csv_data.append(row)
                    
                    if landmark_csv_data:
                        # Convert to CSV string
                        import io
                        import csv
                        
                        output = io.StringIO()
                        fieldnames = landmark_csv_data[0].keys()
                        writer = csv.DictWriter(output, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(landmark_csv_data)
                        csv_string = output.getvalue()
                        
                        # Show summary and download button
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Frames", len(landmark_csv_data))
                        with col2:
                            st.metric("Data Points per Frame", "135 fields")
                        
                        st.download_button(
                            label="📥 Download Body Landmarks CSV",
                            data=csv_string,
                            file_name=f"cricket_body_landmarks_{len(landmark_csv_data)}_frames.csv",
                            mime="text/csv",
                            help="Complete X-Y coordinates for all body landmarks across all analyzed frames"
                        )
                        
                        st.info(f"""
                        **CSV Contains:**
                        - {len(landmark_csv_data)} frames of raw MediaPipe pose data (all frames regardless of confidence)
                        - All 33 MediaPipe body landmarks with X, Y, Z coordinates and visibility scores
                        - Frame numbers, timestamps, and pose confidence scores
                        - Only raw landmark coordinates - no calculated values or derived metrics
                        - All coordinates normalized to 0-1 scale relative to video frame
                        """)
                    else:
                        st.warning("No landmark data available for export")
                else:
                    st.warning("No analysis results available for landmarks export")
                
                # Speed Time Series Chart section
                st.subheader("Speed")
                st.markdown("**Time series analysis of movement speeds**")
                
                if all_results:
                    # Get cricket events timing from session state
                    trigger_time = st.session_state.cricket_events.get('trigger', 0) or 0
                    swing_time = st.session_state.cricket_events.get('swing_start', 0) or 0
                    contact_time = st.session_state.cricket_events.get('bat_ball_connect', 0) or 0
                    
                    # Calculate speed data
                    speed_data = []
                    speed_timestamps = []
                    
                    # Variables to store previous frame data for speed calculation
                    prev_left_wrist_x = None
                    prev_left_wrist_y = None
                    prev_shoulder_angle_change = None
                    prev_shoulder_line_tilt = None
                    prev_shoulder_twist_hip = None
                    prev_hip_angle_change = None
                    prev_timestamp = None
                    
                    for result in all_results:
                        if result.get('biomech_data') and result['pose_confidence'] > 0.5:
                            biomech_data = result['biomech_data']
                            timestamp = result['timestamp']
                            
                            # Extract current frame data
                            left_wrist_x = biomech_data.get('left_wrist_x', 0)
                            left_wrist_y = biomech_data.get('left_wrist_y', 0)
                            
                            # Get angle measurements for body angles speed
                            shoulder_angle_change = biomech_data.get('shoulder_angle_change', 0)
                            shoulder_line_tilt = biomech_data.get('shoulder_line_tilt', 0)
                            shoulder_twist_hip = biomech_data.get('shoulder_twist_hip', 0)
                            hip_angle_change = biomech_data.get('hip_angle_change', 0)
                            
                            # Calculate speeds if we have previous frame data
                            if (prev_left_wrist_x is not None and prev_timestamp is not None and 
                                timestamp > prev_timestamp):
                                
                                time_diff = timestamp - prev_timestamp
                                
                                # 1. Left wrist speed (euclidean distance per second)
                                wrist_distance = ((left_wrist_x - prev_left_wrist_x) ** 2 + 
                                                (left_wrist_y - prev_left_wrist_y) ** 2) ** 0.5
                                left_wrist_speed = wrist_distance / time_diff if time_diff > 0 else 0
                                
                                # 2. Body angles speed (average of absolute speeds of 4 key angles)
                                shoulder_angle_speed = abs(shoulder_angle_change - prev_shoulder_angle_change) / time_diff if time_diff > 0 else 0
                                shoulder_tilt_speed = abs(shoulder_line_tilt - prev_shoulder_line_tilt) / time_diff if time_diff > 0 else 0
                                shoulder_twist_speed = abs(shoulder_twist_hip - prev_shoulder_twist_hip) / time_diff if time_diff > 0 else 0
                                hip_angle_speed = abs(hip_angle_change - prev_hip_angle_change) / time_diff if time_diff > 0 else 0
                                
                                body_angles_speed = (shoulder_angle_speed + shoulder_tilt_speed + 
                                                   shoulder_twist_speed + hip_angle_speed) / 4
                                
                                speed_data.append({
                                    'left_wrist_speed': left_wrist_speed,
                                    'body_angles_speed': body_angles_speed
                                })
                                speed_timestamps.append(timestamp)
                            
                            # Store current frame data for next iteration
                            prev_left_wrist_x = left_wrist_x
                            prev_left_wrist_y = left_wrist_y
                            prev_shoulder_angle_change = shoulder_angle_change
                            prev_shoulder_line_tilt = shoulder_line_tilt
                            prev_shoulder_twist_hip = shoulder_twist_hip
                            prev_hip_angle_change = hip_angle_change
                            prev_timestamp = timestamp
                    
                    if speed_data:
                        # Create the Speed time series chart
                        import plotly.graph_objects as go
                        
                        fig_speed = go.Figure()
                        
                        # Extract data arrays for plotting
                        left_wrist_speeds = [d['left_wrist_speed'] for d in speed_data]
                        body_angles_speeds = [d['body_angles_speed'] for d in speed_data]
                        
                        # Add speed measurements as traces
                        fig_speed.add_trace(go.Scatter(
                            x=speed_timestamps, y=left_wrist_speeds,
                            mode='lines+markers', name='1. Left Wrist Speed',
                            line=dict(color='#1f77b4', width=2), marker=dict(size=3)
                        ))
                        
                        fig_speed.add_trace(go.Scatter(
                            x=speed_timestamps, y=body_angles_speeds,
                            mode='lines+markers', name='2. Body Angles Speed',
                            line=dict(color='#ff7f0e', width=2), marker=dict(size=3)
                        ))
                        
                        # Add cricket event points as scatter overlays
                        if trigger_time > 0:
                            trigger_idx = min(range(len(speed_timestamps)), 
                                             key=lambda i: abs(speed_timestamps[i] - trigger_time))
                            if trigger_idx < len(speed_data):
                                trigger_data = speed_data[trigger_idx]
                                fig_speed.add_trace(go.Scatter(
                                    x=[speed_timestamps[trigger_idx]] * 2,
                                    y=[trigger_data['left_wrist_speed'], trigger_data['body_angles_speed']],
                                    mode='markers',
                                    name='Trigger Point',
                                    marker=dict(color='red', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        if swing_time > 0:
                            swing_idx = min(range(len(speed_timestamps)), 
                                          key=lambda i: abs(speed_timestamps[i] - swing_time))
                            if swing_idx < len(speed_data):
                                swing_data = speed_data[swing_idx]
                                fig_speed.add_trace(go.Scatter(
                                    x=[speed_timestamps[swing_idx]] * 2,
                                    y=[swing_data['left_wrist_speed'], swing_data['body_angles_speed']],
                                    mode='markers',
                                    name='Swing Start',
                                    marker=dict(color='blue', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        if contact_time > 0:
                            contact_idx = min(range(len(speed_timestamps)), 
                                            key=lambda i: abs(speed_timestamps[i] - contact_time))
                            if contact_idx < len(speed_data):
                                contact_data = speed_data[contact_idx]
                                fig_speed.add_trace(go.Scatter(
                                    x=[speed_timestamps[contact_idx]] * 2,
                                    y=[contact_data['left_wrist_speed'], contact_data['body_angles_speed']],
                                    mode='markers',
                                    name='Bat-Ball Connect',
                                    marker=dict(color='green', size=8, symbol='diamond'),
                                    showlegend=True
                                ))
                        
                        fig_speed.update_layout(
                            title="Speed - Time Series Analysis of Movement Speeds",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Speed (units per second)",
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02
                            )
                        )
                        
                        st.plotly_chart(fig_speed, use_container_width=True)
                        
                        # Show summary information
                        if speed_data:
                            max_wrist_speed = max(left_wrist_speeds)
                            max_body_speed = max(body_angles_speeds)
                            avg_wrist_speed = sum(left_wrist_speeds) / len(left_wrist_speeds)
                            avg_body_speed = sum(body_angles_speeds) / len(body_angles_speeds)
                            
                            st.info(f"""
                            **Speed Chart Information:**
                            - Total frames analyzed: {len(speed_data)}
                            - Time range: {min(speed_timestamps):.2f}s to {max(speed_timestamps):.2f}s
                            - Max left wrist speed: {max_wrist_speed:.4f} units/second
                            - Average left wrist speed: {avg_wrist_speed:.4f} units/second
                            - Max body angles speed: {max_body_speed:.2f} degrees/second
                            - Average body angles speed: {avg_body_speed:.2f} degrees/second
                            - Body angles speed includes: Shoulder angle, Shoulder tilt, Shoulder twist-hip, Hip angle
                            """)
                    
                    else:
                        st.warning("No pose data available for Speed analysis")
                else:
                    st.warning("No video analysis results available for Speed chart")
                
                # Debug section 1: Show frames from 0 to 1 seconds (n-3 frame comparison)
                st.subheader("Debug: Shot Trigger Analysis (0s - 1s)")
                st.markdown("**Detailed frame-by-frame analysis for debugging shot trigger detection**")
                
                debug_start_time_1 = 0.0
                debug_end_time_1 = 1.0
                video_duration = st.session_state.video_processor.get_duration()
                
                if debug_start_time_1 < video_duration:
                    # Get frames in the debug range
                    fps = st.session_state.video_processor.get_fps()
                    frame_interval = 1.0 / fps  # Show every frame
                    debug_timestamps_1 = []
                    
                    current_time = debug_start_time_1
                    while current_time <= min(debug_end_time_1, video_duration):
                        debug_timestamps_1.append(current_time)
                        current_time += frame_interval
                    
                    # Show debug info
                    st.info(f"Video duration: {video_duration:.2f}s | FPS: {fps:.1f} | Analyzing {len(debug_timestamps_1)} frames from 0s to {min(debug_end_time_1, video_duration):.2f}s")
                    
                    # Create stance detector for debug analysis
                    debug_detector_1 = StanceDetector(
                        stability_threshold=stability_threshold,
                        min_stability_duration=min_stability_duration,
                        confidence_threshold=confidence_threshold,
                        camera_perspective=camera_perspective,
                        batsman_height=batsman_height
                    )
                    
                    # Process debug frames
                    with st.spinner("Processing debug frames (0s-1s)..."):
                        cols_per_row = 2
                        for i in range(0, len(debug_timestamps_1), cols_per_row):
                            cols = st.columns(cols_per_row)
                            
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(debug_timestamps_1):
                                    timestamp = debug_timestamps_1[idx]
                                    
                                    with cols[j]:
                                        try:
                                            # Get frame at timestamp
                                            frame = st.session_state.video_processor.get_frame_at_time(timestamp)
                                            if frame is not None:
                                                # Crop to analysis area
                                                x1, y1, x2, y2 = st.session_state.rectangle_coords
                                                cropped_frame = frame[y1:y2, x1:x2]
                                                
                                                # Process with pose detection
                                                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                                                pose_results = debug_detector_1.pose.process(rgb_frame)
                                                
                                                # Draw pose landmarks if detected
                                                annotated_frame = rgb_frame.copy()
                                                
                                                if pose_results.pose_landmarks:
                                                    debug_detector_1.mp_drawing.draw_landmarks(
                                                        annotated_frame, 
                                                        pose_results.pose_landmarks, 
                                                        debug_detector_1.mp_pose.POSE_CONNECTIONS
                                                    )
                                                
                                                # Display frame
                                                st.image(
                                                    annotated_frame, 
                                                    caption=f"Debug Frame at {timestamp:.3f}s", 
                                                    use_container_width=True
                                                )
                                                
                                                # Analyze stance and movement for this frame
                                                if pose_results.pose_landmarks:
                                                    # Get detailed stance analysis
                                                    is_stable_stance, pose_data = debug_detector_1.detect_stance(cropped_frame, timestamp)
                                                    
                                                    # Calculate movement compared to frame 3 positions earlier (n+3 comparison)
                                                    frame_number = int(timestamp * fps)
                                                    skip_frames = 3
                                                    earlier_frame_number = frame_number - skip_frames
                                                    
                                                    movement_data = None
                                                    if earlier_frame_number >= 0:
                                                        # Get earlier frame for comparison
                                                        earlier_timestamp = earlier_frame_number / fps
                                                        earlier_frame = st.session_state.video_processor.get_frame_at_time(earlier_timestamp)
                                                        
                                                        if earlier_frame is not None:
                                                            # Crop earlier frame
                                                            earlier_cropped = earlier_frame[y1:y2, x1:x2]
                                                            earlier_rgb = cv2.cvtColor(earlier_cropped, cv2.COLOR_BGR2RGB)
                                                            earlier_pose_results = debug_detector_1.pose.process(earlier_rgb)
                                                            
                                                            if hasattr(earlier_pose_results, 'pose_landmarks') and earlier_pose_results.pose_landmarks:
                                                                # Get biomechanical data for both frames
                                                                current_features = debug_detector_1._calculate_stance_features(pose_results.pose_landmarks.landmark)
                                                                earlier_features = debug_detector_1._calculate_stance_features(earlier_pose_results.pose_landmarks.landmark)
                                                                
                                                                # Calculate parameter changes (matching shot trigger algorithm)
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
                                                                
                                                                triggered_params = []
                                                                
                                                                # Check each parameter for movement
                                                                param_changes = {
                                                                    'shoulder_line_angle': abs(current_features.get('shoulder_line_angle', 0) - earlier_features.get('shoulder_line_angle', 0)),
                                                                    'hip_line_angle': abs(current_features.get('hip_line_angle', 0) - earlier_features.get('hip_line_angle', 0)),
                                                                    'shoulder_line_twist': abs(current_features.get('shoulder_line_twist', 0) - earlier_features.get('shoulder_line_twist', 0)),
                                                                    'hip_line_twist': abs(current_features.get('hip_line_twist', 0) - earlier_features.get('hip_line_twist', 0)),
                                                                }
                                                                
                                                                # Knee angles (max of left/right)
                                                                current_knee_max = max(current_features.get('left_knee_angle', 0), current_features.get('right_knee_angle', 0))
                                                                earlier_knee_max = max(earlier_features.get('left_knee_angle', 0), earlier_features.get('right_knee_angle', 0))
                                                                param_changes['knee_angle'] = abs(current_knee_max - earlier_knee_max)
                                                                
                                                                # Knee-to-ankle angles (max of left/right)
                                                                current_knee_ankle_max = max(current_features.get('left_knee_to_ankle_angle', 0), current_features.get('right_knee_to_ankle_angle', 0))
                                                                earlier_knee_ankle_max = max(earlier_features.get('left_knee_to_ankle_angle', 0), earlier_features.get('right_knee_to_ankle_angle', 0))
                                                                param_changes['knee_to_ankle_angle'] = abs(current_knee_ankle_max - earlier_knee_ankle_max)
                                                                
                                                                # Elbow-wrist angles (max of left/right)
                                                                current_elbow_wrist_max = max(current_features.get('left_elbow_wrist_angle', 0), current_features.get('right_elbow_wrist_angle', 0))
                                                                earlier_elbow_wrist_max = max(earlier_features.get('left_elbow_wrist_angle', 0), earlier_features.get('right_elbow_wrist_angle', 0))
                                                                param_changes['elbow_wrist_line_angle'] = abs(current_elbow_wrist_max - earlier_elbow_wrist_max)
                                                                
                                                                # Shoulder-elbow angles (max of left/right)
                                                                current_shoulder_elbow_max = max(current_features.get('left_shoulder_elbow_angle', 0), current_features.get('right_shoulder_elbow_angle', 0))
                                                                earlier_shoulder_elbow_max = max(earlier_features.get('left_shoulder_elbow_angle', 0), earlier_features.get('right_shoulder_elbow_angle', 0))
                                                                param_changes['shoulder_elbow_line_angle'] = abs(current_shoulder_elbow_max - earlier_shoulder_elbow_max)
                                                                
                                                                # Ankle coordinates (max of all coordinate changes)
                                                                ankle_changes = [
                                                                    abs(current_features.get('left_ankle_x', 0) - earlier_features.get('left_ankle_x', 0)),
                                                                    abs(current_features.get('left_ankle_y', 0) - earlier_features.get('left_ankle_y', 0)),
                                                                    abs(current_features.get('right_ankle_x', 0) - earlier_features.get('right_ankle_x', 0)),
                                                                    abs(current_features.get('right_ankle_y', 0) - earlier_features.get('right_ankle_y', 0))
                                                                ]
                                                                param_changes['ankle_coordinates'] = max(ankle_changes)
                                                                
                                                                # Check which parameters exceed thresholds
                                                                for param, change in param_changes.items():
                                                                    if change > movement_threshold[param]:
                                                                        triggered_params.append({
                                                                            'param': param,
                                                                            'change': change,
                                                                            'threshold': movement_threshold[param]
                                                                        })
                                                                
                                                                # Display movement analysis
                                                                st.markdown(f"**Frame {frame_number} vs Frame {earlier_frame_number} (n-3 comparison)**")
                                                                st.markdown(f"<p style='font-size:10px; margin:0;'><b>Comparison Time:</b> {timestamp:.3f}s vs {earlier_timestamp:.3f}s</p>", 
                                                                           unsafe_allow_html=True)
                                                                
                                                                triggered_count = len(triggered_params)
                                                                if triggered_count >= 3:
                                                                    st.error(f"🚨 SHOT TRIGGER: {triggered_count}/9 parameters exceed thresholds")
                                                                elif triggered_count > 0:
                                                                    st.warning(f"ℹ️ {triggered_count}/9 parameters exceed thresholds")
                                                                else:
                                                                    st.success("✅ All parameters within normal range")
                                                                
                                                                # Display movement parameters table with directional information
                                                                movement_table = []
                                                                time_span = skip_frames / fps
                                                                
                                                                for param, change in param_changes.items():
                                                                    threshold = movement_threshold[param]
                                                                    exceeds_threshold = change > threshold
                                                                    
                                                                    # Calculate directional change
                                                                    if param == 'ankle_coordinates':
                                                                        left_x_dir = current_features.get('left_ankle_x', 0) - earlier_features.get('left_ankle_x', 0)
                                                                        left_y_dir = current_features.get('left_ankle_y', 0) - earlier_features.get('left_ankle_y', 0)
                                                                        right_x_dir = current_features.get('right_ankle_x', 0) - earlier_features.get('right_ankle_x', 0)
                                                                        right_y_dir = current_features.get('right_ankle_y', 0) - earlier_features.get('right_ankle_y', 0)
                                                                        
                                                                        changes = [left_x_dir, left_y_dir, right_x_dir, right_y_dir]
                                                                        max_change_idx = max(range(len(changes)), key=lambda i: abs(changes[i]))
                                                                        directional_change = changes[max_change_idx]
                                                                    elif param == 'knee_angle':
                                                                        current_max = max(current_features.get('left_knee_angle', 0), current_features.get('right_knee_angle', 0))
                                                                        earlier_max = max(earlier_features.get('left_knee_angle', 0), earlier_features.get('right_knee_angle', 0))
                                                                        directional_change = current_max - earlier_max
                                                                    elif param == 'knee_to_ankle_angle':
                                                                        current_max = max(current_features.get('left_knee_to_ankle_angle', 0), current_features.get('right_knee_to_ankle_angle', 0))
                                                                        earlier_max = max(earlier_features.get('left_knee_to_ankle_angle', 0), earlier_features.get('right_knee_to_ankle_angle', 0))
                                                                        directional_change = current_max - earlier_max
                                                                    elif param == 'elbow_wrist_line_angle':
                                                                        current_max = max(current_features.get('left_elbow_wrist_angle', 0), current_features.get('right_elbow_wrist_angle', 0))
                                                                        earlier_max = max(earlier_features.get('left_elbow_wrist_angle', 0), earlier_features.get('right_elbow_wrist_angle', 0))
                                                                        directional_change = current_max - earlier_max
                                                                    elif param == 'shoulder_elbow_line_angle':
                                                                        current_max = max(current_features.get('left_shoulder_elbow_angle', 0), current_features.get('right_shoulder_elbow_angle', 0))
                                                                        earlier_max = max(earlier_features.get('left_shoulder_elbow_angle', 0), earlier_features.get('right_shoulder_elbow_angle', 0))
                                                                        directional_change = current_max - earlier_max
                                                                    else:
                                                                        directional_change = current_features.get(param, 0) - earlier_features.get(param, 0)
                                                                    
                                                                    # Calculate velocity and direction
                                                                    velocity = change / time_span
                                                                    direction_arrow = "↗" if directional_change > 0 else "↘"
                                                                    
                                                                    # Format display
                                                                    display_name = param.replace('_', ' ').title()
                                                                    if param == 'ankle_coordinates':
                                                                        change_str = f"{change:.3f}"
                                                                        velocity_str = f"{velocity:.3f}/s"
                                                                        threshold_str = f"{threshold:.3f}"
                                                                        direction_str = f"{directional_change:+.3f}"
                                                                    else:
                                                                        change_str = f"{change:.1f}°"
                                                                        velocity_str = f"{velocity:.1f}°/s"
                                                                        threshold_str = f"{threshold}°"
                                                                        direction_str = f"{directional_change:+.1f}°"
                                                                    
                                                                    status = "🔴" if exceeds_threshold else "🟢"
                                                                    
                                                                    movement_table.append({
                                                                        'Parameter': display_name,
                                                                        'Change': change_str,
                                                                        'Direction': f"{direction_str} {direction_arrow}",
                                                                        'Velocity': velocity_str,
                                                                        'Threshold': threshold_str,
                                                                        'Status': status
                                                                    })
                                                                
                                                                # Display table
                                                                st.table(movement_table)
                                                    
                                                    else:
                                                        st.info("No earlier frame available for comparison")
                                                
                                                else:
                                                    st.warning("No pose detected in this frame")
                                            
                                            else:
                                                st.error("Could not load frame")
                                                
                                        except Exception as e:
                                            st.error(f"Error processing debug frame: {str(e)}")
                else:
                    st.info(f"Debug time range (0s - 1s) is outside video duration ({video_duration:.1f}s)")
                
                # Debug section 2: Show frames from 14.8 to 15.3 seconds (n-3 frame comparison)
                st.subheader("Debug: Shot Trigger Analysis (14.8s - 15.3s)")
                st.markdown("**Detailed frame-by-frame analysis for debugging shot trigger detection**")
                
                debug_start_time = 14.8
                debug_end_time = 15.3
                video_duration = st.session_state.video_processor.get_duration()
                
                if debug_start_time < video_duration and debug_end_time <= video_duration:
                    # Get frames in the debug range
                    fps = st.session_state.video_processor.get_fps()
                    frame_interval = 1.0 / fps  # Show every frame
                    debug_timestamps = []
                    
                    current_time = debug_start_time
                    while current_time <= debug_end_time:
                        debug_timestamps.append(current_time)
                        current_time += frame_interval
                    
                    # Create stance detector for debug analysis
                    debug_detector = StanceDetector(
                        stability_threshold=stability_threshold,
                        min_stability_duration=min_stability_duration,
                        confidence_threshold=confidence_threshold,
                        camera_perspective=camera_perspective,
                        batsman_height=batsman_height
                    )
                    
                    # Process debug frames
                    with st.spinner("Processing debug frames..."):
                        cols_per_row = 2
                        for i in range(0, len(debug_timestamps), cols_per_row):
                            cols = st.columns(cols_per_row)
                            
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(debug_timestamps):
                                    timestamp = debug_timestamps[idx]
                                    
                                    with cols[j]:
                                        try:
                                            # Get frame at timestamp
                                            frame = st.session_state.video_processor.get_frame_at_time(timestamp)
                                            if frame is not None:
                                                # Crop to analysis area
                                                x1, y1, x2, y2 = st.session_state.rectangle_coords
                                                cropped_frame = frame[y1:y2, x1:x2]
                                                
                                                # Process with pose detection
                                                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                                                pose_results = debug_detector.pose.process(rgb_frame)
                                                
                                                # Draw pose landmarks if detected
                                                annotated_frame = rgb_frame.copy()
                                                
                                                if hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
                                                    try:
                                                        # Draw pose landmarks
                                                        debug_detector.mp_drawing.draw_landmarks(
                                                            annotated_frame,
                                                            pose_results.pose_landmarks,
                                                            debug_detector.mp_pose.POSE_CONNECTIONS,
                                                            landmark_drawing_spec=debug_detector.mp_drawing.DrawingSpec(
                                                                color=(255, 0, 0), thickness=1, circle_radius=2
                                                            ),
                                                            connection_drawing_spec=debug_detector.mp_drawing.DrawingSpec(
                                                                color=(0, 255, 0), thickness=1
                                                            )
                                                        )
                                                    except Exception as e:
                                                        st.warning(f"Error drawing pose: {str(e)}")
                                                
                                                # Display frame
                                                st.image(
                                                    annotated_frame, 
                                                    caption=f"Debug Frame at {timestamp:.3f}s", 
                                                    use_container_width=True
                                                )
                                                
                                                # Analyze stance and movement for this frame
                                                if pose_results.pose_landmarks:
                                                    # Get detailed stance analysis
                                                    is_stable_stance, pose_data = debug_detector.detect_stance(cropped_frame, timestamp)
                                                    
                                                    # Calculate movement compared to frame 3 positions earlier (n+3 comparison)
                                                    frame_number = int(timestamp * fps)
                                                    skip_frames = 3
                                                    earlier_frame_number = frame_number - skip_frames
                                                    
                                                    movement_data = None
                                                    earlier_timestamp = 0
                                                    if earlier_frame_number >= 0:
                                                        # Get earlier frame for comparison
                                                        earlier_timestamp = earlier_frame_number / fps
                                                        earlier_frame = st.session_state.video_processor.get_frame_at_time(earlier_timestamp)
                                                        
                                                        if earlier_frame is not None:
                                                            # Crop earlier frame
                                                            earlier_cropped = earlier_frame[y1:y2, x1:x2]
                                                            earlier_rgb = cv2.cvtColor(earlier_cropped, cv2.COLOR_BGR2RGB)
                                                            earlier_pose_results = debug_detector.pose.process(earlier_rgb)
                                                            
                                                            if hasattr(earlier_pose_results, 'pose_landmarks') and earlier_pose_results.pose_landmarks:
                                                                # Get biomechanical data for both frames
                                                                _, earlier_pose_data = debug_detector.detect_stance(earlier_cropped, earlier_timestamp)
                                                                
                                                                # Calculate movement parameters
                                                                movement_data = {
                                                                    'shoulder_line_angle': abs(pose_data.get('shoulder_line_angle', 0) - earlier_pose_data.get('shoulder_line_angle', 0)),
                                                                    'hip_line_angle': abs(pose_data.get('hip_line_angle', 0) - earlier_pose_data.get('hip_line_angle', 0)),
                                                                    'shoulder_line_twist': abs(pose_data.get('shoulder_line_twist', 0) - earlier_pose_data.get('shoulder_line_twist', 0)),
                                                                    'hip_line_twist': abs(pose_data.get('hip_line_twist', 0) - earlier_pose_data.get('hip_line_twist', 0)),
                                                                    'knee_to_ankle_angle': max(
                                                                        abs(pose_data.get('left_knee_to_ankle_angle', 0) - earlier_pose_data.get('left_knee_to_ankle_angle', 0)),
                                                                        abs(pose_data.get('right_knee_to_ankle_angle', 0) - earlier_pose_data.get('right_knee_to_ankle_angle', 0))
                                                                    ),
                                                                    'knee_angle': max(
                                                                        abs(pose_data.get('left_knee_angle', 0) - earlier_pose_data.get('left_knee_angle', 0)),
                                                                        abs(pose_data.get('right_knee_angle', 0) - earlier_pose_data.get('right_knee_angle', 0))
                                                                    ),
                                                                    'elbow_wrist_line_angle': max(
                                                                        abs(pose_data.get('left_elbow_wrist_angle', 0) - earlier_pose_data.get('left_elbow_wrist_angle', 0)),
                                                                        abs(pose_data.get('right_elbow_wrist_angle', 0) - earlier_pose_data.get('right_elbow_wrist_angle', 0))
                                                                    ),
                                                                    'ankle_coordinates': max(
                                                                        abs(pose_data.get('left_ankle_x', 0) - earlier_pose_data.get('left_ankle_x', 0)),
                                                                        abs(pose_data.get('left_ankle_y', 0) - earlier_pose_data.get('left_ankle_y', 0)),
                                                                        abs(pose_data.get('right_ankle_x', 0) - earlier_pose_data.get('right_ankle_x', 0)),
                                                                        abs(pose_data.get('right_ankle_y', 0) - earlier_pose_data.get('right_ankle_y', 0))
                                                                    )
                                                                }
                                                                
                                                                # Add shoulder-elbow line parameter if not already present
                                                                if 'shoulder_elbow_line_angle' not in movement_data:
                                                                    movement_data['shoulder_elbow_line_angle'] = max(
                                                                        abs(pose_data.get('left_shoulder_elbow_angle', 0) - earlier_pose_data.get('left_shoulder_elbow_angle', 0)),
                                                                        abs(pose_data.get('right_shoulder_elbow_angle', 0) - earlier_pose_data.get('right_shoulder_elbow_angle', 0))
                                                                    )
                                                    
                                                    # Show movement parameters
                                                    if movement_data:
                                                        st.markdown(f"**Movement vs {earlier_timestamp:.3f}s:**")
                                                        
                                                        # Movement thresholds
                                                        thresholds = {
                                                            'shoulder_line_angle': 10,
                                                            'hip_line_angle': 2.5, 
                                                            'shoulder_line_twist': 20,
                                                            'hip_line_twist': 2,
                                                            'knee_to_ankle_angle': 5,
                                                            'knee_angle': 10,
                                                            'elbow_wrist_line_angle': 10,
                                                            'shoulder_elbow_line_angle': 5,
                                                            'ankle_coordinates': 0.025
                                                        }
                                                        
                                                        # Display movement in compact table format
                                                        movement_table = []
                                                        time_span = skip_frames / fps
                                                        triggered_count = 0
                                                        
                                                        for param, change in movement_data.items():
                                                            threshold = thresholds[param]
                                                            exceeds_threshold = change > threshold
                                                            if exceeds_threshold:
                                                                triggered_count += 1
                                                            
                                                            # Calculate directional change for this parameter
                                                            if param == 'ankle_coordinates':
                                                                left_x_dir = pose_data.get('left_ankle_x', 0) - earlier_pose_data.get('left_ankle_x', 0)
                                                                left_y_dir = pose_data.get('left_ankle_y', 0) - earlier_pose_data.get('left_ankle_y', 0)
                                                                right_x_dir = pose_data.get('right_ankle_x', 0) - earlier_pose_data.get('right_ankle_x', 0)
                                                                right_y_dir = pose_data.get('right_ankle_y', 0) - earlier_pose_data.get('right_ankle_y', 0)
                                                                
                                                                changes = [left_x_dir, left_y_dir, right_x_dir, right_y_dir]
                                                                max_change_idx = max(range(len(changes)), key=lambda i: abs(changes[i]))
                                                                directional_change = changes[max_change_idx]
                                                            elif param == 'knee_angle':
                                                                current_max = max(pose_data.get('left_knee_angle', 0), pose_data.get('right_knee_angle', 0))
                                                                earlier_max = max(earlier_pose_data.get('left_knee_angle', 0), earlier_pose_data.get('right_knee_angle', 0))
                                                                directional_change = current_max - earlier_max
                                                            elif param == 'knee_to_ankle_angle':
                                                                current_max = max(pose_data.get('left_knee_to_ankle_angle', 0), pose_data.get('right_knee_to_ankle_angle', 0))
                                                                earlier_max = max(earlier_pose_data.get('left_knee_to_ankle_angle', 0), earlier_pose_data.get('right_knee_to_ankle_angle', 0))
                                                                directional_change = current_max - earlier_max
                                                            elif param == 'elbow_wrist_line_angle':
                                                                current_max = max(pose_data.get('left_elbow_wrist_angle', 0), pose_data.get('right_elbow_wrist_angle', 0))
                                                                earlier_max = max(earlier_pose_data.get('left_elbow_wrist_angle', 0), earlier_pose_data.get('right_elbow_wrist_angle', 0))
                                                                directional_change = current_max - earlier_max
                                                            elif param == 'shoulder_elbow_line_angle':
                                                                current_max = max(pose_data.get('left_shoulder_elbow_angle', 0), pose_data.get('right_shoulder_elbow_angle', 0))
                                                                earlier_max = max(earlier_pose_data.get('left_shoulder_elbow_angle', 0), earlier_pose_data.get('right_shoulder_elbow_angle', 0))
                                                                directional_change = current_max - earlier_max
                                                            else:
                                                                directional_change = pose_data.get(param, 0) - earlier_pose_data.get(param, 0)
                                                            
                                                            # Calculate velocity and direction arrow
                                                            velocity = change / time_span
                                                            direction_arrow = "↗" if directional_change > 0 else "↘"
                                                            
                                                            # Format display
                                                            display_name = param.replace('_', ' ').title()
                                                            if param == 'ankle_coordinates':
                                                                change_str = f"{change:.3f}"
                                                                velocity_str = f"{velocity:.3f}/s"
                                                                threshold_str = f"{threshold:.3f}"
                                                                direction_str = f"{directional_change:+.3f}"
                                                            else:
                                                                change_str = f"{change:.1f}°"
                                                                velocity_str = f"{velocity:.1f}°/s"
                                                                threshold_str = f"{threshold}°"
                                                                direction_str = f"{directional_change:+.1f}°"
                                                            
                                                            status = "🔴" if exceeds_threshold else "🟢"
                                                            
                                                            movement_table.append({
                                                                'Parameter': display_name,
                                                                'Change': change_str,
                                                                'Direction': f"{direction_str} {direction_arrow}",
                                                                'Velocity': velocity_str,
                                                                'Threshold': threshold_str,
                                                                'Status': status
                                                            })
                                                        
                                                        # Display table
                                                        st.table(movement_table)
                                                        
                                                        # Summary
                                                        if triggered_count >= 3:
                                                            st.error(f"⚠️ POTENTIAL TRIGGER: {triggered_count}/9 parameters exceed thresholds")
                                                        elif triggered_count > 0:
                                                            st.warning(f"ℹ️ {triggered_count}/9 parameters exceed thresholds")
                                                        else:
                                                            st.success("✅ All parameters within normal range")
                                                    
                                                    else:
                                                        st.info("No earlier frame available for comparison")
                                                
                                                else:
                                                    st.warning("No pose detected in this frame")
                                            
                                            else:
                                                st.error("Could not load frame")
                                                
                                        except Exception as e:
                                            st.error(f"Error processing debug frame: {str(e)}")
                else:
                    st.info(f"Debug time range (14.8s - 15.3s) is outside video duration ({video_duration:.1f}s)")

                # Debug Section 3: Batting Stance Detection (42.0s - 42.5s)
                st.subheader("🔍 Debug Section: Batting Stance Detection (42.0s - 42.5s)")
                st.markdown("**Detailed analysis of batting stance criteria with n-7 frame comparison**")
                
                video_duration = st.session_state.video_processor.get_duration()
                fps = st.session_state.video_processor.get_fps()
                
                if 42.0 <= video_duration and video_duration >= 42.5:
                    # Initialize stance detector for debug analysis
                    debug_detector = StanceDetector(
                        stability_threshold=stability_threshold,
                        min_stability_duration=min_stability_duration,
                        confidence_threshold=confidence_threshold,
                        camera_perspective=camera_perspective,
                        batsman_height=batsman_height
                    )
                    
                    debug_start_time = 42.0
                    debug_end_time = 42.5
                    debug_step = 0.1  # Check every 0.1 seconds
                    
                    # Generate time points every 0.033s (30fps) from 42.0 to 42.5 seconds
                    debug_times = []
                    current_time = 42.0
                    while current_time <= 42.5:
                        debug_times.append(current_time)
                        current_time += (1.0 / fps)  # Add one frame duration
                    
                    for debug_time in debug_times:
                        if debug_time <= video_duration:
                            try:
                                st.markdown(f"**Frame at {debug_time:.3f}s:**")
                                
                                # Get current frame
                                current_frame = st.session_state.video_processor.get_frame_at_time(debug_time)
                                if current_frame is not None:
                                    current_pose_data = debug_detector.detect_stance(current_frame, debug_time)[1]
                                    
                                    # Get n-7 frame (7 frames earlier)
                                    skip_frames = 7
                                    earlier_time = debug_time - (skip_frames / fps)
                                    
                                    if earlier_time >= 0:
                                        earlier_frame = st.session_state.video_processor.get_frame_at_time(earlier_time)
                                        if earlier_frame is not None:
                                            earlier_pose_data = debug_detector.detect_stance(earlier_frame, earlier_time)[1]
                                            
                                            if current_pose_data and earlier_pose_data:
                                                # Check all 6 batting stance criteria
                                                criteria_results = {}
                                                
                                                # 1. Ankle Stability (both left and right at same coordinates)
                                                ankle_threshold = 0.01  # 1% movement
                                                left_ankle_x_change = abs(current_pose_data.get('left_ankle_x', 0) - earlier_pose_data.get('left_ankle_x', 0))
                                                left_ankle_y_change = abs(current_pose_data.get('left_ankle_y', 0) - earlier_pose_data.get('left_ankle_y', 0))
                                                right_ankle_x_change = abs(current_pose_data.get('right_ankle_x', 0) - earlier_pose_data.get('right_ankle_x', 0))
                                                right_ankle_y_change = abs(current_pose_data.get('right_ankle_y', 0) - earlier_pose_data.get('right_ankle_y', 0))
                                                
                                                left_ankle_stable = left_ankle_x_change <= ankle_threshold and left_ankle_y_change <= ankle_threshold
                                                right_ankle_stable = right_ankle_x_change <= ankle_threshold and right_ankle_y_change <= ankle_threshold
                                                ankle_stability = left_ankle_stable and right_ankle_stable
                                                
                                                # 2. Hip Line Angle (< 2 degree change)
                                                hip_angle_change = abs(current_pose_data.get('hip_line_angle', 0) - earlier_pose_data.get('hip_line_angle', 0))
                                                hip_angle_stable = hip_angle_change < 2.0
                                                
                                                # 3. Shoulder Line Twist (< 10 degrees change)
                                                shoulder_twist_change = abs(current_pose_data.get('shoulder_line_twist', 0) - earlier_pose_data.get('shoulder_line_twist', 0))
                                                shoulder_twist_stable = shoulder_twist_change < 10.0
                                                
                                                # 4. Shoulder-Elbow Line Angles (both < 2 degrees change)
                                                left_shoulder_elbow_change = abs(current_pose_data.get('left_shoulder_elbow_angle', 0) - earlier_pose_data.get('left_shoulder_elbow_angle', 0))
                                                right_shoulder_elbow_change = abs(current_pose_data.get('right_shoulder_elbow_angle', 0) - earlier_pose_data.get('right_shoulder_elbow_angle', 0))
                                                shoulder_elbow_stable = left_shoulder_elbow_change < 2.0 and right_shoulder_elbow_change < 2.0
                                                
                                                # 5. Camera Perspective (back not towards camera)
                                                shoulder_twist = current_pose_data.get('shoulder_line_twist', 0)
                                                camera_perspective_ok = abs(shoulder_twist) < 45.0
                                                

                                                
                                                # Create detailed criteria table
                                                criteria_table = [
                                                    {
                                                        'Criterion': 'Ankle Stability',
                                                        'Status': '✅ PASS' if ankle_stability else '❌ FAIL',
                                                        'Details': f"L: {left_ankle_x_change:.3f}, {left_ankle_y_change:.3f} | R: {right_ankle_x_change:.3f}, {right_ankle_y_change:.3f}",
                                                        'Threshold': f"< {ankle_threshold:.3f}",
                                                        'Current': f"L_ankle: ({current_pose_data.get('left_ankle_x', 0):.3f}, {current_pose_data.get('left_ankle_y', 0):.3f})",
                                                        'Earlier': f"L_ankle: ({earlier_pose_data.get('left_ankle_x', 0):.3f}, {earlier_pose_data.get('left_ankle_y', 0):.3f})"
                                                    },
                                                    {
                                                        'Criterion': 'Hip Line Angle',
                                                        'Status': '✅ PASS' if hip_angle_stable else '❌ FAIL',
                                                        'Details': f"Change: {hip_angle_change:.1f}°",
                                                        'Threshold': "< 2.0°",
                                                        'Current': f"{current_pose_data.get('hip_line_angle', 0):.1f}°",
                                                        'Earlier': f"{earlier_pose_data.get('hip_line_angle', 0):.1f}°"
                                                    },
                                                    {
                                                        'Criterion': 'Shoulder Twist',
                                                        'Status': '✅ PASS' if shoulder_twist_stable else '❌ FAIL',
                                                        'Details': f"Change: {shoulder_twist_change:.1f}°",
                                                        'Threshold': "< 10.0°",
                                                        'Current': f"{current_pose_data.get('shoulder_line_twist', 0):.1f}°",
                                                        'Earlier': f"{earlier_pose_data.get('shoulder_line_twist', 0):.1f}°"
                                                    },
                                                    {
                                                        'Criterion': 'Shoulder-Elbow Angles',
                                                        'Status': '✅ PASS' if shoulder_elbow_stable else '❌ FAIL',
                                                        'Details': f"L: {left_shoulder_elbow_change:.1f}° | R: {right_shoulder_elbow_change:.1f}°",
                                                        'Threshold': "< 2.0°",
                                                        'Current': f"L: {current_pose_data.get('left_shoulder_elbow_angle', 0):.1f}° | R: {current_pose_data.get('right_shoulder_elbow_angle', 0):.1f}°",
                                                        'Earlier': f"L: {earlier_pose_data.get('left_shoulder_elbow_angle', 0):.1f}° | R: {earlier_pose_data.get('right_shoulder_elbow_angle', 0):.1f}°"
                                                    },
                                                    {
                                                        'Criterion': 'Camera Perspective',
                                                        'Status': '✅ PASS' if camera_perspective_ok else '❌ FAIL',
                                                        'Details': f"Shoulder twist: {shoulder_twist:.1f}°",
                                                        'Threshold': "< 45.0°",
                                                        'Current': f"Twist: {shoulder_twist:.1f}°",
                                                        'Earlier': f"Twist: {earlier_pose_data.get('shoulder_line_twist', 0):.1f}°"
                                                    }
                                                ]
                                                
                                                st.table(criteria_table)
                                                
                                                # Overall result
                                                all_criteria_passed = all([
                                                    ankle_stability, hip_angle_stable, shoulder_twist_stable,
                                                    shoulder_elbow_stable, camera_perspective_ok
                                                ])
                                                
                                                passed_count = sum([
                                                    ankle_stability, hip_angle_stable, shoulder_twist_stable,
                                                    shoulder_elbow_stable, camera_perspective_ok
                                                ])
                                                
                                                if all_criteria_passed:
                                                    st.success(f"🎯 **BATTING STANCE DETECTED** - All 5/5 criteria passed")
                                                else:
                                                    st.warning(f"⚠️ **Partial match** - {passed_count}/5 criteria passed")
                                                
                                                # Comparison frame info
                                                st.caption(f"Comparing frame at {debug_time:.1f}s with frame at {earlier_time:.3f}s (n-5 comparison)")
                                                
                                            else:
                                                st.warning("Pose not detected in one or both frames")
                                        else:
                                            st.warning(f"Could not load earlier frame at {earlier_time:.3f}s")
                                    else:
                                        st.info(f"Earlier frame time {earlier_time:.3f}s is before video start")
                                else:
                                    st.warning(f"Could not load frame at {debug_time:.1f}s")
                                
                                st.divider()
                                
                            except Exception as e:
                                st.error(f"Error processing batting stance debug frame at {debug_time:.1f}s: {str(e)}")
                else:
                    st.info(f"Debug time range (42.0s - 42.5s) is outside video duration ({video_duration:.1f}s)")
            
            # Weight Distribution Debug Section
            st.subheader("🏏 Weight Distribution Debug (0.6s - 1.2s)")
            st.markdown("Analyze weight distribution classification and center of gravity calculations")
            
            video_duration = st.session_state.video_processor.get_duration()
            debug_start_time = 0.6
            debug_end_time = 1.2
            
            if debug_end_time <= video_duration:
                # Filter results for the debug time range
                debug_results = []
                fps = st.session_state.video_processor.get_fps()
                for result in all_results:
                    frame_time = result['timestamp']
                    if debug_start_time <= frame_time <= debug_end_time:
                        debug_results.append(result)
                
                if debug_results:
                    st.info(f"Showing {len(debug_results)} frames from {debug_start_time}s to {debug_end_time}s")
                    
                    # Create weight distribution summary table
                    weight_summary = []
                    for result in debug_results:
                        biomech_data = result.get('biomech_data', {})
                        weight_summary.append({
                            'Frame': result['frame'],
                            'Time (s)': f"{result['timestamp']:.3f}",
                            'Weight Distribution': biomech_data.get('weight_distribution_text', 'Unknown'),
                            'CoG X': f"{biomech_data.get('cog_x', 0):.4f}",
                            'Stance Center X': f"{biomech_data.get('stance_center_x', 0):.4f}",
                            'CoG Offset': f"{biomech_data.get('cog_offset_from_center', 0):.4f}",
                            'Balanced Threshold': f"{biomech_data.get('balanced_threshold', 0):.4f}",
                            'Stance Width': f"{biomech_data.get('stance_width', 0):.4f}"
                        })
                    
                    st.dataframe(weight_summary, use_container_width=True)
                    
                    # Detailed frame analysis
                    st.subheader("Frame-by-Frame Weight Distribution Analysis")
                    
                    for result in debug_results[:6]:  # Show first 6 frames in detail
                        biomech_data = result.get('biomech_data', {})
                        frame_idx = result['frame']
                        frame_time = result['timestamp']
                        
                        with st.expander(f"Frame {frame_idx} at {frame_time:.3f}s - Weight: {biomech_data.get('weight_distribution_text', 'Unknown')}"):
                            # Load and display frame
                            frame = st.session_state.video_processor.get_frame_at_index(frame_idx)
                            if frame is not None:
                                # Convert to RGB and crop
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                if 'rectangle_coords' in st.session_state:
                                    x1, y1, x2, y2 = st.session_state.rectangle_coords
                                    frame_rgb = frame_rgb[y1:y2, x1:x2]
                                
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.image(frame_rgb, caption=f"Frame {frame_idx}", use_container_width=True)
                                
                                with col2:
                                    # Weight distribution details
                                    st.markdown("**Weight Distribution Analysis:**")
                                    
                                    weight_state = biomech_data.get('weight_distribution_text', 'Unknown')
                                    if weight_state == 'Left Foot':
                                        st.markdown("🔵 **Left Foot** - CoG shifted toward left foot")
                                    elif weight_state == 'Right Foot':
                                        st.markdown("🔴 **Right Foot** - CoG shifted toward right foot")
                                    elif weight_state == 'Balanced':
                                        st.markdown("🟢 **Balanced** - CoG within balanced zone")
                                    else:
                                        st.markdown(f"⚪ **{weight_state}**")
                                    
                                    # Technical details
                                    st.markdown("**Technical Details:**")
                                    st.text(f"Center of Gravity X: {biomech_data.get('cog_x', 0):.4f}")
                                    st.text(f"Stance Center X: {biomech_data.get('stance_center_x', 0):.4f}")
                                    st.text(f"CoG Offset from Center: {biomech_data.get('cog_offset_from_center', 0):.4f}")
                                    st.text(f"Balanced Threshold: {biomech_data.get('balanced_threshold', 0):.4f}")
                                    st.text(f"Stance Width: {biomech_data.get('stance_width', 0):.4f}")
                                    st.text(f"Left Foot Distance: {biomech_data.get('left_foot_distance', 0):.4f}")
                                    st.text(f"Right Foot Distance: {biomech_data.get('right_foot_distance', 0):.4f}")
                                    
                                    # Classification logic explanation
                                    cog_offset = biomech_data.get('cog_offset_from_center', 0)
                                    threshold = biomech_data.get('balanced_threshold', 0)
                                    
                                    st.markdown("**Classification Logic:**")
                                    if abs(cog_offset) <= threshold:
                                        st.text(f"|{cog_offset:.4f}| ≤ {threshold:.4f} → Balanced")
                                    elif cog_offset < 0:
                                        st.text(f"{cog_offset:.4f} < 0 → Left Foot")
                                    else:
                                        st.text(f"{cog_offset:.4f} > 0 → Right Foot")
                            
                            else:
                                st.warning(f"Could not load frame {frame_idx}")
                            
                            st.divider()
                else:
                    st.warning(f"No frames found in time range {debug_start_time}s - {debug_end_time}s")
            else:
                st.info(f"Debug time range ({debug_start_time}s - {debug_end_time}s) is outside video duration ({video_duration:.1f}s)")

            # Reset button
            if st.button("Analyze New Video"):
                # Clear session state
                for key in ['video_processor', 'rectangle_coords', 'first_frame', 'analysis_complete', 'stance_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Clean up temporary file
                try:
                    os.unlink(video_path)
                except:
                    pass
                
                st.rerun()

else:
    st.info("Please upload a cricket video file to begin stance detection.")
    
    # Instructions
    st.subheader("How to Use")
    st.markdown("""
    1. **Upload Video**: Choose a cricket video file (MP4, AVI, MOV, MKV)
    2. **Select Analysis Area**: Draw a rectangle around the batting area on the first frame
    3. **Configure Parameters**: Adjust detection sensitivity in the sidebar
    4. **Run Analysis**: Click 'Start Analysis' to detect stable batting stances
    5. **View Results**: Examine the timeline and detailed results
    
    **Stable Stance Criteria:**
    - Chest and shoulders facing the camera
    - Knees slightly bent
    - Feet almost parallel to each other
    - Head facing at least 30° to the right of camera
    - Upper body slightly lunged towards camera
    - Stationary for at least 100ms (configurable)
    - Rhythmic bat tapping is allowed
    """)
    
    st.subheader("Video Requirements")
    st.markdown("""
    - Configure camera perspective based on bowler position
    - Clear view of the batsman in the selected analysis area
    - Good lighting conditions for pose detection
    - Minimal camera shake for accurate analysis
    - Batsman should be clearly visible near the stumps
    """)

# Cleanup function to handle temporary files
def cleanup_temp_files():
    """Clean up temporary video files to prevent disk quota issues"""
    if hasattr(st.session_state, 'temp_video_path') and st.session_state.temp_video_path:
        try:
            if os.path.exists(st.session_state.temp_video_path):
                os.unlink(st.session_state.temp_video_path)
                st.session_state.temp_video_path = None
        except Exception:
            pass

# Register cleanup on session end
import atexit
atexit.register(cleanup_temp_files)
