import streamlit as st
import cv2
import numpy as np
import tempfile
import os
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

# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    
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

# File upload
uploaded_file = st.file_uploader("Choose a cricket video file", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Clean up any existing temporary files first
    if hasattr(st.session_state, 'temp_video_path') and st.session_state.temp_video_path:
        try:
            os.unlink(st.session_state.temp_video_path)
        except:
            pass
    
    # Save uploaded file to temporary location with cleanup
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir='/tmp') as tmp_file:
            # Write in chunks to handle large files
            uploaded_file.seek(0)
            while True:
                chunk = uploaded_file.read(8192)  # 8KB chunks
                if not chunk:
                    break
                tmp_file.write(chunk)
            video_path = tmp_file.name
            st.session_state.temp_video_path = video_path
    except OSError as e:
        st.error("Video file too large for processing. Please use a smaller video file (under 100MB recommended).")
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
        st.subheader("Step 2: Stance Detection Analysis")
        
        if not st.session_state.analysis_complete:
            if st.button("Start Analysis", type="primary"):
                x1, y1, x2, y2 = st.session_state.rectangle_coords
                
                # Initialize stance detector
                stance_detector = StanceDetector(
                    stability_threshold=stability_threshold,
                    min_stability_duration=min_stability_duration,
                    confidence_threshold=confidence_threshold,
                    camera_perspective=camera_perspective,
                    batsman_height=batsman_height
                )
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Process video
                    results = []
                    frame_count = st.session_state.video_processor.get_frame_count()
                    
                    frame_idx = 0
                    while True:
                        frame = st.session_state.video_processor.get_next_frame()
                        if frame is None:
                            break
                        
                        # Update progress
                        progress = frame_idx / frame_count
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_idx + 1}/{frame_count}")
                        
                        # Crop frame to analysis area
                        cropped_frame = frame[y1:y2, x1:x2]
                        
                        # Detect stance
                        timestamp = frame_idx / st.session_state.video_processor.get_fps()
                        is_stable_stance, pose_data = stance_detector.detect_stance(cropped_frame, timestamp)
                        
                        # Store comprehensive biomechanical data for shot trigger analysis
                        biomech_data = None
                        if pose_data and pose_data.get('confidence', 0) > 0.5:
                            biomech_data = {
                                'left_ankle_x': pose_data.get('left_ankle_x', 0),
                                'left_ankle_y': pose_data.get('left_ankle_y', 0),
                                'right_ankle_x': pose_data.get('right_ankle_x', 0),
                                'right_ankle_y': pose_data.get('right_ankle_y', 0),
                                'shoulder_line_angle': pose_data.get('shoulder_line_angle', 0),
                                'hip_line_angle': pose_data.get('hip_line_angle', 0),
                                'head_tilt_angle': pose_data.get('head_tilt_angle', 0),
                                'left_knee_angle': pose_data.get('left_knee_angle', 170),
                                'right_knee_angle': pose_data.get('right_knee_angle', 170)
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
                
                # Mark shot trigger points with vertical lines
                for i, trigger in enumerate(shot_triggers):
                    fig.add_vline(
                        x=trigger['trigger_time'],
                        line=dict(color="orange", width=2, dash="dash"),
                        annotation_text=f"Shot Trigger {i+1}",
                        annotation_position="bottom",
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
                
                # Shot Trigger Analysis Results
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
                            'Sustained Frames': trigger['sustained_frames'],
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
                                                            'ankle_coordinates': 0.025
                                                        }
                                                        
                                                        # Display movement in table format
                                                        movement_table = []
                                                        for param, change in movement_data.items():
                                                            threshold = thresholds[param]
                                                            exceeds_threshold = change > threshold
                                                            
                                                            # Format display name
                                                            display_name = param.replace('_', ' ').title()
                                                            if param == 'ankle_coordinates':
                                                                unit = 'norm'
                                                                change_str = f"{change:.3f}"
                                                                threshold_str = f"{threshold:.3f}"
                                                            else:
                                                                unit = '°'
                                                                change_str = f"{change:.1f}°"
                                                                threshold_str = f"{threshold}°"
                                                            
                                                            status = "🔴 TRIGGER" if exceeds_threshold else "🟢 Normal"
                                                            
                                                            movement_table.append({
                                                                'Parameter': display_name,
                                                                'Change': change_str,
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
            
            else:
                st.warning("No analysis results available.")
            
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
