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

# File upload
uploaded_file = st.file_uploader("Choose a cricket video file", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
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
                    camera_perspective=camera_perspective
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
                        
                        results.append({
                            'frame': frame_idx,
                            'timestamp': timestamp,
                            'is_stable_stance': is_stable_stance,
                            'pose_confidence': pose_data.get('confidence', 0) if pose_data else 0
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
                    avg_confidence = np.mean([r['pose_confidence'] for r in all_results if r['pose_confidence'] > 0])
                    st.metric("Avg Pose Confidence", f"{avg_confidence:.2f}")
                
                # Timeline visualization
                st.subheader("Stance Detection Timeline")
                
                # Prepare data for timeline
                timestamps = [r['timestamp'] for r in all_results]
                is_stable = [1 if r['is_stable_stance'] else 0 for r in all_results]
                confidences = [r['pose_confidence'] for r in all_results]
                
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
                
                # Highlight stable periods
                for period in stable_periods:
                    fig.add_vrect(
                        x0=period['start_time'],
                        x1=period['end_time'],
                        fillcolor="rgba(255,0,0,0.2)",
                        layer="below",
                        line_width=0,
                    )
                
                fig.update_layout(
                    title="Cricket Stance Detection Timeline",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Stable Stance (0/1)",
                    yaxis2=dict(
                        title="Pose Confidence",
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
                        periods_data.append({
                            'Period': i + 1,
                            'Start Time (s)': f"{period['start_time']:.2f}",
                            'End Time (s)': f"{period['end_time']:.2f}",
                            'Duration (s)': f"{period['duration']:.2f}",
                            'Avg Confidence': f"{period['avg_confidence']:.3f}"
                        })
                    
                    st.table(periods_data)
                
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
                        camera_perspective=camera_perspective
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
                                                
                                                # Show stance and pose detection status
                                                frame_result = next((r for r in all_results if abs(r['timestamp'] - timestamp) < 0.1), None)
                                                
                                                col_status1, col_status2 = st.columns(2)
                                                with col_status1:
                                                    if frame_result and frame_result['is_stable_stance']:
                                                        st.success("✓ Stable Stance")
                                                    else:
                                                        st.info("○ Not Stable")
                                                
                                                with col_status2:
                                                    if pose_detected:
                                                        st.success("✓ Pose Detected")
                                                    else:
                                                        st.warning("⚠ No Pose")
                                            
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
    - Head facing towards the bowler (configurable side)
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
