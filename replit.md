# Cricket Stance Detection System

## Overview

This is a computer vision application built with Streamlit that analyzes cricket batting stances from video footage. The system uses MediaPipe for pose detection and analyzes batsman stability to identify key batting stances. The application provides an interactive web interface for uploading cricket videos and viewing stance analysis results with detailed biomechanical insights.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **User Interface**: Single-page application with sidebar controls
- **Visualization**: Plotly for interactive charts and graphs
- **Image Processing**: OpenCV and PIL for video frame manipulation
- **Deployment**: Autoscale deployment on port 5000

### Backend Architecture
- **Core Processing**: Python-based modular architecture
- **Pose Detection**: MediaPipe ML framework for human pose estimation
- **Video Processing**: OpenCV for video frame extraction and analysis
- **State Management**: Streamlit session state for maintaining application state

## Key Components

### 1. Main Application (`app.py`)
- Streamlit web interface entry point
- Handles file uploads and user interactions
- Manages application state and workflow
- Integrates all components for complete stance analysis

### 2. Stance Detection Engine (`stance_detector.py`)
- **Core Function**: Analyzes pose stability and movement patterns
- **Pose Estimation**: Uses MediaPipe with configurable confidence thresholds
- **Stability Analysis**: Tracks movement history over time windows
- **Biomechanics**: Calculates center of gravity using body segment weights
- **Camera Perspective**: Supports left/right bowler positioning configurations

### 3. Video Processing (`video_processor.py`)
- **Video Handling**: Manages video file loading and frame extraction
- **Frame Navigation**: Provides sequential frame access
- **Video Properties**: Extracts FPS, dimensions, and duration
- **Memory Management**: Efficient frame processing for large videos

### 4. Utility Functions (`utils.py`)
- **Drawing Operations**: Rectangle overlay functions for ROI selection
- **Coordinate Validation**: Ensures boundaries stay within image dimensions
- **Helper Functions**: Common operations for image manipulation

## Data Flow

1. **Video Upload**: User uploads cricket video through Streamlit interface
2. **Initial Processing**: VideoProcessor extracts first frame and video properties
3. **ROI Selection**: User defines region of interest for batsman detection
4. **Configuration**: Camera perspective and detection parameters set via sidebar
5. **Frame Analysis**: Each frame processed through MediaPipe pose detection
6. **Stability Assessment**: Movement patterns analyzed over time windows
7. **Results Generation**: Stance detection results compiled with timing data
8. **Visualization**: Interactive charts and annotated frames displayed

## External Dependencies

### Core ML/CV Libraries
- **MediaPipe (>=0.10.21)**: Google's ML framework for pose estimation
- **OpenCV (>=4.11.0.86)**: Computer vision operations and video processing
- **NumPy (>=1.26.4)**: Numerical computing for array operations

### Web Framework & Visualization
- **Streamlit (>=1.45.1)**: Web application framework
- **Plotly (>=6.1.2)**: Interactive plotting and visualization
- **Pillow (>=11.2.1)**: Image processing utilities

### System Dependencies (Nix)
- Graphics libraries: libGL, libGLU for hardware acceleration
- Image format support: libjpeg, libwebp, libtiff, openjpeg
- GUI toolkit: tcl, tk for potential desktop features

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Nix package manager
- **Python Version**: 3.11
- **Package Management**: UV for dependency resolution
- **Configuration**: Auto-scaling deployment target

### Production Considerations
- **Port Configuration**: External port 80 mapped to internal port 5000
- **File Upload Limits**: 1000MB max upload size configured
- **CORS**: Disabled for simplified deployment
- **Performance**: Optimized for video processing workloads

### Workflow Automation
- **Parallel Execution**: Dependency installation and app startup
- **Service Management**: Automated port waiting and health checks
- **Reset Functionality**: Complete application state reset capability

## Changelog

Changelog:
- June 13, 2025. Initial setup
- June 14, 2025. Added "Stance Stability Check" section with comprehensive 14-parameter biomechanical analysis chart
- June 19, 2025. Fixed critical wrist coordinate extraction bug in stance_detector.py
- June 19, 2025. **BLOCKED**: OpenCV libGL.so.1 dependency issue preventing application startup - support ticket created
- June 26, 2025. Completely removed cricket events marking functionality as requested by user

## Current Status

**Application Status**: BLOCKED - Dependency Issue
- Core functionality complete with critical bug fixes applied
- OpenCV VideoCapture dependency preventing application startup
- All code ready for deployment once system dependencies resolved
- Support ticket submitted for libGL.so.1 library configuration

**Ready Components**:
- ✓ Wrist coordinate extraction bug fixed in stance detection
- ✓ Comprehensive biomechanical analysis with 14 parameters
- ✓ Interactive Streamlit interface with video upload
- ✓ Stance stability detection with 300ms window analysis
- ✓ Shot trigger analysis and directional movement tracking

## User Preferences

Preferred communication style: Simple, everyday language.