# Cricket Stance Detection - Terminal Setup Guide

## Running from Python Terminal

### Method 1: Direct Streamlit Command (Recommended)

#### Prerequisites
1. **Python Installation**: Python 3.11 or higher
2. **Required Dependencies**: All packages must be installed in your Python environment

#### Step 1: Install Required Packages
```bash
pip install streamlit opencv-python mediapipe plotly pillow numpy
```

#### Step 2: Navigate to Project Directory
```bash
cd /path/to/cricket-stance-detection
```

#### Step 3: Run the Application
```bash
streamlit run app.py --server.port 5000
```

#### Expected Output
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:5000
Network URL: http://192.168.x.x:5000
```

#### Step 4: Access the Application
- Open your web browser
- Navigate to: `http://localhost:5000`
- The cricket stance detection interface will load

## System Requirements

### Hardware Requirements
- **RAM**: Minimum 4GB, recommended 8GB+
- **CPU**: Multi-core processor recommended for video processing
- **Storage**: At least 1GB free space for video uploads

### Software Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: Version 3.11 or higher
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

### Network Requirements
- **Port Access**: Port 5000 must be available
- **Internet**: Required for initial package downloads

## Package Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.45.1 | Web application framework |
| opencv-python | ≥4.11.0.86 | Video processing and computer vision |
| mediapipe | ≥0.10.21 | Pose detection and landmark extraction |
| plotly | ≥6.1.2 | Interactive charts and visualization |
| pillow | ≥11.2.1 | Image processing utilities |
| numpy | ≥1.26.4 | Numerical computing |

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
**Error**: `Port 5000 is already in use`
**Solution**: Use a different port
```bash
streamlit run app.py --server.port 8501
```

#### 2. Package Import Errors
**Error**: `ModuleNotFoundError: No module named 'cv2'`
**Solution**: Install missing packages
```bash
pip install opencv-python
```

#### 3. Permission Errors
**Error**: `Permission denied`
**Solution**: Run with appropriate permissions or use virtual environment

#### 4. Memory Issues
**Error**: Application crashes during video processing
**Solution**: 
- Reduce video file size
- Close other applications
- Increase system RAM if possible

### Performance Tips

1. **Video Size**: Keep uploaded videos under 500MB for optimal performance
2. **Browser**: Use Chrome or Firefox for best compatibility
3. **System Resources**: Close unnecessary applications during video analysis
4. **Network**: Use stable internet connection for package downloads

## Alternative Installation Methods

### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv cricket_env

# Activate virtual environment
# Windows:
cricket_env\Scripts\activate
# macOS/Linux:
source cricket_env/bin/activate

# Install packages
pip install streamlit opencv-python mediapipe plotly pillow numpy

# Run application
streamlit run app.py --server.port 5000
```

### Using Conda
```bash
# Create conda environment
conda create -n cricket_env python=3.11

# Activate environment
conda activate cricket_env

# Install packages
conda install streamlit opencv pillow numpy
pip install mediapipe plotly

# Run application
streamlit run app.py --server.port 5000
```

## Application Features Available in Terminal Mode

- ✓ Full video upload and processing capabilities
- ✓ Interactive pose detection visualization
- ✓ Biomechanical analysis charts (Angles, Distances, Speed)
- ✓ Weight distribution analysis with In-Transition detection
- ✓ CSV export for Body Landmarks data
- ✓ Cricket event timing specification
- ✓ Region of Interest selection
- ✓ Camera perspective configuration

## Stopping the Application

To stop the application:
1. Return to terminal where app is running
2. Press `Ctrl+C` (Windows/Linux) or `Cmd+C` (macOS)
3. Confirm termination if prompted

## File Structure Required

Ensure these files are present in your project directory:
```
cricket-stance-detection/
├── app.py                 # Main Streamlit application
├── stance_detector.py     # Core pose detection logic
├── video_processor.py     # Video handling utilities
├── utils.py              # Helper functions
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── TERMINAL_SETUP.md     # This guide
```

---

**Last Updated**: July 2, 2025
**Compatible With**: Python 3.11+, All major operating systems