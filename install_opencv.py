#!/usr/bin/env python3

import subprocess
import sys
import os

def install_opencv_headless():
    """Install only opencv-python-headless without GUI dependencies"""
    try:
        # Set environment variables to avoid GUI dependencies
        env = os.environ.copy()
        env['OPENCV_DISABLE_GUI'] = '1'
        
        # Install only headless version
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'opencv-python-headless==4.8.1.78', 
            '--no-deps', '--force-reinstall'
        ], env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("OpenCV headless installed successfully")
        else:
            print(f"Installation failed: {result.stderr}")
            
        # Test the installation
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        print(f"VideoCapture available: {hasattr(cv2, 'VideoCapture')}")
        
        if hasattr(cv2, 'VideoCapture'):
            print("✓ OpenCV installation successful")
        else:
            print("✗ OpenCV installation incomplete")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    install_opencv_headless()