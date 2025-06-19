#!/usr/bin/env python3

import sys
import os

# Test imports one by one to identify the problematic library
print("Testing imports...")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except Exception as e:
    print(f"✗ NumPy failed: {e}")

try:
    import streamlit as st
    print("✓ Streamlit imported successfully")
except Exception as e:
    print(f"✗ Streamlit failed: {e}")

try:
    import cv2
    print("✓ OpenCV imported successfully")
except Exception as e:
    print(f"✗ OpenCV failed: {e}")

try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
except Exception as e:
    print(f"✗ MediaPipe failed: {e}")

try:
    import plotly
    print("✓ Plotly imported successfully")
except Exception as e:
    print(f"✗ Plotly failed: {e}")

print("Import test completed.")