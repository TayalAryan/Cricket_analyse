"""
MediaPipe Z Coordinate Explanation

The Z coordinate in MediaPipe pose landmarks represents the relative depth/distance 
from the camera plane. Here's what you need to know:

1. WHAT IS Z COORDINATE:
   - Z represents the landmark's depth relative to the hip center
   - Positive Z = closer to camera than hip center
   - Negative Z = further from camera than hip center
   - Z = 0 would be at the same depth as the hip center

2. HOW IT'S CALCULATED:
   - MediaPipe uses the midpoint between left and right hips as the reference (Z=0)
   - All other landmarks are measured relative to this hip center depth
   - The values are normalized and scaled based on the person's overall size

3. PRACTICAL MEANING:
   - For a cricket batsman facing sideways:
     * Face/front shoulder: typically positive Z (closer to camera)
     * Back shoulder: typically negative Z (further from camera)
     * Feet: usually close to Z=0 (similar depth as hips)
     * Extended arms: can be positive or negative depending on batting stance

4. IMPORTANT LIMITATIONS:
   - Z is estimated from a single camera view (not true 3D measurement)
   - Less accurate than X,Y coordinates
   - Should be used for relative comparisons, not absolute distances
   - Accuracy depends on pose estimation confidence

5. USE CASES IN CRICKET ANALYSIS:
   - Detecting body rotation during swing
   - Measuring shoulder alignment (facing camera vs sideways)
   - Tracking forward/backward movement of limbs
   - Analyzing batting stance depth changes

6. TECHNICAL NOTES:
   - Z values are typically in range of -0.5 to +0.5
   - Scale is proportional to person size in the frame
   - More reliable for torso landmarks than extremities
"""

# Example of how Z coordinates change during cricket batting:
example_z_values = {
    "stance_phase": {
        "left_shoulder": -0.1,   # Slightly back from hips
        "right_shoulder": 0.1,   # Slightly forward (facing camera)
        "nose": 0.2,             # Face forward toward camera
        "left_wrist": -0.2,      # Back hand position
        "right_wrist": 0.0       # Front hand near hip depth
    },
    "swing_phase": {
        "left_shoulder": 0.2,    # Rotating forward during swing
        "right_shoulder": -0.1,  # Rotating back
        "nose": 0.1,             # Head still mostly forward
        "left_wrist": 0.3,       # Following through forward
        "right_wrist": 0.1       # Moving forward with swing
    }
}

print("MediaPipe Z coordinates represent depth relative to hip center")
print("Positive = closer to camera, Negative = further from camera")