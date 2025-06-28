import mediapipe as mp

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose

# Get all pose landmarks
pose_landmarks = mp_pose.PoseLandmark

print("All MediaPipe Pose Landmarks (33 total):")
print("=========================================")

for i, landmark in enumerate(pose_landmarks):
    print(f"{i:2d}. {landmark.name}")

print(f"\nTotal landmarks: {len(pose_landmarks)}")