import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Helper function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    horizontal_positions = []
    vertical_positions = []
    head_positions_x = []
    head_positions_y = []

    blinks = 0
    blink_frames = 0
    blink_durations = []
    EAR_THRESHOLD = 0.2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            # Eye landmarks (left eye example)
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            left_eye = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in left_eye_idx])

            # EAR for blink detection
            ear = eye_aspect_ratio(left_eye)
            if ear < EAR_THRESHOLD:
                blink_frames += 1
            else:
                if blink_frames > 1:
                    blinks += 1
                    blink_durations.append(blink_frames)
                blink_frames = 0

            # Eye center
            eye_center = np.mean(left_eye, axis=0)
            horizontal_positions.append(eye_center[0])
            vertical_positions.append(eye_center[1])

            # Head position (using nose tip landmark)
            nose = landmarks.landmark[1]
            head_positions_x.append(nose.x * w)
            head_positions_y.append(nose.y * h)

    cap.release()

    # Compute basic stats
    horiz_var = np.var(horizontal_positions) if horizontal_positions else 0
    vert_var = np.var(vertical_positions) if vertical_positions else 0
    horiz_std = np.std(horizontal_positions) if horizontal_positions else 0
    vert_std = np.std(vertical_positions) if vertical_positions else 0
    horiz_mean = np.mean(horizontal_positions) if horizontal_positions else 0
    vert_mean = np.mean(vertical_positions) if vertical_positions else 0

    # Head movement variance
    head_var_x = np.var(head_positions_x) if head_positions_x else 0
    head_var_y = np.var(head_positions_y) if head_positions_y else 0

    # Blink metrics
    avg_blink_duration = np.mean(blink_durations) if blink_durations else 0

    return [
        horiz_var, vert_var, horiz_std, vert_std, horiz_mean, vert_mean,
        head_var_x, head_var_y, blinks, avg_blink_duration
    ]

def process_dataset(dataset_path="dataset"):
    data = []
    for label in ["reading", "not_reading"]:
        folder = os.path.join(dataset_path, label)
        for video_file in os.listdir(folder):
            if video_file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(folder, video_file)
                print(f"Processing: {video_path}")
                features = extract_features_from_video(video_path)
                data.append([video_file] + features + [label])

    columns = [
        "video_name",
        "horizontal_var", "vertical_var", "horizontal_std", "vertical_std",
        "horizontal_mean", "vertical_mean",
        "head_var_x", "head_var_y",
        "blink_count", "avg_blink_duration",
        "label"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("eye_features_advanced.csv", index=False)
    print("\nAdvanced feature extraction completed. Saved as eye_features_advanced.csv.")

if __name__ == "__main__":
    process_dataset("dataset")
