import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Configuration
data_dir = "Data"
output_csv = "landmarks.csv"
output_images_dir = "Landmark_Images"

# Create folder for landmark images
os.makedirs(output_images_dir, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Get gesture classes
gesture_names = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
gesture_to_idx = {name: idx for idx, name in enumerate(gesture_names)}

# Process images and extract landmarks
all_rows = []

with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for gesture in gesture_names:
        folder = os.path.join(data_dir, gesture)
        gesture_output_dir = os.path.join(output_images_dir, gesture)
        os.makedirs(gesture_output_dir, exist_ok=True)

        for img_file in os.listdir(folder):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]

                # Save landmark coordinates into CSV data
                row = []
                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(gesture_to_idx[gesture])
                all_rows.append(row)

                # Create a blank white canvas same size as input image
                blank_img = np.ones_like(img) * 255

                # Draw landmarks & connections on blank image
                mp_drawing.draw_landmarks(
                    blank_img,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_styles.get_default_hand_connections_style()
                )

                # CROP HAND REGION
                h, w, _ = img.shape
                x_coords = [lm.x * w for lm in landmarks.landmark]
                y_coords = [lm.y * h for lm in landmarks.landmark]

                # Bounding box with padding
                x_min, x_max = max(0, int(min(x_coords) - 20)), min(w, int(max(x_coords) + 20))
                y_min, y_max = max(0, int(min(y_coords) - 20)), min(h, int(max(y_coords) + 20))

                cropped_hand = blank_img[y_min:y_max, x_min:x_max]

                # Save the cropped landmark image
                save_path = os.path.join(gesture_output_dir, img_file)
                cv2.imwrite(save_path, cropped_hand)

            else:
                print(f"No hand detected in {img_path}")

# Save landmarks to CSV
columns = [f"{axis}{i+1}" for i in range(21) for axis in ["x", "y", "z"]] + ["label"]
df = pd.DataFrame(all_rows, columns=columns)
df.to_csv(output_csv, index=False)

print(f"\nLandmarks saved to: {output_csv}")
print(f"Cropped landmark images saved under: {output_images_dir}")
print("Gesture mapping:", gesture_to_idx)