import os
import torch
import cv2
import mediapipe as mp

# Configuration
model_path = "landmark_model.pth"
test_folder = "TestImages"
summary_file = "summary.txt"
output_images_dir = "TestImagesLandmarks"

# Create output folder
os.makedirs(output_images_dir, exist_ok=True)

# Load model
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
num_classes = checkpoint["num_classes"]

class LandmarkClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = LandmarkClassifier(63, num_classes)
model.load_state_dict(checkpoint["model_state"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prediction function
def predict_landmarks(landmarks_row):
    x = torch.tensor(landmarks_row, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out, 1)
        return pred.item()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Collect predictions
gesture_names = sorted([d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))])
gesture_to_idx = {name: idx for idx, name in enumerate(gesture_names)}
idx_to_gesture = {v: k for k, v in gesture_to_idx.items()}

all_results = {}
class_counts = {gesture: {"correct":0, "total":0} for gesture in gesture_names}
total_correct = 0
total_images = 0

for gesture in gesture_names:
    folder = os.path.join(test_folder, gesture)
    all_results[gesture] = []

    # Create subfolder in output directory
    save_subfolder = os.path.join(output_images_dir, gesture)
    os.makedirs(save_subfolder, exist_ok=True)

    for img_file in sorted(os.listdir(folder)):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_mp = hands.process(img_rgb)

        if results_mp.multi_hand_landmarks:
            landmarks = results_mp.multi_hand_landmarks[0]
            row = []
            for lm in landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            pred_idx = predict_landmarks(row)
            pred_gesture = idx_to_gesture[pred_idx]
            correct = pred_gesture == gesture

            # Draw landmarks on the image (rainbow style like extraction code)
            mp_drawing.draw_landmarks(
                img,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_styles.get_default_hand_connections_style()
            )
        else:
            pred_gesture = "No hand detected"
            correct = False

        all_results[gesture].append((os.path.join(gesture, img_file), pred_gesture, correct))

        class_counts[gesture]["total"] += 1
        if correct:
            class_counts[gesture]["correct"] += 1
            total_correct += 1
        total_images += 1

        # Save image with landmarks drawn
        save_path = os.path.join(save_subfolder, img_file)
        cv2.imwrite(save_path, img)

hands.close()

# Write summary.txt
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(" Gesture Accuracies \n")
    for gesture in gesture_names:
        total = class_counts[gesture]["total"]
        correct = class_counts[gesture]["correct"]
        acc = 100 * correct / total if total > 0 else 0
        f.write(f"{gesture} Accuracy: {acc:.2f}% ({correct}/{total})\n")
    overall_acc = 100 * total_correct / total_images if total_images > 0 else 0
    f.write(f"\nOverall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_images})\n\n")

    f.write(" Prediction Log \n\n")
    for gesture in gesture_names:
        f.write(f" {gesture.upper()} \n")
        for filename, pred_gesture, correct in all_results[gesture]:
            f.write(f"{filename} → Predicted: {pred_gesture} | Correct: {correct}\n")
        f.write("\n")

print(f"Inference complete. Summary saved to {summary_file}")
print(f"Landmark images saved in {output_images_dir}")

