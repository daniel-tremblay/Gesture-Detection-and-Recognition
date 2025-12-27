import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # <-- for plotting graphs

# Load and Prepare Dataset

# Load CSV file containing hand landmarks and labels
df = pd.read_csv("landmarks.csv")

# Split features (X) and labels (y)
X = df.drop("label", axis=1).values
y = df["label"].values

# Train/validation split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Create DataLoaders for batching
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

# Define Model

class LandmarkClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LandmarkClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Determine number of unique classes
num_classes = len(df["label"].unique())
model = LandmarkClassifier(input_size=63, num_classes=num_classes)

# Training Configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []

# Training Loop

epochs = 500
print("Starting training...\n")

for epoch in range(epochs):
    model.train()  # training mode
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

    acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracies.append(acc)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

print("\nTraining complete!")

# Save Model

torch.save({
    "model_state": model.state_dict(),
    "num_classes": num_classes
}, "landmark_model.pth")

print("Model saved as landmark_model.pth")

# Plot and Save Graphs

plt.figure(figsize=(10, 5))

# Plot Loss Curve
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.legend()

# Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy vs Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

print("Training graphs saved as training_curves.png")
