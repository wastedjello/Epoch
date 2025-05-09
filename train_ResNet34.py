import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from fusion_dutils import SpectrogramAudioDataset
from models.model_resnet_fusion import FusionModel
import os
import numpy as np
from torchvision import transforms

# ---- Hyperparameters ----
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.0006
WEIGHT_DECAY = 6e-5
TRAIN_DATA_PATH = "spectrograms"  # Path to spectrogram images
TOTAL_AUDIO_FEATURES = 102  # pitch(100) + loudness(1) + speaking rate(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Image Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- Prepare dataset and loader ----
all_image_paths = []
all_labels = []
all_audio_features = []
class_to_idx = {}

# Create a list of all image paths, labels, and corresponding audio features
for idx, class_name in enumerate(sorted(os.listdir(TRAIN_DATA_PATH))):
    class_dir = os.path.join(TRAIN_DATA_PATH, class_name)
    if os.path.isdir(class_dir):
        class_to_idx[class_name] = idx
        for fname in os.listdir(class_dir):
            if fname.endswith(".png"):
                all_image_paths.append(os.path.join(class_dir, fname))
                all_labels.append(idx)
                # Add corresponding audio features (ensure you have this in your dataset)
                audio_feats = np.random.rand(TOTAL_AUDIO_FEATURES).tolist()  # Replace with actual data
                all_audio_features.append(audio_feats)

# Create dataset and splits
dataset = SpectrogramAudioDataset(all_image_paths, all_labels, all_audio_features, transform=transform)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Model, Loss, Optimizer ----
model = FusionModel(audio_features=TOTAL_AUDIO_FEATURES, num_classes=len(class_to_idx))
model = model.to(DEVICE)

# CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()

# Optimizer (Adam with weight decay for regularization)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ---- Training Loop ----
for epoch in range(1, EPOCHS + 1):
    model.train()  # Set model to training mode
    train_loss, train_correct = 0, 0

    for images, audio_feats, labels in train_loader:
        images, audio_feats, labels = images.to(DEVICE), audio_feats.to(DEVICE), labels.to(DEVICE)

        # Forward pass: compute model outputs (logits)
        outputs = model(images, audio_feats)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track training loss and accuracy
        train_loss += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # Calculate average training loss and accuracy
    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    # ---- Validation Loop ----
    model.eval()  # Set model to evaluation mode
    val_loss, val_correct = 0, 0
    with torch.no_grad():  # No gradients needed during evaluation
        for images, audio_feats, labels in val_loader:
            images, audio_feats, labels = images.to(DEVICE), audio_feats.to(DEVICE), labels.to(DEVICE)

            # Forward pass: compute model outputs (logits)
            outputs = model(images, audio_feats)

            # Compute loss
            loss = criterion(outputs, labels)

            # Track validation loss and accuracy
            val_loss += loss.item() * labels.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    # Calculate average validation loss and accuracy
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    # Print the results for the epoch
    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
