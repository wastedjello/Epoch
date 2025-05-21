# ---- Training Script ----

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import numpy as np
from fusion_dutils import SpectrogramAudioDataset
from models.model_fusion import FusionModel  # Make sure this import path is correct

# ---- Hyperparameters ----
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.0006
WEIGHT_DECAY = 6e-5
TRAIN_DATA_PATH = "spectrograms"
TOTAL_AUDIO_FEATURES = 102
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Image Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- Dataset Preparation ----
all_image_paths, all_labels, all_audio_features = [], [], []
class_to_idx = {}

for idx, class_name in enumerate(sorted(os.listdir(TRAIN_DATA_PATH))):
    class_dir = os.path.join(TRAIN_DATA_PATH, class_name)
    if os.path.isdir(class_dir):
        class_to_idx[class_name] = idx
        for fname in os.listdir(class_dir):
            if fname.endswith(".png"):
                all_image_paths.append(os.path.join(class_dir, fname))
                all_labels.append(idx)
                audio_feats = np.random.rand(TOTAL_AUDIO_FEATURES).tolist()  # Replace with real features
                all_audio_features.append(audio_feats)

# ---- Dataset and Loaders ----
dataset = SpectrogramAudioDataset(all_image_paths, all_labels, all_audio_features, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Model, Loss, Optimizer ----
model = FusionModel(audio_features=TOTAL_AUDIO_FEATURES, num_classes=len(class_to_idx)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ---- Training and Validation Loop ----
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_train_loss, correct_train = 0.0, 0

    for images, audio_feats, labels in train_loader:
        images, audio_feats, labels = images.to(DEVICE), audio_feats.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, audio_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * labels.size(0)
        correct_train += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_train_loss / len(train_loader.dataset)
    train_acc = correct_train / len(train_loader.dataset)

    model.eval()
    running_val_loss, correct_val = 0.0, 0
    with torch.no_grad():
        for images, audio_feats, labels in val_loader:
            images, audio_feats, labels = images.to(DEVICE), audio_feats.to(DEVICE), labels.to(DEVICE)
            outputs = model(images, audio_feats)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * labels.size(0)
            correct_val += (outputs.argmax(1) == labels).sum().item()

    val_loss = running_val_loss / len(val_loader.dataset)
    val_acc = correct_val / len(val_loader.dataset)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
