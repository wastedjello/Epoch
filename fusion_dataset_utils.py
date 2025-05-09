import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class SpectrogramAudioDataset(Dataset):
    def __init__(self, image_paths, labels, audio_features, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.audio_features = audio_features  # Assuming it's a list of numpy arrays or tensors
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Get the corresponding label and audio features
        label = self.labels[idx]
        audio_feats = torch.tensor(self.audio_features[idx], dtype=torch.float32)

        return image, audio_feats, label
