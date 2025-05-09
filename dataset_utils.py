import os
import glob
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

emotion_mapping = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}

class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def load_dataset(spectrogram_dir="spectrograms", labels_file="spectrogram_labels.csv", test_size=0.2):
    image_paths = []
    labels = []

    # Read the CSV file that contains the label information
    data = pd.read_csv(labels_file)

    # Ensure 'filename' and 'emotion' columns exist
    if 'filename' not in data.columns or 'emotion' not in data.columns:
        raise KeyError("'filename' or 'emotion' column missing from the CSV file.")

    # Map emotion to label
    for index, row in data.iterrows():
        # Correct the file path using os.path.join
        image_path = os.path.join(spectrogram_dir, row['filename'].replace("\\", "/"))
        image_paths.append(image_path)
        labels.append(emotion_mapping[row['emotion']])  # Using 'emotion' instead of 'label'

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = SpectrogramDataset(train_paths, train_labels, transform)
    val_dataset = SpectrogramDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader
