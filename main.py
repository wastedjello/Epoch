import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

# Paths
RAVDESS_AUDIO_DIR = "audio_data/audio_speech_actors_01-24"
OUTPUT_DIR = "spectrograms"
CSV_FILE = "spectrogram_labels.csv"

# Emotion mapping from RAVDESS filename
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Create output folders
for emotion in emotion_map.values():
    os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

# Helper function to create and save a spectrogram
def save_spectrogram(file_path, save_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_DB, sr=sr, cmap='viridis')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# List to store CSV entries
csv_data = []

# Process audio files
for actor_folder in tqdm(os.listdir(RAVDESS_AUDIO_DIR), desc="Processing Actors"):
    actor_path = os.path.join(RAVDESS_AUDIO_DIR, actor_folder)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)
        parts = file.split('-')
        emotion_id = parts[2]
        emotion_label = emotion_map.get(emotion_id)

        if emotion_label:
            save_name = os.path.splitext(file)[0] + ".png"
            relative_path = os.path.join(emotion_label, save_name)
            save_path = os.path.join(OUTPUT_DIR, relative_path)

            save_spectrogram(file_path, save_path)
            csv_data.append([relative_path, emotion_label])

# Save CSV
df = pd.DataFrame(csv_data, columns=["filename", "emotion"])
df.to_csv(CSV_FILE, index=False)

print(f"\nSpectrograms saved in: {OUTPUT_DIR}")
print(f"CSV file created: {CSV_FILE}")
