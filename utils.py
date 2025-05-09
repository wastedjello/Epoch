import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# === For CNN spectrogram generation ===
def create_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()



# === For fusion model: Extract pitch, loudness, speaking rate ===
def extract_audio_features(audio_path, n_pitch=100):
    try:
        y, sr = librosa.load(audio_path, sr=None)

        # --- Pitch ---
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:
                pitch_values.append(pitch)
        pitch_array = np.array(pitch_values[:n_pitch])
        if len(pitch_array) < n_pitch:
            pitch_array = np.pad(pitch_array, (0, n_pitch - len(pitch_array)), mode='constant')

        # --- Loudness (RMS Energy) ---
        rms = librosa.feature.rms(y=y)[0]
        loudness = np.mean(rms)

        # --- Speaking Rate (estimated syllable rate using zero crossings as proxy) ---
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        speaking_rate = np.mean(zcr)

        # Combine features
        features = np.concatenate([pitch_array, [loudness, speaking_rate]])
        return features

    except Exception as e:
        print(f"[ERROR] Extracting features from {audio_path}: {e}")
        return np.zeros(n_pitch + 2)
