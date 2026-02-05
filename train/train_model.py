import librosa
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

X = []
y = []

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=26
    )

    features = np.mean(mfcc.T, axis=0)  # shape: (26,)
    return features

for label in ["human", "ai"]:
    folder = os.path.join("data", label)

    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):
        if not file.lower().endswith(".mp3"):
            continue

        file_path = os.path.join(folder, file)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append("HUMAN" if label == "human" else "AI_GENERATED")
        except Exception as e:
            print(f"Skipping {file}: {e}")

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/voice_detector.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully with 26 MFCC features")
