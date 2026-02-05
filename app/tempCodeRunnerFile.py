import base64
import librosa
import numpy as np
import io

def decode_mp3(base64_audio: str):
    audio_bytes = base64.b64decode(base64_audio)
    audio_buffer = io.BytesIO(audio_bytes)
    audio, sr = librosa.load(audio_buffer, sr=16000, mono=True)
    return audio, sr

def extract_features(audio, sr):
    features = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    pitch_values = pitches[pitches > 0]
    features.append(np.mean(pitch_values) if len(pitch_values) else 0)
    features.append(np.std(pitch_values) if len(pitch_values) else 0)

    features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_flatness(y=audio)))

    return np.array(features).reshape(1, -1)
