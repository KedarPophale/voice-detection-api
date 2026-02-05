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
    """
    Extract EXACTLY 26 MFCC features
    (must match training pipeline)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=26
    )

    features = np.mean(mfcc.T, axis=0)  # shape: (26,)
    return features.reshape(1, -1)
