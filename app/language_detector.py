import whisper
import numpy as np

# Load model once
model = whisper.load_model("base")

def detect_language(audio, sr):
    """
    Robust language detection from raw audio
    """

    # 1️⃣ Convert to numpy float32
    audio = np.asarray(audio, dtype=np.float32)

    # 2️⃣ Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 3️⃣ Whisper expects 16kHz
    if sr != 16000:
        raise ValueError("Audio sample rate must be 16000 Hz")

    # 4️⃣ Pad / trim audio to 30s (CRITICAL)
    audio = whisper.pad_or_trim(audio)

    # 5️⃣ Convert to mel spectrogram
    mel = whisper.log_mel_spectrogram(audio)

    # 6️⃣ Detect language
    _, probs = model.detect_language(mel)
    lang_code = max(probs, key=probs.get)

    # Map to allowed languages
    language_map = {
        "en": "English",
        "ta": "Tamil",
        "hi": "Hindi",
        "ml": "Malayalam",
        "te": "Telugu"
    }

    return language_map.get(lang_code, "Unknown")
