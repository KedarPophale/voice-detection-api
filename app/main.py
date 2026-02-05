from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
import base64
import os

from app.schemas import VoiceRequest, VoiceResponse
from app.auth import verify_api_key
from app.audio_utils import decode_mp3, extract_features
from app.model import predict_voice
from app.storage import add_log, get_logs

app = FastAPI(
    title="AI Voice Detection API",
    version="1.0.0"
)

SUPPORTED_LANGUAGES = [
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
]

# ----------------------------
# BASE64 API (PS REQUIRED)
# ----------------------------
@app.post("/api/voice-detection", response_model=VoiceResponse)
def detect_voice(
    request: VoiceRequest,
    api_key: str = Depends(verify_api_key)
):
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    audio, sr = decode_mp3(request.audioBase64)
    features = extract_features(audio, sr)

    classification, confidence, explanation = predict_voice(features)
    add_log(request.language, classification, round(confidence, 2))

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }

# ----------------------------
# FILE UPLOAD API (AUTO)
# ----------------------------
@app.post("/api/voice-detection/upload")
def detect_voice_upload(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if file.content_type != "audio/mpeg":
        raise HTTPException(status_code=400, detail="Only MP3 files allowed")

    audio_bytes = file.file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    audio, sr = decode_mp3(audio_base64)
    features = extract_features(audio, sr)

    classification, confidence, explanation = predict_voice(features)

    return {
        "status": "success",
        "language": "Unknown",
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }

# ----------------------------
# LOG VIEW (OPTIONAL)
# ----------------------------
@app.get("/api/uploads")
def view_uploads(api_key: str = Depends(verify_api_key)):
    logs = get_logs()
    return {
        "status": "success",
        "totalUploads": len(logs),
        "data": logs
    }
