from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from app.schemas import VoiceRequest, VoiceResponse
from app.auth import verify_api_key
from app.audio_utils import decode_mp3, extract_features
from app.model import predict_voice
from app.language_detector import detect_language
from app.storage import add_log, get_logs
import base64

app = FastAPI(
    title="AI Generated Voice Detection API",
    description="Detect whether an uploaded voice sample is AI-generated or Human",
    version="1.0.0"
)

SUPPORTED_LANGUAGES = [
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
]

# -------------------------------------------------------------------
# BASE64 API (Optional – for programmatic use)
# -------------------------------------------------------------------
@app.post("/api/voice-detection", response_model=VoiceResponse)
def detect_voice_base64(
    request: VoiceRequest,
    api_key: str = Depends(verify_api_key)
):
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    audio, sr = decode_mp3(request.audioBase64)

    detected_language = detect_language(audio, sr)
    if detected_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    features = extract_features(audio, sr)
    classification, confidence, explanation = predict_voice(features)

    add_log(detected_language, classification, round(confidence, 2))

    return {
        "status": "success",
        "language": detected_language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }

# -------------------------------------------------------------------
# FILE UPLOAD API (RECOMMENDED)
# -------------------------------------------------------------------
@app.post("/api/voice-detection/upload")
def detect_voice_upload(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    if file.content_type != "audio/mpeg":
        raise HTTPException(status_code=400, detail="Only MP3 files allowed")

    audio_bytes = file.file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    try:
        audio, sr = decode_mp3(audio_base64)

        detected_language = detect_language(audio, sr)
        features = extract_features(audio, sr)

        classification, confidence, explanation = predict_voice(features)

        add_log(detected_language, classification, round(confidence, 2))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "language": detected_language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }

# -------------------------------------------------------------------
# LOGS / AUDIT ENDPOINT
# -------------------------------------------------------------------
@app.get("/api/uploads")
def view_uploaded_data(api_key: str = Depends(verify_api_key)):
    logs = get_logs()
    return {
        "status": "success",
        "totalUploads": len(logs),
        "data": logs
    }
