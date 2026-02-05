from datetime import datetime

uploaded_logs = []

def add_log(language, classification, confidence):
    uploaded_logs.append({
        "id": len(uploaded_logs) + 1,
        "language": language,
        "classification": classification,
        "confidenceScore": confidence,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def get_logs():
    return uploaded_logs
