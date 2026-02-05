import pickle

with open("model/voice_detector.pkl", "rb") as f:
    model = pickle.load(f)

def predict_voice(features):
    # Safety check (professional ML practice)
    if features.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: got {features.shape[1]}, "
            f"expected {model.n_features_in_}"
        )

    probabilities = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]

    confidence = float(max(probabilities))

    explanation = (
        "Unnatural pitch consistency and robotic speech patterns detected"
        if prediction == "AI_GENERATED"
        else "Natural pitch variation and human speech dynamics detected"
    )

    return prediction, confidence, explanation
