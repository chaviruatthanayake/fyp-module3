import joblib
from video_process.extract_eye_features import extract_features_from_video

# Load model once
model = joblib.load("eye_reading_model_advanced.pkl")

def predict_eye_behavior(video_path: str, threshold: float = 0.6):
    """
    Predict if the person is reading (engaged) using ML model.
    Returns:
        label (str): "reading" or "not_reading"
        confidence (float): probability of prediction
        is_attentive (bool): if confidence is above threshold
    """
    features = extract_features_from_video(video_path)
    if not features or len(features) != 10:
        return "undetected", 0.0, False

    prob = model.predict_proba([features])[0]
    label = model.classes_[prob.argmax()]
    confidence = max(prob)

    is_attentive = label == "reading" and confidence >= threshold
    return label, round(confidence * 100, 2), is_attentive
