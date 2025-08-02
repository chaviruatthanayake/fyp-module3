import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fer import FER
import cv2
import librosa
import numpy as np
import os

# Load text-based personality model (Big Five)
MODEL_NAME = "Minej/bert-base-personality"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Predict Big Five traits from text
def predict_personality_from_text(text: str):
    if not text.strip():
        return {t: 50.0 for t in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().tolist()
    return {
        "Openness": round((logits[0] + 1) / 2 * 100, 1),
        "Conscientiousness": round((logits[1] + 1) / 2 * 100, 1),
        "Extraversion": round((logits[2] + 1) / 2 * 100, 1),
        "Agreeableness": round((logits[3] + 1) / 2 * 100, 1),
        "Neuroticism": round((logits[4] + 1) / 2 * 100, 1),
    }

# Analyze emotions from video frames
def analyze_video_emotions(video_path):
    detector = FER(mtcnn=False)
    cap = cv2.VideoCapture(video_path)
    emotion_counts = {e: 0 for e in ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]}
    frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        emotion = detector.top_emotion(frame)
        if emotion:
            dominant = emotion[0]
            if dominant in emotion_counts:
                emotion_counts[dominant] += 1
    cap.release()

    return {e: round((cnt / frames) * 100, 2) if frames > 0 else 0.0 for e, cnt in emotion_counts.items()}

# Extract audio-based features (pitch, energy, tempo)
def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path)

        # ZCR returns 2D array [[val1, val2, ...]]
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
        rms = float(np.mean(librosa.feature.rms(y=y)[0]))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return {
            "zcr": round(zcr, 4),
            "rms": round(rms, 4),
            "tempo": round(float(tempo), 2)
        }
    except Exception as e:
        print(f"[AUDIO] Error extracting audio features: {e}")
        return {"zcr": 0.0, "rms": 0.0, "tempo": 0.0}

# Combine text, video, and audio into final Big Five scores
def simulate_big_five_scores(video_path):
    from video_process.video_utils import process_video

    print(f"[PERSONALITY] Analyzing Big Five traits for video: {video_path}")

    # Step 1: Transcribe text
    text = process_video(video_path, "temp", "upload") or ""
    text_traits = predict_personality_from_text(text)

    # Step 2: Extract emotion from video
    video_emotions = analyze_video_emotions(video_path)

    # Step 3: Extract audio features from generated audio
    audio_path = os.path.join("upload", "temp.wav")
    audio_features = extract_audio_features(audio_path)

    # Step 4: Fuse traits
    fused_traits = text_traits.copy()

    fused_traits["Extraversion"] = round(
        0.5 * text_traits["Extraversion"] +
        0.25 * ((video_emotions["happy"] + video_emotions["neutral"]) / 2) +
        0.25 * (audio_features["rms"] * 100), 1)

    fused_traits["Agreeableness"] = round(
        0.6 * text_traits["Agreeableness"] +
        0.4 * (100 - video_emotions["angry"]), 1)

    fused_traits["Neuroticism"] = round(
        0.5 * text_traits["Neuroticism"] +
        0.25 * ((video_emotions["sad"] + video_emotions["fear"]) / 2) +
        0.25 * (100 - audio_features["tempo"]), 1)

    fused_traits["Openness"] = round(
        0.6 * text_traits["Openness"] +
        0.4 * ((video_emotions["surprise"] + video_emotions["happy"]) / 2), 1)

    fused_traits["Conscientiousness"] = round(text_traits["Conscientiousness"], 1)

    print(f"[PERSONALITY] Fused Traits: {fused_traits}")
    return fused_traits

# Role-fit scoring
def score_roles(traits):
    scores = {
        'software_engineer': traits["Conscientiousness"] * 0.4 +
                             traits["Openness"] * 0.3 +
                             (100 - traits["Neuroticism"]) * 0.3,

        'associate_software_engineer': traits["Conscientiousness"] * 0.3 +
                                       traits["Agreeableness"] * 0.2 +
                                       traits["Openness"] * 0.2 +
                                       (100 - traits["Neuroticism"]) * 0.3,

        'it_intern': traits["Openness"] * 0.4 +
                     traits["Extraversion"] * 0.2 +
                     traits["Agreeableness"] * 0.1 +
                     (100 - traits["Neuroticism"]) * 0.3,
    }

    return {role: int(round(score)) for role, score in scores.items()}

# Averaging traits across multiple videos
def average_traits(traits_list):
    avg = {}
    total = len(traits_list)
    for trait in traits_list[0].keys():
        avg[trait] = round(sum(t[trait] for t in traits_list) / total, 2)
    return avg
