import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from video_process.video_utils import process_video
from video_process.personality import simulate_big_five_scores, score_roles, average_traits
from video_process.answer_analyzer import AnswerAnalyzer
from video_process.ai_detection import AIDetector
from video_process.ml_eye_detector import predict_eye_behavior

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/convert", methods=["POST"])
def convert_video_to_text():
    try:
        video = request.files.get("video")
        if not video:
            return jsonify({"error": "No video file uploaded"}), 400

        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        # Call the existing transcription logic
        from video_process.video_utils import process_video
        transcribed_text = process_video(video_path, "temp", UPLOAD_FOLDER)

        return jsonify({
            "filename": video.filename,
            "transcription": transcribed_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/evaluate", methods=["POST"])
def evaluate_candidate():
    try:
        videos = request.files.getlist("videos")
        applied_role = request.form.get("applied_role", "").lower()
        candidate_id = request.form.get("candidate_id", "unknown")

        if not videos:
            return jsonify({"error": "No video files uploaded"}), 400

        personality_results = []
        eye_scores = []
        eye_labels = []
        ai_detection_results = []
        video_analysis = []

        for video in videos:
            video_path = os.path.join(UPLOAD_FOLDER, video.filename)
            video.save(video_path)

            # 1. Eye Tracking
            eye_label, eye_confidence, is_attentive = predict_eye_behavior(video_path)
            eye_scores.append(eye_confidence)
            eye_labels.append(eye_label)

            # 2. Personality Analysis
            traits = simulate_big_five_scores(video_path)
            personality_results.append(traits)

            # 3. AI-Generated Answer Detection
            transcript_text = process_video(video_path, "temp", "upload")
            detector = AIDetector()
            label, ai_prob = detector.detect_text(transcript_text)
            ai_detection_results.append({
                "video": video.filename,
                "ai_generated": label,
                "ai_probability": round(ai_prob * 100, 2)
            })

            # 4. Append per-video breakdown
            video_analysis.append({
                "video": video.filename,
                "eye_tracking_label": eye_label,
                "eye_tracking_score": eye_confidence,
                "personality_traits": traits
            })

        # Aggregate Results
        avg_traits = average_traits(personality_results)
        avg_eye_score = round(sum(eye_scores) / len(eye_scores), 2)
        analyzer = AnswerAnalyzer()
        final_score = analyzer.calculate_final_score(avg_traits, avg_eye_score)

        # Role Fit Scoring
        role_fit = score_roles(avg_traits)
        applied_role_score = role_fit.get(applied_role, "N/A")

        # Basic Role Fit Explanation
        explanation = "Candidate shows good potential." if isinstance(applied_role_score, int) and applied_role_score >= 60 \
            else "Candidate might need development in this role." if isinstance(applied_role_score, int) else "Role not recognized."

        # Candidate suitability from average eye contact score
        candidate_suitability = "Suitable" if avg_eye_score >= 60 else "Not Suitable"

        return jsonify({
            "candidate_id": candidate_id,
            "applied_role": applied_role,
            "applied_role_score": applied_role_score,
            "applied_role_explanation": explanation,
            "candidate_suitability": candidate_suitability,
            "eye_tracking_label": eye_labels[0] if eye_labels else "undetected",
            "eye_tracking_score": avg_eye_score,
            "personality_traits": avg_traits,
            "final_score": final_score,
            "role_fit_scores": role_fit,
            "ai_detection": ai_detection_results,
            "video_analysis": video_analysis
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Candidate Evaluation API is running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
