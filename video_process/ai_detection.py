# ai_detection.py
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

class AIDetector:
    def __init__(self, model_path="models/ai_text_detector_roberta"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def detect_text(self, text: str):
        if not text.strip():
            return "Unknown", 0.0

        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_label = int(probs.argmax())
        confidence = probs[predicted_label]

        label_name = "AI" if predicted_label == 1 else "Human"
        return label_name, confidence
