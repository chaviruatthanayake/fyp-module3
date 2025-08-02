class AnswerAnalyzer:
    def __init__(self):
        pass  # No keyword/content checking required.

    def calculate_final_score(self, personality_scores, eye_score):
        """
        Calculate the final candidate score based on personality and eye-tracking only.

        - personality_scores: dict of Big Five traits (0-100%)
        - eye_score: eye tracking score (0-100)
        """
        personality_avg = sum(personality_scores.values()) / len(personality_scores)
        final_score = 0.7 * personality_avg + 0.3 * eye_score

        return round(min(max(final_score, 0), 100), 2)
