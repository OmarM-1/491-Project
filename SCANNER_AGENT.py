import re
from Spotter_AI import chat_text  # Only needed if you're testing with live model output

def audit_response(response: str, user_goal: str) -> dict:
    """
    Checks an agent response for clarity, vagueness, and alignment with user goals.
    """
    issues = []
    text = response.lower()

    if "eat healthy" in text and not any(nutrient in text for nutrient in ["protein", "fiber", "carbs", "fats"]):
        issues.append("Vague nutrition advice. It lacks specific nutrients or examples.")

    if "exercise" in text and not any(term in text for term in ["sets", "reps", "duration", "intensity", "rest"]):
        issues.append("Vague exercise advice. It's missing structure or measurable actions.")

    if "calories" in text and user_goal.lower() not in text:
        issues.append("Calorie advice may not align with user goal.")

    if not any(ref in text for ref in ["mifflin", "katch", "source", "based on", "according to"]):
        issues.append("Missing reference to method, formula, or source.")

    if not re.search(r"\b(you|your)\b", text):
        issues.append("Response may lack direct user engagement as it's missing 'you' or 'your'.")

    if len(response.strip().split()) < 15:
        issues.append("Response may be too short to be informative.")

    return {
        "issues": issues,
        "is_valid": len(issues) == 0,
        "score": max(0, 10 - len(issues) * 2),
        "summary": "Pass" if len(issues) == 0 else "Needs revision"
    }