# nutrition_tools.py
from typing import Optional, Literal, Dict, Any
import requests  # if you keep Calorie Agent as a FastAPI microservice

from diet_agent import DietAgent, Goal

Sex = Literal["male", "female"]

def call_calorie_agent(
    sex: Sex,
    age: int,
    height_cm: Optional[float] = None,
    height_in: Optional[float] = None,
    weight_kg: Optional[float] = None,
    weight_lb: Optional[float] = None,
    activity_level: str = "moderate",
    goal: str = "maintain",
    weekly_rate_kg: Optional[float] = None,
    weekly_rate_lb: Optional[float] = None,
    base_url: str = "http://127.0.0.1:8000"
) -> Dict[str, Any]:
    """
    Thin client for the Calorie Agent FastAPI /calculate endpoint.
    """
    payload = {
        "sex": sex,
        "age": age,
        "height_cm": height_cm,
        "height_in": height_in,
        "weight_kg": weight_kg,
        "weight_lb": weight_lb,
        "activity_level": activity_level,
        "goal": goal,
        "weekly_rate_kg": weekly_rate_kg,
        "weekly_rate_lb": weekly_rate_lb,
    }
    # Strip out None fields to avoid validation noise
    payload = {k: v for k, v in payload.items() if v is not None}

    resp = requests.post(f"{base_url}/calculate", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def format_calorie_response(data: Dict[str, Any]) -> str:
    """
    Turn Calorie Agent JSON into a user-facing string.
    """
    method = data.get("method")
    bmr = data.get("bmr")
    tdee = data.get("tdee")
    maintain = data.get("maintain_calories")
    suggestions = data.get("suggestions", {})
    notes = data.get("notes", "")

    lines = [
        f"I estimated your calories using {method}.",
        f"• BMR: {bmr} kcal/day",
        f"• TDEE (maintenance): ~{tdee} kcal/day",
        f"• Suggested maintenance intake: ~{maintain} kcal/day",
    ]

    # Highlight a couple of presets if present
    if "cut_moderate_-0.5kg_wk" in suggestions:
        lines.append(
            f"• Moderate fat loss: ~{suggestions['cut_moderate_-0.5kg_wk']} kcal/day"
        )
    if "gain_moderate_+0.5kg_wk" in suggestions:
        lines.append(
            f"• Moderate gain: ~{suggestions['gain_moderate_+0.5kg_wk']} kcal/day"
        )

    if "custom_target" in suggestions:
        lines.append(f"• Custom target: ~{suggestions['custom_target']} kcal/day")

    if notes:
        lines.append("")
        lines.append(notes)

    return "\n".join(lines)

def call_diet_agent(
    goal: Goal,
    target_calories: int,
    weight_kg: float,
    dietary_style: Optional[str] = None,
) -> str:
    """
    Wrap DietAgent.suggest and format into readable answer.
    """
    agent = DietAgent()
    result = agent.suggest(
        goal=goal,
        target_calories=target_calories,
        weight_kg=weight_kg,
        dietary_style=dietary_style,
    )

    lines = [
        f"Here's a simple diet outline around ~{result['target_calories']} kcal/day:",
        f"• Protein: ~{result['protein_g']} g",
        f"• Carbs: ~{result['carbs_g']} g",
        f"• Fat: ~{result['fat_g']} g",
        "",
        "Guidelines:",
    ]
    for g in result["guidelines"]:
        lines.append(f"- {g}")
    lines.append("")
    lines.append("Example meals:")
    for m in result["meal_templates"]:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("Snack ideas:")
    for s in result["snack_options"]:
        lines.append(f"- {s}")

    return "\n".join(lines)
