# rag_chat.py
# Terminal Chat service
# Run: python -m uvicorn rag_chat:app --reload --port 8001

import re
from typing import Literal, Optional, Dict, Any

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from safety_agent import safety_gate_agent  
from diet import DietAgent

app = FastAPI(
    title="Terminal Chat",
    description="Routes calorie questions to Calorie Agent, then optionally provides diet suggestions.",
    version="2.0.0",
)

# ----------------------------
# Simple session memory
# ----------------------------
SESSION: Dict[str, Dict[str, Any]] = {}
diet_agent = DietAgent()

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSION:
        SESSION[session_id] = {}
    return SESSION[session_id]


# ----------------------------
# Intent routing + parsing
# ----------------------------
Route = Literal["calorie_calc", "diet_followup", "other"]

def route_intent(text: str, session: Dict[str, Any]) -> Route:
    t = text.lower().strip()

    
    if session.get("awaiting_diet_opt_in"):
        if t in {"yes", "y", "yeah", "yep", "sure", "ok", "okay"}:
            return "diet_followup"
        if t in {"no", "n", "nope", "nah"}:
            return "other"
        
        return "diet_followup"

    looks_like_stats = (
        bool(re.search(r"\b(male|female)\b", t))
        and bool(re.search(r"\b\d{2}\b", t))
        and (
            bool(re.search(r"\b(\d)\s*(ft|feet|')\s*(\d{1,2})?\b", t)) or ("cm" in t)
        )
        and (("lb" in t) or ("lbs" in t) or ("pounds" in t) or ("kg" in t))
    )

    keywords = ["calorie", "calories", "bmr", "tdee", "maintenance", "maintain", "cut", "bulk", "lose", "gain"]
    mentions_calories = any(k in t for k in keywords)

    if looks_like_stats or mentions_calories:
        return "calorie_calc"
    return "other"


def parse_height_weight_age_sex(text: str):
    t = text.lower()

    # sex
    sex = "male" if "male" in t else ("female" if "female" in t else None)

    # age
    age = None
    m = re.search(r"\b(\d{2})\s*(years old|yo|y/o)?\b", t)
    if m:
        age = int(m.group(1))

    # height
    height_in = None
    m = re.search(r"\b(\d)\s*(ft|feet|')\s*(\d{1,2})?\b", t)
    if m:
        ft = int(m.group(1))
        inch = int(m.group(3)) if m.group(3) else 0
        height_in = ft * 12 + inch

    height_cm = None
    m = re.search(r"\b(\d{2,3})\s*cm\b", t)
    if m:
        height_cm = float(m.group(1))

    # weight
    weight_lb = None
    m = re.search(r"\b(\d{2,3}(\.\d+)?)\s*(lb|lbs|pounds)\b", t)
    if m:
        weight_lb = float(m.group(1))

    weight_kg = None
    m = re.search(r"\b(\d{2,3}(\.\d+)?)\s*kg\b", t)
    if m:
        weight_kg = float(m.group(1))

    # activity level (optional)
    activity_level = "moderate"
    if "sedentary" in t:
        activity_level = "sedentary"
    elif "light" in t:
        activity_level = "light"
    elif "very active" in t:
        activity_level = "very_active"
    elif "active" in t:
        activity_level = "active"

    # goal (optional)
    goal = "maintain"
    if "lose" in t or "cut" in t or "fat loss" in t:
        goal = "lose"
    elif "gain" in t or "bulk" in t or "muscle" in t:
        goal = "gain"

    return {
        "sex": sex,
        "age": age,
        "height_cm": height_cm,
        "height_in": height_in,
        "weight_kg": weight_kg,
        "weight_lb": weight_lb,
        "activity_level": activity_level,
        "body_fat_percent": None,
        "goal": goal,
        "weekly_rate_kg": None,
        "weekly_rate_lb": None,
    }


# ----------------------------
# API models
# ----------------------------
class ChatIn(BaseModel):
    message: str
    session_id: str = "default"  # client can pass unique ID per user

class ChatOut(BaseModel):
    route: str
    answer: str


# ----------------------------
# Calorie Agent integration
# ----------------------------
CALORIE_AGENT_URL = "http://127.0.0.1:8000/calculate"

def call_calorie_agent(extracted: dict) -> dict:
    r = requests.post(CALORIE_AGENT_URL, json=extracted, timeout=10)
    r.raise_for_status()
    return r.json()


@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    session = get_session(payload.session_id)

    is_safe, warning = safety_gate_agent(payload.message)
    if not is_safe:
        return ChatOut(route="safety_block", answer=warning)

    route = route_intent(payload.message, session)

    # ----------------------------
    # Diet follow-up flow
    # ----------------------------
    if route == "diet_followup":
        # If user said "no"
        if payload.message.lower().strip() in {"no", "n", "nope", "nah"}:
            session["awaiting_diet_opt_in"] = False
            return ChatOut(route="diet_followup", answer="No problem — want workout suggestions next?")

        # If user says yes but we don’t have a calorie result stored, fallback
        last = session.get("last_calorie_result")
        last_extracted = session.get("last_extracted")
        if not last or not last_extracted:
            session["awaiting_diet_opt_in"] = False
            return ChatOut(
                route="diet_followup",
                answer="I can do that — first tell me your sex, age, height, weight, and activity level so I can compute your calorie target.",
            )

        session["awaiting_diet_opt_in"] = False

        # Pick the most relevant target based on goal
        goal = last_extracted.get("goal", "maintain")
        suggestions = last.get("suggestions", {})
        if goal == "lose":
            # default to moderate cut (you can change this)
            target = suggestions.get("cut_moderate_-0.5kg_wk", last["maintain_calories"])
        elif goal == "gain":
            target = suggestions.get("gain_slow_+0.25kg_wk", last["maintain_calories"])
        else:
            target = last["maintain_calories"]

        # We need weight_kg for protein target; calorie_agent normalized but doesn't return weight.
        # Use extracted weight (kg or convert from lb).
        weight_kg = last_extracted.get("weight_kg")
        if weight_kg is None and last_extracted.get("weight_lb") is not None:
            weight_kg = last_extracted["weight_lb"] * 0.45359237

        if weight_kg is None:
            weight_kg = 75.0  # safe fallback; better than crashing

        diet = diet_agent.suggest(goal=goal, target_calories=int(target), weight_kg=float(weight_kg))

        answer = (
            f"Diet suggestions for goal **{goal}**:\n"
            f"- Target calories: **{diet['target_calories']} kcal/day**\n"
            f"- Macros (approx): **P {diet['protein_g']}g / C {diet['carbs_g']}g / F {diet['fat_g']}g**\n\n"
            f"Guidelines:\n- " + "\n- ".join(diet["guidelines"]) + "\n\n"
            f"Simple meal templates:\n- " + "\n- ".join(diet["meal_templates"]) + "\n\n"
            f"Snack ideas:\n- " + "\n- ".join(diet["snack_options"])
        )
        return ChatOut(route="diet_followup", answer=answer)

    # ----------------------------
    # Default / other
    # ----------------------------
    if route == "other":
        # If we were awaiting yes/no, keep prompting clearly
        if session.get("awaiting_diet_opt_in"):
            return ChatOut(route="diet_followup", answer="Just reply yes/no — do you want diet suggestions?")
        return ChatOut(
            route=route,
            answer="Tell me your sex, age, height, weight, and activity level (and say lose/maintain/gain). Example: 'male 25 5'10 175 lb moderate I want to lose weight'."
        )

    # ----------------------------
    # Calorie flow
    # ----------------------------
    extracted = parse_height_weight_age_sex(payload.message)

    # Basic validation before calling the calorie agent
    has_height = extracted.get("height_cm") is not None or extracted.get("height_in") is not None
    has_weight = extracted.get("weight_kg") is not None or extracted.get("weight_lb") is not None
    if not (extracted.get("sex") and extracted.get("age") and has_height and has_weight):
        return ChatOut(
            route=route,
            answer="I need your age, sex, height, and weight. Example: 'male 25 5'10 175 lb moderate lose'."
        )

    try:
        result = call_calorie_agent(extracted)

        # Save state for next message
        session["last_calorie_result"] = result
        session["last_extracted"] = extracted
        session["awaiting_diet_opt_in"] = True

        return ChatOut(
            route=route,
            answer=(
                f"BMR: {result['bmr']}\n"
                f"TDEE: {result['tdee']}\n"
                f"Maintain: {result['maintain_calories']}\n"
                f"Notes: {result['notes']}\n\n"
                f"Do you want diet suggestions based on this? (yes/no)"
            ),
        )
    except Exception:
        return ChatOut(
            route=route,
            answer="Could not reach the Calorie Agent. Make sure it's running on port 8000."
        )

