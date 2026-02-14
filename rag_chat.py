# Terminal Chat service
# Run: python -m uvicorn rag_chat:app --reload --port 8001

import re
from typing import Literal

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from safety_agent import safety_gate_agent

app = FastAPI(
    title="Terminal Chat",
    description="Routes calorie questions to Calorie Agent. No KB/RAG.",
    version="1.0.0",
)

# ----------------------------
# Intent routing + parsing
# ----------------------------
def route_intent(text: str) -> Literal["calorie_calc", "other"]:
    t = text.lower()

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

    return {
        "sex": sex,
        "age": age,
        "height_cm": height_cm,
        "height_in": height_in,
        "weight_kg": weight_kg,
        "weight_lb": weight_lb,
        "activity_level": activity_level,
        "body_fat_percent": None,
        "goal": "maintain",
        "weekly_rate_kg": None,
        "weekly_rate_lb": None,
    }


# ----------------------------
# API models
# ----------------------------
class ChatIn(BaseModel):
    message: str

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
    is_safe, warning = safety_gate_agent(payload.message)
    if not is_safe:
        return ChatOut(route="safety_block", answer = warning)
    route = route_intent(payload.message)

    if route == "other":
        return ChatOut(
            route=route,
            answer="I can only help with calorie/BMR/TDEE calculations right now. Tell me your sex, age, height, weight, and activity level."
        )

    extracted = parse_height_weight_age_sex(payload.message)

    # Basic validation before calling the calorie agent
    has_height = extracted.get("height_cm") is not None or extracted.get("height_in") is not None
    has_weight = extracted.get("weight_kg") is not None or extracted.get("weight_lb") is not None
    if not (extracted.get("sex") and extracted.get("age") and has_height and has_weight):
        return ChatOut(
            route=route,
            answer="I need your age, sex, height, and weight. Example: 'male 25 5'10 175 lb moderate'."
        )

    try:
        result = call_calorie_agent(extracted)
        return ChatOut(
            route=route,
            answer=(
                f"BMR: {result['bmr']}\n"
                f"TDEE: {result['tdee']}\n"
                f"Maintain: {result['maintain_calories']}\n"
                f"Notes: {result['notes']}"
            ),
        )
    except Exception:
        return ChatOut(
            route=route,
            answer="Could not reach the Calorie Agent. Make sure it's running on port 8000."
        )

