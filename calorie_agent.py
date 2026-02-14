# Calorie Agent — FastAPI service
# Author: Saketh Kakarla
# License: MIT
#
# Run locally:
#   python -m uvicorn main:app --reload
#
# Interactive docs:
#   http://127.0.0.1:8000/docs

from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator  

app = FastAPI(
    title="Calorie Agent",
    description=(
        "Simple API to estimate BMR, TDEE, and target calories to lose/maintain/gain weight. "
        "Uses Mifflin–St Jeor by default or Katch–McArdle if body fat % is provided."
    ),
    version="1.0.0",
)



def pound_to_kg(pounds: float) -> float:
    return pounds * 0.45359237

def inches_to_cm(inches: float) -> float:
    return inches * 2.54

def kcal_for_kg_fat(kg: float) -> float:
    # ~7700 kcal per kg of fat mass (heuristic)
    return kg * 7700.0

def kcal_for_lb_fat(lb: float) -> float:
    # ~3500 kcal per lb of fat mass (heuristic)
    return lb * 3500.0

ACTIVITY_FACTORS_DICT = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

Sex = Literal["male", "female"]



class CalculateInput(BaseModel):
    sex: Sex = Field(..., description="Biological sex for formula selection ('male' or 'female').")
    age: int = Field(..., ge=10, le=100, description="Age in years (10–100).")
    # Provide either metric or imperial for height/weight. We'll normalize to kg, cm.
    height_cm: Optional[float] = Field(None, gt=0, description="Height in centimeters.")
    height_in: Optional[float] = Field(None, gt=0, description="Height in inches (if not using cm).")
    weight_kg: Optional[float] = Field(None, gt=0, description="Weight in kilograms.")
    weight_lb: Optional[float] = Field(None, gt=0, description="Weight in pounds (if not using kg).")
    activity_level: Literal["sedentary", "light", "moderate", "active", "very_active"] = Field(
        "moderate", description="Estimated daily activity level multiplier."
    )
    body_fat_percent: Optional[float] = Field(
        None, ge=2, le=75, description="If provided, uses Katch–McArdle (requires body fat %)."
    )

    # Goal handling
    goal: Literal["lose", "maintain", "gain"] = Field("maintain", description="Desired goal.")
    
    weekly_rate_kg: Optional[float] = Field(None, gt=0, description="Target change per week in kg.")
    weekly_rate_lb: Optional[float] = Field(None, gt=0, description="Target change per week in lb.")

    @field_validator("height_cm", "height_in", "weight_kg", "weight_lb")
    @classmethod
    def positive_vals(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Values must be > 0")
        return v

    @field_validator("activity_level")
    @classmethod
    def valid_activity(cls, v):
        if v not in ACTIVITY_FACTORS_DICT:
            raise ValueError(f"activity_level must be one of {list(ACTIVITY_FACTORS_DICT)}")
        return v

    @field_validator("weekly_rate_kg", "weekly_rate_lb")
    @classmethod
    def valid_rate(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Weekly rates must be positive numbers.")
        return v

    def normalized(self):
        # Ensure we have height_cm & weight_kg
        if self.height_cm is None and self.height_in is None:
            raise HTTPException(status_code=422, detail="Provide height_cm or height_in.")
        if self.weight_kg is None and self.weight_lb is None:
            raise HTTPException(status_code=422, detail="Provide weight_kg or weight_lb.")

        # Use the helpers you defined (names fixed here)
        height_cm = self.height_cm if self.height_cm is not None else inches_to_cm(self.height_in)  
        weight_kg = self.weight_kg if self.weight_kg is not None else pound_to_kg(self.weight_lb)  

        # Weekly rate (kg/week)
        rate_kg = None
        if self.weekly_rate_kg is not None:
            rate_kg = self.weekly_rate_kg
        elif self.weekly_rate_lb is not None:
            rate_kg = self.weekly_rate_lb * 0.45359237

        return {
            "sex": self.sex,
            "age": self.age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "activity_level": self.activity_level,
            "body_fat_percent": self.body_fat_percent,
            "goal": self.goal,
            "weekly_rate_kg": rate_kg,
        }

class CalculateOutput(BaseModel):
    method: Literal["Mifflin-St Jeor", "Katch-McArdle"]
    bmr: int
    tdee: int
    activity_factor: float
    maintain_calories: int
    suggestions: dict
    notes: str



def bmr_mifflin_st_jeor(sex: Sex, weight_kg: float, height_cm: float, age: int) -> float:
    # Male:   BMR = 10*W + 6.25*H - 5*A + 5
    # Female: BMR = 10*W + 6.25*H - 5*A - 161
    base = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age
    return base + (5.0 if sex == "male" else -161.0)

def bmr_katch_mcardle(weight_kg: float, body_fat_percent: float) -> float:
    lbm_kg = weight_kg * (1.0 - body_fat_percent / 100.0)
    return 370.0 + 21.6 * lbm_kg

def tdee_from_activity(bmr: float, activity_level: str) -> float:
    return bmr * ACTIVITY_FACTORS_DICT[activity_level]



@app.post("/calculate", response_model=CalculateOutput, tags=["Calculate"])
def calculate(payload: CalculateInput):
    data = payload.normalized()

    # Choose method
    if data["body_fat_percent"] is not None:
        method = "Katch-McArdle"
        bmr = bmr_katch_mcardle(data["weight_kg"], data["body_fat_percent"])  
    else:
        method = "Mifflin-St Jeor"
        bmr = bmr_mifflin_st_jeor(data["sex"], data["weight_kg"], data["height_cm"], data["age"])  

    activity_factor = ACTIVITY_FACTORS_DICT[data["activity_level"]]
    tdee = tdee_from_activity(bmr, data["activity_level"])
    maintain = round(tdee)

    suggestions = {
        "maintain": maintain,
        "cut_mild_-0.25kg_wk": max(1200, round(maintain - kcal_for_kg_fat(0.25) / 7.0)),
        "cut_moderate_-0.5kg_wk": max(1200, round(maintain - kcal_for_kg_fat(0.5) / 7.0)),
        "cut_aggressive_-0.75kg_wk": max(1200, round(maintain - kcal_for_kg_fat(0.75) / 7.0)),
        "gain_slow_+0.25kg_wk": round(maintain + kcal_for_kg_fat(0.25) / 7.0),
        "gain_moderate_+0.5kg_wk": round(maintain + kcal_for_kg_fat(0.5) / 7.0),
    }

    notes = "Presets shown use ~7700 kcal per kg of fat as a heuristic. Actual needs vary."
    if data["weekly_rate_kg"] is not None and data["goal"] != "maintain":
        daily_delta = kcal_for_kg_fat(abs(data["weekly_rate_kg"])) / 7.0
        if data["goal"] == "lose":
            suggestions["custom_target"] = max(1200, round(maintain - daily_delta))
            notes += f" Custom target set for {data['goal']} at ~{data['weekly_rate_kg']:.2f} kg/week."
        elif data["goal"] == "gain":
            suggestions["custom_target"] = round(maintain + daily_delta)
            notes += f" Custom target set for {data['goal']} at ~{data['weekly_rate_kg']:.2f} kg/week."

    return CalculateOutput(
        method=method,
        bmr=round(bmr),
        tdee=round(tdee),
        activity_factor=activity_factor,
        maintain_calories=maintain,
        suggestions=suggestions,
        notes=notes,
    )

# ---------- Root ----------

@app.get("/", tags=["Help"])
def root():
    return {
        "message": "Welcome to Calorie Agent. POST to /calculate with JSON body. See /docs for interactive UI.",
        "example_input": {
            "sex": "male",
            "age": 25,
            "height_cm": 178,
            "weight_kg": 80,
            "activity_level": "moderate",
            "goal": "lose",
            "weekly_rate_kg": 0.5
        }
    }






