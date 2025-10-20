# Calorie Agent (FastAPI)

A tiny API that estimates **BMR**, **TDEE**, and target calories to **lose / maintain / gain** weight — like the calculators on fitness sites, but open-source and ready for your gym chatbot.

- **Formulas:** Uses **Mifflin–St Jeor** by default; switches to **Katch–McArdle** if you provide body fat %.
- **Units:** Accepts metric or imperial (cm/inches, kg/pounds).
- **Goals:** Presets for mild/moderate/aggressive cut or slow/moderate bulk. Optional custom **weekly rate**.
- **Docs:** Auto-generated Swagger UI at `/docs`.



## Quickstart

```bash
# 1) Create & activate a virtual env (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API.

## Example (curl)

```bash
curl -X POST "http://127.0.0.1:8000/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "sex": "male",
    "age": 28,
    "height_in": 70,
    "weight_lb": 180,
    "activity_level": "moderate",
    "goal": "lose",
    "weekly_rate_lb": 1.0
  }'
```

**Response**

```json
{
  "method": "Mifflin-St Jeor",
  "bmr": 1772,
  "tdee": 2747,
  "activity_factor": 1.55,
  "maintain_calories": 2747,
  "suggestions": {
    "maintain": 2747,
    "cut_mild_-0.25kg_wk": 2476,
    "cut_moderate_-0.5kg_wk": 2204,
    "cut_aggressive_-0.75kg_wk": 1933,
    "gain_slow_+0.25kg_wk": 3018,
    "gain_moderate_+0.5kg_wk": 3290,
    "custom_target": 2247
  },
  "notes": "Presets shown use ~7700 kcal per kg of fat as a heuristic. Actual needs vary. Custom target set for lose at ~0.45 kg/week."
}
```

## Inputs

| Field | Type | Notes |
|---|---|---|
| `sex` | `"male" \| "female"` | Required |
| `age` | int | 10–100 |
| `height_cm` or `height_in` | float | One is required |
| `weight_kg` or `weight_lb` | float | One is required |
| `activity_level` | `"sedentary" \| "light" \| "moderate" \| "active" \| "very_active"` | Default `"moderate"` |
| `body_fat_percent` | float | Optional; switches to Katch–McArdle |
| `goal` | `"lose" \| "maintain" \| "gain"` | Default `"maintain"` |
| `weekly_rate_kg` or `weekly_rate_lb` | float | Optional custom weekly rate |

### Activity multipliers (heuristics)

- sedentary: **1.2**
- light: **1.375**
- moderate: **1.55**
- active: **1.725**
- very_active: **1.9**

## How it works

- **BMR (Mifflin–St Jeor)**  
  - Male: `10*W + 6.25*H - 5*A + 5`  
  - Female: `10*W + 6.25*H - 5*A - 161`
- **BMR (Katch–McArdle)** when `body_fat_percent` provided  
  - `370 + 21.6 * LBM_kg` where `LBM = weight_kg * (1 - bf%)`
- **TDEE** = `BMR * activity_factor`
- **Targets**: daily deltas from a weekly rate using ~**7700 kcal/kg** (≈**3500 kcal/lb**).

## Use in a Gym Chatbot

Call the `/calculate` endpoint from your bot when the user shares their stats/goals and surface the `suggestions` back to them.

Example pseudo-flow:
1. Ask: sex, age, height, weight, activity level, goal (and optionally weekly rate or body fat %).
2. POST to `/calculate`.
3. Present the `maintain` and relevant `cut_`/`gain_` suggestion as target calories, with a health disclaimer.

## Repo structure

```
calorie-agent/
├─ main.py
├─ requirements.txt
└─ README.md
```

## License

MIT — see `LICENSE`.
