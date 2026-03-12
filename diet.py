
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

Goal = Literal["lose", "maintain", "gain"]

@dataclass
class DietSuggestions:
    target_calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    guidelines: List[str]
    meal_templates: List[str]
    snack_options: List[str]

class DietAgent:
    """
    Simple, beginner-friendly diet suggestion generator.
    Not medical advice. Designed for gym chatbot flows.
    """

    def _protein_target(self, weight_kg: float, goal: Goal) -> float:
        # Slightly higher for fat loss / recomposition adherence
        if goal == "lose":
            return 2.0 * weight_kg
        if goal == "gain":
            return 1.8 * weight_kg
        return 1.6 * weight_kg

    def _macro_split(self, calories: int, protein_g: int, goal: Goal) -> tuple[int, int]:
        """
        Returns (fat_g, carbs_g) with protein fixed.
        """
        # Fat: 25-30% calories (slightly higher on cut for satiety)
        fat_pct = 0.30 if goal == "lose" else 0.25
        fat_g = int((calories * fat_pct) / 9)

        # Remaining calories -> carbs
        remaining = calories - (protein_g * 4) - (fat_g * 9)
        carbs_g = max(0, int(remaining / 4))
        return fat_g, carbs_g

    def suggest(
        self,
        goal: Goal,
        target_calories: int,
        weight_kg: float,
        dietary_style: Optional[str] = None,  # e.g., "halal", "vegetarian", "no_pork"
    ) -> Dict:
        protein_g = int(round(self._protein_target(weight_kg, goal)))
        fat_g, carbs_g = self._macro_split(target_calories, protein_g, goal)

        guidelines = [
            f"Hit ~{protein_g}g protein/day (helps muscle retention and fullness).",
            "Aim for 25–35g fiber/day (fruits, veggies, beans, whole grains).",
            "Drink water consistently; keep sugary drinks rare.",
            "80/20 rule: mostly whole foods, some flexibility.",
        ]

        if goal == "lose":
            guidelines += [
                "Keep most carbs around workouts if you train hard.",
                "Prioritize lean proteins + high-volume foods (veg, fruit, soups).",
            ]
        elif goal == "gain":
            guidelines += [
                "Add calories gradually (don’t jump too high).",
                "Include easy calories if needed: rice, pasta, olive oil, nuts.",
            ]

        # Meal templates (simple, not recipe-heavy)
        meal_templates = [
            "Breakfast: eggs/egg whites + oatmeal + fruit",
            "Lunch: chicken/turkey + rice/potatoes + veggies",
            "Dinner: salmon/lean beef + carbs + big salad/greens",
        ]

        snack_options = [
            "Greek yogurt + berries",
            "Protein shake + banana",
            "Cottage cheese + fruit",
            "Protein bar (watch added sugar)",
            "Nuts (portion-controlled)",
        ]

        # Optional style tweaks
        if dietary_style:
            ds = dietary_style.lower()
            if "vegetarian" in ds:
                meal_templates = [
                    "Breakfast: Greek yogurt (or soy) + oats + fruit",
                    "Lunch: tofu/tempeh + rice + veggies",
                    "Dinner: lentils/beans + potatoes + salad",
                ]
                snack_options = [
                    "Greek yogurt or soy yogurt",
                    "Protein shake (plant-based)",
                    "Roasted chickpeas",
                    "Cottage cheese (if allowed)",
                    "Nuts (portion-controlled)",
                ]

        return DietSuggestions(
            target_calories=target_calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
            guidelines=guidelines,
            meal_templates=meal_templates,
            snack_options=snack_options,
        ).__dict__

