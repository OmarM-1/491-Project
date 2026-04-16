# agentic_rag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re
import math

from optimized_rag import get_rag, retrieve, generate_grounded_answer

@dataclass
class AgenticRAGAgent:
    rag: object
    verbose: bool = False

_AGENT: Optional[AgenticRAGAgent] = None

def get_agent(verbose: bool = False) -> AgenticRAGAgent:
    """Hybrid orchestrator imports this."""
    global _AGENT
    if _AGENT is None:
        _AGENT = AgenticRAGAgent(rag=get_rag(), verbose=verbose)
    return _AGENT

def _decompose(query: str) -> List[str]:
    """Split multi-part questions into sub-questions."""
    parts = re.split(r"\b(?:and|then|also|plus)\b", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [query]

# ─────────────────────────────────────────────────────────────
# Calculator Tool
# ─────────────────────────────────────────────────────────────

def _calc_tool(query: str) -> List[str]:
    """
    Detects fitness math in the query and returns computed fact strings
    to inject into the generation context. Returns empty list if no math found.
    """
    q = query.lower()
    results = []

    # ── BMI ───────────────────────────────────────────────────
    m = re.search(r'(\d+\.?\d*)\s*kg.*?(\d+\.?\d*)\s*(?:m\b|meter)', q)
    if 'bmi' in q and m:
        weight_kg, height_m = float(m.group(1)), float(m.group(2))
        bmi = round(weight_kg / (height_m ** 2), 1)
        category = (
            "Underweight" if bmi < 18.5 else
            "Normal weight" if bmi < 25 else
            "Overweight" if bmi < 30 else "Obese"
        )
        results.append(f"[CALC] BMI = {bmi} ({category})")

    # ── Pounds/kg BMI ─────────────────────────────────────────
    m = re.search(r'(\d+\.?\d*)\s*(?:lbs?|pounds?).*?(\d+\.?\d*)\s*(?:ft|feet|\')\s*(\d*)', q)
    if 'bmi' in q and m:
        weight_lbs = float(m.group(1))
        feet = float(m.group(2))
        inches = float(m.group(3)) if m.group(3) else 0
        height_in = feet * 12 + inches
        bmi = round((weight_lbs / (height_in ** 2)) * 703, 1)
        category = (
            "Underweight" if bmi < 18.5 else
            "Normal weight" if bmi < 25 else
            "Overweight" if bmi < 30 else "Obese"
        )
        results.append(f"[CALC] BMI = {bmi} ({category})")

    # ── Calorie deficit to lose weight ────────────────────────
    m = re.search(r'lose\s+(\d+\.?\d*)\s*(?:lbs?|pounds?).*?(\d+)\s*week', q)
    if m:
        lbs = float(m.group(1))
        weeks = int(m.group(2))
        daily_deficit = round((lbs * 3500) / (weeks * 7))
        results.append(
            f"[CALC] To lose {lbs} lbs in {weeks} weeks: "
            f"daily calorie deficit of {daily_deficit} kcal needed "
            f"(~{round(lbs/weeks, 1)} lbs/week)."
        )

    # ── Calorie surplus to gain weight ────────────────────────
    m = re.search(r'gain\s+(\d+\.?\d*)\s*(?:lbs?|pounds?).*?(\d+)\s*week', q)
    if m:
        lbs = float(m.group(1))
        weeks = int(m.group(2))
        daily_surplus = round((lbs * 3500) / (weeks * 7))
        results.append(
            f"[CALC] To gain {lbs} lbs in {weeks} weeks: "
            f"daily calorie surplus of {daily_surplus} kcal needed."
        )

    # ── 1-Rep Max (Epley formula) ─────────────────────────────
    m = re.search(r'(\d+\.?\d*)\s*(lbs?|kg).*?(\d+)\s*rep', q)
    if m and any(kw in q for kw in ['1rm', '1 rep max', 'one rep max', 'max lift']):
        weight = float(m.group(1))
        unit = m.group(2)
        reps = int(m.group(3))
        one_rm = round(weight * (1 + reps / 30), 1)
        results.append(f"[CALC] Estimated 1-rep max (Epley): {one_rm} {unit}")

    # ── TDEE / Daily calories needed ──────────────────────────
    m = re.search(r'(\d+\.?\d*)\s*(?:lbs?|pounds?)', q)
    if m and any(kw in q for kw in ['tdee', 'maintenance calories', 'how many calories', 'daily calories']):
        weight_lbs = float(m.group(1))
        weight_kg = weight_lbs * 0.453592
        # Mifflin-St Jeor (assume moderate activity, neutral sex estimate)
        bmr = 10 * weight_kg + 6.25 * 170 - 5 * 25  # rough neutral estimate
        tdee = round(bmr * 1.55)  # moderate activity
        results.append(
            f"[CALC] Estimated TDEE ≈ {tdee} kcal/day at moderate activity "
            f"(for {weight_lbs} lbs — adjust for height, age, sex, and activity level)."
        )

    # ── Macro split from calorie target ───────────────────────
    m = re.search(r'(\d+)\s*(?:kcal|calories|cal)', q)
    if m and any(kw in q for kw in ['macro', 'protein', 'carb', 'fat']):
        cals = int(m.group(1))
        protein_g = round(cals * 0.30 / 4)
        carbs_g   = round(cals * 0.40 / 4)
        fat_g     = round(cals * 0.30 / 9)
        results.append(
            f"[CALC] Macros for {cals} kcal (30/40/30 split): "
            f"Protein {protein_g}g | Carbs {carbs_g}g | Fat {fat_g}g"
        )

    return results


# ─────────────────────────────────────────────────────────────
# Agentic Answer
# ─────────────────────────────────────────────────────────────

def agentic_answer(agent: AgenticRAGAgent, query: str, k: int = 6) -> str:
    """
    Agentic flow:
    1) Run calculator tool — inject any computed facts
    2) Decompose into subquestions (if multi-part)
    3) Retrieve evidence per subquestion
    4) Generate with computed facts + retrieved context
    """
    # Step 1: Calculator tool
    computed_facts = _calc_tool(query)
    if agent.verbose and computed_facts:
        print(f"[calc_tool] {computed_facts}")

    subqs = _decompose(query)

    # Step 2: Retrieve evidence
    evidence_blocks = []
    if computed_facts:
        evidence_blocks.append("COMPUTED FACTS (use these numbers exactly):\n" + "\n".join(computed_facts))

    for i, sq in enumerate(subqs, 1):
        docs, _conf = retrieve(sq, k=max(3, k // 2))
        if docs:
            context = "\n".join([f"[{j+1}] {d['text']}" for j, d in enumerate(docs)])
        else:
            context = "(No relevant context found)"
        label = f"SUBQUESTION {i}: {sq}" if len(subqs) > 1 else "CONTEXT"
        evidence_blocks.append(f"{label}\n{context}\n")

    synthesis_prompt = (
        f"User question:\n{query}\n\n"
        "Use the evidence below to answer clearly and accurately.\n"
        "If COMPUTED FACTS are present, use those exact numbers in your answer.\n"
        "Cite retrieved sources with [1], [2], etc.\n\n"
        + "\n---\n".join(evidence_blocks)
    )

    return generate_grounded_answer(synthesis_prompt)
