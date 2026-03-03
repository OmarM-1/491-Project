#SAFETY-GATE-AGENT
import re 
from typing import Tuple, Callable, Optional, List, Dict, Any

FLAG = [
    r"chest pain| tight(ness)?| pressure| squeezing| discomfort",
    r"shortness of breath| difficulty breathing| breath(less)?| wheezing",
    r"pain radiating to arm| heavy-lift(ing)?|steroid(s)?|tren| anabolic|sarms|dnp",
    r"nausea| vomiting| sweating| lightheadedness| dizz(y|iness)",
    r"severe headache| sudden numbness| weakness| confusion| trouble speaking| vision changes| loss of balance| coordination",
    r"abdominal pain| cramping| bloating",
    r"severe allergic reaction| anaphylaxis| swelling of face| lips| tongue",
    r"high fever| persistent cough| shortness of breath",
    r"severe abdominal pain| persistent vomiting| blood in vomit"
]

SAFE_RULES = re.compile("|".join(FLAG), re.IGNORECASE)

WARNING_MSG = (
    "I might not be able to help with that. Your message includes symptoms or terms that could indicate a "
    "medical emergency or dangerous substance use. If you or someone else is experiencing these symptoms, "
    "please seek immediate medical attention or call emergency services. "
    "If you're safe and not in immediate danger, tell me more and I can help with general information."
)

def safety_gate_agent(
    user_input: str,
    chat: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> Tuple[bool, str]:
    """
    Returns (is_safe, message).
    - If regex flags something, we block immediately (no LLM required).
    - If you later add an LLM 'chat' function, you can optionally do a second-pass check.
    """
    flagged = bool(SAFE_RULES.search(user_input))
    if not flagged:
        return True, ""

    # If no LLM, block on regex match (recommended for now)
    if chat is None:
        return False, WARNING_MSG

    # Optional: second-pass with LLM (only if you actually have a chat function)
    try:
        messages = [
            {"role": "system", "content": "Return exactly SAFE or UNSAFE for the user input."},
            {"role": "user", "content": user_input},
        ]
        resp = chat(messages).strip().upper()
        if "UNSAFE" in resp:
            return False, WARNING_MSG
    except Exception:
        # If LLM fails, be conservative when regex flagged
        return False, WARNING_MSG

    return True, ""