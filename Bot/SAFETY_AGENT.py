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

def safety_gate_agent(user_input: str, chat: Callable) -> Tuple[bool, str]:
    if SAFE_RULES.search(user_input.lower()):
        messages = SAFETY_PROMPT + [{"role": "user", "content": f"User input: {user_input}"}]
        response = chat(messages, max_tokens=100, temperature=0).strip()[-6:].upper()
        if "UNSAFE" in response:
            return (False, "I might not be able to help with that. If you are experiencing a medical emergency, please seek immediate medical attention or call emergency services. I can help with all else once you are cleared of any issue. Take care!")
    return (True, "")
