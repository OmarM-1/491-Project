#SAFETY-GATE-AGENT
import re 
from typing import Tuple, Callable
from Spotter_AI import chat_text

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

SAFETY_PROMPT = [
    {"role": "system", "content": "You are a safety gate agent. Your job is to identify if the user input contains any phrases that may indicate a medical emergency. If you find any, you must flag the input and provide a warning message advising the user to seek immediate medical attention or call emergency services."},
    {"role": "user", "content": "Screen message for medical emergencies, eating disorders, or steroid/drug abuse. Return safe or unsafe."},
    {"role": "assistant", "content": "If the input contains any phrases that may indicate a medical emergency, respond with: 'The input contains phrases that may indicate a medical emergency. Please seek immediate medical attention or call emergency services if you are experiencing any of these symptoms.' Otherwise, respond with: 'The input does not contain any phrases that indicate a medical emergency.'"}
]

def safety_gate_agent(user_input: str, chat: Callable) -> Tuple[bool, str]:
    if SAFE_RULES.search(user_input.lower()):
        messages = SAFETY_PROMPT + [{"role": "user", "content": f"User input: {user_input}"}]
        response = chat(messages, max_tokens=100, temperature=0).strip()[-6:].upper()
        if "UNSAFE" in response:
            return (False, "I might not be able to help with that. If you are experiencing a medical emergency, please seek immediate medical attention or call emergency services. I can help with all else once you are cleared of any issue. Take care!")
    return (True, "")
