# Spotter_AI.py
"""
Unified chat backend for GymBot (Qwen Instruct family).
- Dynamic sampling based on intent + retrieval confidence
- Per-call seeding (no global randomness)
- Safe context-length trimming
- Works with tokenizer.apply_chat_template (text) or processor.apply_chat_template (VL)

Env vars (optional):
  SPOTTER_MODEL=Qwen/Qwen2.5-7B-Instruct
  DEVICE_MAP=auto|cpu|cuda
  LOAD_IN_4BIT=true|false   (requires bitsandbytes)
"""

from __future__ import annotations
import os, re, math, hashlib
from typing import List, Dict, Any, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,          # for VL models if available
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# --------------------
# Model / tokenizer / processor setup
# --------------------
MODEL_NAME = os.getenv("SPOTTER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")  # "auto" | "cuda" | "cpu"
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() in {"1","true","yes","y"}

# Choose dtype
if torch.cuda.is_available():
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32

# Tokenizer (always present)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Try to load a processor (VL models have one); if it fails, we’ll use tokenizer for chat template
processor = None
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
except Exception:
    processor = None

# Load model (optionally 4-bit)
bnb_config = None
if LOAD_IN_4BIT:
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    except Exception:
        bnb_config = None  # fall back to regular load

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE_MAP,
    torch_dtype=DTYPE,
    quantization_config=bnb_config if LOAD_IN_4BIT and bnb_config is not None else None,
)

# --------------------
# Helpers: intent detection + sampling policy
# --------------------
def detect_intent(text: str) -> str:
    """Very light heuristic intent classifier."""
    t = (text or "").lower()
    if any(k in t for k in ["injury","pain","hurts","impingement","strain","sprain","tendonitis","tendinitis","back pain","shoulder pain"]):
        return "safety"
    if any(k in t for k in ["plan","program","template","make me a","session","routine","workout","split"]):
        return "plan"
    if any(k in t for k in ["motivate","pep talk","caption","slogan"]):
        return "creative"
    return "knowledge"

def sampling_params(intent: str, confidence: float) -> dict:
    """
    Returns kwargs for sampling. Confidence is 0..1 from your RAG.
    """
    # Conservative for factual/safety or low confidence
    if intent in {"knowledge","safety"}: return dict(do_sample=False)
    if confidence < 0.55:              return dict(do_sample=False)

    if intent == "plan":
        return dict(do_sample=True, temperature=0.30, top_p=0.90, repetition_penalty=1.05)
    if intent == "creative":
        return dict(do_sample=True, temperature=0.70, top_p=0.95, repetition_penalty=1.05)
    # default mild creativity
    return dict(do_sample=True, temperature=0.20, top_p=0.90, repetition_penalty=1.05)

def deterministic_seed_from(text: str, user_id: str = "anon") -> int:
    """Optional: use for stable phrasing per (user, text) when sampling."""
    h = hashlib.sha256(f"{user_id}:{text}".encode()).hexdigest()
    return int(h, 16) % (2**31 - 1)

# --------------------
# Chat template application + safe truncation
# --------------------
def _apply_chat_template(msgs: List[Dict[str,str]]):
    """
    Returns input_ids tensor on model.device using either processor or tokenizer template.
    """
    obj = processor if (processor is not None and hasattr(processor, "apply_chat_template")) else tokenizer
    return obj.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

def _token_len(msgs: List[Dict[str,str]]) -> int:
    try:
        return _apply_chat_template(msgs).shape[-1]
    except Exception:
        # fallback: rough estimate via tokenizer
        text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in msgs)
        return len(tokenizer.encode(text))

def truncate_messages_to_fit(
    messages: List[Dict[str,str]],
    reserve_new_tokens: int = 400
) -> List[Dict[str,str]]:
    """
    Iteratively trims oldest non-system messages until the prompt fits the model context
    with space for generation.
    """
    max_ctx = getattr(tokenizer, "model_max_length", 4096)
    cur = messages[:]
    while True:
        n = _token_len(cur)
        if n + reserve_new_tokens <= max_ctx:
            return cur
        # drop oldest non-system message
        drop_idx = next((i for i, m in enumerate(cur) if m.get("role") != "system"), None)
        if drop_idx is None or len(cur) <= 2:
            # extreme fallback: keep system + last user
            return cur[-2:]

# --------------------
# Public: build_messages + chat_text
# --------------------
def build_messages(system: str, user: str) -> List[Dict[str,str]]:
    """Convenience to create a minimal message list."""
    return [{"role":"system","content":system}, {"role":"user","content":user}]

def chat_text(
    messages: List[Dict[str,str]],
    *,
    max_new_tokens: int = 400,
    intent: Optional[str] = None,       # pass from RAG if you already detected it
    confidence: float = 1.0,            # 0..1 from RAG; used to tighten sampling
    seed: Optional[int] = None,         # per-call seed for reproducible sampling
    do_sample: Optional[bool] = None,   # manual overrides (optional)
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> str:
    """
    Text-only chat. Messages follow {"role": "...", "content": "..."}.
    """
    # 1) Decide sampling policy
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    _intent = intent or detect_intent(last_user)
    policy = sampling_params(_intent, confidence)

    # Allow explicit overrides
    if do_sample is not None:          policy["do_sample"] = do_sample
    if temperature is not None:        policy["temperature"] = temperature
    if top_p is not None:              policy["top_p"] = top_p
    if repetition_penalty is not None: policy["repetition_penalty"] = repetition_penalty

    # 2) Trim to context
    msgs = truncate_messages_to_fit(messages, reserve_new_tokens=max_new_tokens)

    # 3) Tokenize
    inputs = {"input_ids": _apply_chat_template(msgs)}

    # 4) Generator (for reproducible sampling)
    gen = None
    if policy.get("do_sample", False) and seed is not None:
        gen = torch.Generator(device=model.device).manual_seed(int(seed))

    # 5) Build kwargs (avoid passing None)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=policy.get("do_sample", False),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )
    if gen is not None:
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None and k != 'generator'}
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = float(policy.get("temperature", 0.2))
        gen_kwargs["top_p"] = float(policy.get("top_p", 0.9))
        gen_kwargs["repetition_penalty"] = float(policy.get("repetition_penalty", 1.0))
        gen_kwargs["no_repeat_ngram_size"] = 3  # mild loop guard

    # 6) Generate
    model.eval()
    with torch.inference_mode():
        #print(gen_kwargs)
        out = model.generate(**inputs, **{k:v for k,v in gen_kwargs.items() if v is not None})

    # Decode only the NEW tokens (exclude the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = out[:, input_length:]
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    return text

# Public API
__all__ = [
    "chat_text",
    "build_messages",
    "detect_intent",
    "sampling_params",
    "deterministic_seed_from",
]

# --------------------
# Quick manual test
# --------------------
if __name__ == "__main__":
    print("Spotter AI - LLM Backend")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE_MAP}")
    print(f"4-bit: {LOAD_IN_4BIT}")
    print("\nUse this module by importing:")
    print("  from Spotter_AI import chat_text, build_messages")

