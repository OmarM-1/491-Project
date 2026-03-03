# qwen_vl_chat.py
# Minimal, single-file Qwen2.5-VL-7B-Instruct chat (text and image+text).

from typing import List, Dict, Union
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
import os

# ------------------------------
# MODEL CHOICE (VL = vision-language)
# ------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ------------------------------
# DEVICE & DTYPE
# ------------------------------
HAS_MPS = torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()
DEVICE = "mps" if HAS_MPS else ("cuda" if HAS_CUDA else "cpu")

# Prefer bfloat16 on capable GPUs; else float16 on GPU; else float32 on CPU.
if DEVICE in ("mps", "cuda"):
    # bfloat16 if CUDA says it’s supported; Apple “mps” generally likes float16
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    dtype = torch.float32

print(f"[INFO] Loading {MODEL_ID} on {DEVICE} (dtype={dtype}) ...")

# ------------------------------
# LOAD PROCESSOR + MODEL
# ------------------------------
# Processor handles BOTH text formatting (chat template) and vision pre/post-processing.
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=dtype,                 
    device_map="auto",
    trust_remote_code=True,
).eval()

# ------------------------------
# HELPERS
# ------------------------------
def chat_text(messages: List[Dict], max_new_tokens: int = 300, temperature: float = 0.5) -> str:
    """
    messages example:
    [
      {"role": "system", "content": "You are a concise, safety-first fitness coach."},
      {"role": "user",   "content": "Explain progressive overload in 2 sentences."}
    ]
    """
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def chat_vision(image_path: str, user_text: str) -> str:
    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": "You are a precise, safety-first fitness coach."},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": user_text}
        ]}
    ]
    # text side (adds special tokens)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # image tensors
    vis = processor(images=[img], return_tensors="pt")
    inputs.update(vis)
    # move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, temperature=0.2, do_sample=True)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


# ------------------------------
# DEMO (run: python qwen_vl_chat.py)
# ------------------------------
if __name__ == "__main__":
    # Text-only demo
    text_demo = [
        {"role": "system", "content": "You are a concise, safety-first fitness coach."},
        {"role": "user", "content": "Make me a 3-day full-body plan for a beginner."}
    ]
    print("\n=== TEXT CHAT ===")
    print(chat_text(text_demo, max_new_tokens=350, temperature=0.4))

    # Image+text demo (uncomment and point to a real image file on disk)
    # print("\n=== IMAGE + TEXT CHAT ===")
    # print(chat_vision("squat.jpg", "Is my lumbar spine neutral? Give 3 cues and a safer regression."))
