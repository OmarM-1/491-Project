# Spotter_AI.py
# Vision-language model for GymBot — text chat + image/video analysis

from typing import List, Dict
from PIL import Image
import torch
import os

# ─────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────
MODEL_ID = os.environ.get("SPOTTER_MODEL", "Qwen/Qwen2-VL-2B-Instruct")

# Pick the right model class based on which model is being loaded
# Qwen2-VL-*   → Qwen2VLForConditionalGeneration
# Qwen2.5-VL-* → Qwen2_5_VLForConditionalGeneration
if "2.5" in MODEL_ID or "2_5" in MODEL_ID:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    ModelClass = Qwen2_5_VLForConditionalGeneration
else:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    ModelClass = Qwen2VLForConditionalGeneration

# ─────────────────────────────────────────────
# DEVICE & DTYPE
# ─────────────────────────────────────────────
HAS_MPS  = torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()
DEVICE   = "mps" if HAS_MPS else ("cuda" if HAS_CUDA else "cpu")

if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
elif DEVICE in ("mps", "cuda"):
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"[INFO] Loading {MODEL_ID} on {DEVICE} (dtype={dtype}) ...")

# ─────────────────────────────────────────────
# LOAD MODEL + PROCESSOR  (once, at import)
# ─────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
model = ModelClass.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
).eval()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_messages(system: str, user: str) -> List[Dict]:
    """Build a standard messages list for chat_text."""
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def chat_text(
    messages: List[Dict],
    max_new_tokens: int = 300,
    temperature: float = 0.5,
    **kwargs          # accept extra kwargs from RAG callers (intent, confidence, seed)
) -> str:
    """
    Text-only chat.

    messages format:
      [
        {"role": "system", "content": "You are ..."},
        {"role": "user",   "content": "Question..."}
      ]
    """
    # Step 1: format the chat into a single string
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Step 2: tokenize to tensors
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    # Return only the newly generated tokens (strip the prompt)
    new_tokens = out[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def load_image(path: str) -> Image.Image:
    """Load a PIL image from a file path."""
    return Image.open(path).convert("RGB")


def load_video(path: str, max_frames: int = 8) -> List[Image.Image]:
    """Extract up to max_frames evenly-spaced frames from a video."""
    import cv2
    cap    = cv2.VideoCapture(path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / max_frames) for i in range(max_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def chat_vision(
    messages: List[Dict],
    max_new_tokens: int = 500,
    temperature: float = 0.3,
    **kwargs
) -> str:
    """
    Vision chat — messages may contain PIL image blocks.

    messages format:
      [
        {"role": "system", "content": "..."},
        {"role": "user", "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text",  "text": "Analyse my squat..."}
        ]}
      ]
    """
    # Collect PIL images in message order
    images = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image":
                    images.append(block["image"])

    # Build text with special image tokens
    text_input = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text_input],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    new_tokens = out[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    msgs = build_messages(
        system="You are a concise, safety-first fitness coach.",
        user="Make me a 3-day full-body plan for a beginner."
    )
    print("\n=== TEXT CHAT ===")
    print(chat_text(msgs, max_new_tokens=350, temperature=0.4))
