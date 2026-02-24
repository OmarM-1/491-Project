<<<<<<< HEAD
# Spotter_AI.py
"""
Unified chat backend for GymBot (Qwen Instruct + Qwen2-VL family).
- Dynamic sampling based on intent + retrieval confidence
- Per-call seeding (no global randomness)
- Safe context-length trimming
- Works with tokenizer.apply_chat_template (text) or processor.apply_chat_template (VL)
- Vision support for Qwen2-VL models (form checking, exercise recognition)

Env vars (optional):
  SPOTTER_MODEL=Qwen/Qwen2.5-7B-Instruct  (text-only)
  SPOTTER_MODEL=Qwen/Qwen2-VL-2B-Instruct (vision-enabled)
  DEVICE_MAP=auto|cpu|cuda|mps
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
    AutoConfig,
    BitsAndBytesConfig,
)

# Import VL model class
try:
    from transformers import Qwen2VLForConditionalGeneration
    HAS_VL_MODEL = True
except ImportError:
    HAS_VL_MODEL = False

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

# Detect if this is a VL model by checking config
config = AutoConfig.from_pretrained(MODEL_NAME)
is_vl_model = config.__class__.__name__ == "Qwen2VLConfig"

if is_vl_model and HAS_VL_MODEL:
    # Use VL model class for Qwen2-VL
    print(f"Loading vision-language model: {MODEL_NAME}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE_MAP,
        torch_dtype=DTYPE,
        quantization_config=bnb_config if LOAD_IN_4BIT and bnb_config is not None else None,
    )
else:
    # Use standard causal LM for text-only models
    print(f"Loading text-only model: {MODEL_NAME}")
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

# --------------------
# Vision support for Qwen2-VL models
# --------------------
def load_image(image_source):
    """
    Load image from various sources: PIL Image, file path, URL, or base64
    
    Args:
        image_source: Can be:
            - PIL.Image object
            - str (file path or URL)
            - bytes (raw image data)
    
    Returns:
        PIL.Image object
    """
    from PIL import Image
    import io
    
    if isinstance(image_source, Image.Image):
        return image_source
    
    elif isinstance(image_source, str):
        # Check if URL
        if image_source.startswith(('http://', 'https://')):
            import requests
            response = requests.get(image_source)
            return Image.open(io.BytesIO(response.content))
        # Otherwise treat as file path
        else:
            return Image.open(image_source)
    
    elif isinstance(image_source, bytes):
        return Image.open(io.BytesIO(image_source))
    
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

def load_video(video_source, max_frames: int = 8):
    """
    Load and sample frames from video for VLM processing
    
    Args:
        video_source: Video file path or URL
        max_frames: Maximum number of frames to extract (default: 8)
                   VLMs work best with 4-16 frames
    
    Returns:
        List of PIL.Image objects (sampled frames)
    """
    try:
        import cv2
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("Video support requires: pip install opencv-python")
    
    # Open video
    if isinstance(video_source, str) and video_source.startswith(('http://', 'https://')):
        import requests
        import tempfile
        # Download to temp file
        response = requests.get(video_source)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(response.content)
            video_path = tmp.name
    else:
        video_path = video_source
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_source}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError("Video has no frames")
    
    # Calculate frame indices to sample (evenly distributed)
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    
    if not frames:
        raise ValueError("Could not extract any frames from video")
    
    return frames

def build_vision_messages(system: str, user_text: str, images: List) -> List[Dict]:
    """
    Build messages for vision-language models (images only)
    
    Args:
        system: System prompt
        user_text: User's text query
        images: List of images (PIL Images, paths, or URLs)
    
    Returns:
        Messages list with proper multimodal content structure
    """
    # Load all images
    loaded_images = [load_image(img) for img in images]
    
    # Build content list with text and images
    content = []
    
    # Add images first
    for img in loaded_images:
        content.append({"type": "image", "image": img})
    
    # Add text
    content.append({"type": "text", "text": user_text})
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content}
    ]

def build_video_messages(system: str, user_text: str, videos: List, max_frames_per_video: int = 8) -> List[Dict]:
    """
    Build messages for video analysis
    
    Args:
        system: System prompt
        user_text: User's text query
        videos: List of video paths or URLs
        max_frames_per_video: Frames to sample per video (default: 8)
    
    Returns:
        Messages list with video frames as images
    """
    content = []
    
    # Process each video
    for video in videos:
        frames = load_video(video, max_frames=max_frames_per_video)
        for frame in frames:
            content.append({"type": "image", "image": frame})
    
    # Add text
    content.append({"type": "text", "text": user_text})
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content}
    ]

def chat_vision(
    messages: List[Dict[str,Any]],
    *,
    max_new_tokens: int = 400,
    intent: Optional[str] = None,
    confidence: float = 1.0,
    seed: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> str:
    """
    Vision-language chat for Qwen2-VL models
    
    Messages should follow format:
    [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "What's in this image?"}
        ]}
    ]
    
    Or use build_vision_messages() helper function.
    """
    if processor is None:
        raise RuntimeError(
            "Vision chat requires a processor. "
            "Make sure you're using a VL model like Qwen/Qwen2-VL-2B-Instruct"
        )
    
    # 1) Extract text from last user message for intent detection
    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        last_user_text = item.get("text", "")
                        break
            else:
                last_user_text = content
            break
    
    # 2) Decide sampling policy
    _intent = intent or detect_intent(last_user_text)
    policy = sampling_params(_intent, confidence)
    
    # Allow explicit overrides
    if do_sample is not None:          policy["do_sample"] = do_sample
    if temperature is not None:        policy["temperature"] = temperature
    if top_p is not None:              policy["top_p"] = top_p
    if repetition_penalty is not None: policy["repetition_penalty"] = repetition_penalty
    
    # 3) Prepare messages in Qwen2-VL format
    # Extract images and build text with proper image references
    images = []
    formatted_messages = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if isinstance(content, list):
            # Multimodal content - extract images and text
            text_parts = []
            for item in content:
                if item.get("type") == "image":
                    images.append(item.get("image"))
                    # Qwen2-VL expects this exact format for image references
                    text_parts.append({"type": "image"})
                elif item.get("type") == "text":
                    text_parts.append({"type": "text", "text": item.get("text", "")})
            
            formatted_messages.append({
                "role": role,
                "content": text_parts
            })
        else:
            # Simple text content
            formatted_messages.append({
                "role": role,
                "content": content
            })
    
    # 4) Use processor to prepare inputs (handles tokenization + image processing)
    text = processor.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process with images
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 5) Generator (for reproducible sampling)
    gen = None
    if policy.get("do_sample", False) and seed is not None:
        gen = torch.Generator(device=model.device).manual_seed(int(seed))
    
    # 6) Build generation kwargs
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=policy.get("do_sample", False),
        eos_token_id=getattr(processor.tokenizer if hasattr(processor, 'tokenizer') else tokenizer, "eos_token_id", None),
        pad_token_id=getattr(processor.tokenizer if hasattr(processor, 'tokenizer') else tokenizer, "pad_token_id", None),
    )
    
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = float(policy.get("temperature", 0.2))
        gen_kwargs["top_p"] = float(policy.get("top_p", 0.9))
        gen_kwargs["repetition_penalty"] = float(policy.get("repetition_penalty", 1.0))
        gen_kwargs["no_repeat_ngram_size"] = 3
    
    # Filter out None values
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    
    # 7) Generate
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # 8) Decode (extract only new tokens)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, input_length:]
    
    # Use processor's tokenizer for decoding
    decoder = processor.tokenizer if hasattr(processor, 'tokenizer') else tokenizer
    text = decoder.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    
    return text

def chat_video(
    messages: List[Dict[str,Any]],
    *,
    max_new_tokens: int = 600,
    intent: Optional[str] = None,
    confidence: float = 1.0,
    seed: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> str:
    """
    Convenience wrapper for video analysis
    Uses chat_vision under the hood (videos are processed as frame sequences)
    
    Args:
        messages: Messages with video frames as images
        max_new_tokens: Longer default (600) for comprehensive video analysis
        Other args: Same as chat_vision
    
    Returns:
        Analysis text
    """
    return chat_vision(
        messages,
        max_new_tokens=max_new_tokens,
        intent=intent,
        confidence=confidence,
        seed=seed,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

# Public API
__all__ = [
    "chat_text",
    "chat_vision",
    "chat_video",
    "build_messages",
    "build_vision_messages",
    "build_video_messages",
    "load_image",
    "load_video",
    "detect_intent",
    "sampling_params",
    "deterministic_seed_from",
]

# --------------------
# Quick manual test
# --------------------
if __name__ == "__main__":
    print("="*60)
    print("Spotter AI - LLM Backend")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE_MAP}")
    print(f"4-bit: {LOAD_IN_4BIT}")
    print(f"Processor: {'✅ Available (VL capable)' if processor else '❌ Not available (text only)'}")
    
    print("\n" + "="*60)
    print("TEXT CHAT USAGE:")
    print("="*60)
    print("""
from Spotter_AI import chat_text, build_messages

# Simple chat
messages = build_messages(
    system="You are a fitness assistant",
    user="What muscles does a squat work?"
)
response = chat_text(messages)
print(response)
    """)
    
    if processor:
        print("\n" + "="*60)
        print("VISION CHAT USAGE:")
        print("="*60)
        print("""
from Spotter_AI import chat_vision, build_vision_messages, load_image

# Analyze exercise form from image
messages = build_vision_messages(
    system="You are an AI spotter analyzing exercise form",
    user_text="Check my squat form and give me feedback",
    images=["squat_photo.jpg"]
)
response = chat_vision(messages)
print(response)

# Or build manually with multiple images
from PIL import Image
img1 = load_image("exercise1.jpg")
img2 = load_image("exercise2.jpg")

messages = [
    {"role": "system", "content": "You are a fitness expert"},
    {"role": "user", "content": [
        {"type": "image", "image": img1},
        {"type": "image", "image": img2},
        {"type": "text", "text": "Compare these two exercise forms"}
    ]}
]
response = chat_vision(messages)
        """)
        
        print("\n" + "="*60)
        print("VIDEO ANALYSIS USAGE:")
        print("="*60)
        print("""
from Spotter_AI import chat_video, build_video_messages

# Analyze exercise video (automatically samples key frames)
messages = build_video_messages(
    system="You are an AI spotter. Analyze movement patterns across the entire set.",
    user_text="Analyze my squat set. Check form consistency and identify when fatigue affects technique.",
    videos=["squat_set.mp4"],
    max_frames_per_video=8  # Samples 8 evenly-spaced frames
)
response = chat_video(messages)
print(response)

# Compare multiple videos
messages = build_video_messages(
    system="You are a fitness coach comparing progress over time",
    user_text="Compare my form between these two videos. Has my technique improved?",
    videos=["week1_squat.mp4", "week4_squat.mp4"]
)
response = chat_video(messages)
        """)
    else:
        print("\nâš ï¸  Vision chat not available with current model")
        print("To enable vision, use a VL model:")
        print("  export SPOTTER_MODEL='Qwen/Qwen2-VL-2B-Instruct'")
    
    print("\n" + "="*60)


=======
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
>>>>>>> 380e9e4524810277600acd74c7b90bbbe3505971
