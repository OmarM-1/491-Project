#!/usr/bin/env python3
"""
GymBot Backend API
==================
Drop-in backend for your existing UI.

Endpoints:
  GET  /info          → hardware/model info
  POST /chat          → text query → RAG answer
  POST /analyse       → image/video files → vision analysis
  POST /chat-with-media → text + optional files (combined, like Claude.ai)

Run:
  python gymbot_api.py
  python gymbot_api.py --port 8080

Connect your UI to http://localhost:8000
"""

import os
import sys
import re
import time
import uuid
import shutil
import tempfile
import argparse
import platform
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
# HARDWARE AUTO-CONFIG
# ─────────────────────────────────────────────────────────────
def configure_hardware() -> dict:
    system  = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine in ("arm64", "aarch64"):
        os.environ.setdefault("SPOTTER_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
        os.environ.setdefault("DEVICE_MAP",    "mps")
        os.environ.setdefault("LOAD_IN_4BIT",  "false")
        return {"hw": "Apple Silicon", "device": "MPS", "model": "Qwen2-VL-2B"}

    elif system == "Darwin":
        os.environ.setdefault("SPOTTER_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
        os.environ.setdefault("DEVICE_MAP",    "cpu")
        os.environ.setdefault("LOAD_IN_4BIT",  "false")
        return {"hw": "Intel Mac", "device": "CPU", "model": "Qwen2-VL-2B"}

    else:
        try:
            import torch
            if torch.cuda.is_available():
                os.environ.setdefault("SPOTTER_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
                os.environ.setdefault("DEVICE_MAP",    "auto")
                os.environ.setdefault("LOAD_IN_4BIT",  "true")
                return {"hw": "NVIDIA GPU", "device": "CUDA", "model": "Qwen2-VL-7B"}
        except ImportError:
            pass
        os.environ.setdefault("SPOTTER_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
        os.environ.setdefault("DEVICE_MAP",    "cpu")
        os.environ.setdefault("LOAD_IN_4BIT",  "false")
        return {"hw": "CPU", "device": "CPU", "model": "Qwen2-VL-2B"}

HW_CONFIG = configure_hardware()

# ─────────────────────────────────────────────────────────────
# LAZY MODULE LOADING
# ─────────────────────────────────────────────────────────────
_rag_loaded    = False
_vision_loaded = False

def _load_rag():
    global _rag_loaded
    global generate_grounded_answer, agentic_answer
    global REGULAR_RAG_AVAILABLE, AGENTIC_RAG_AVAILABLE
    global safety_gate_agent, chat_text
    if _rag_loaded:
        return

    REGULAR_RAG_AVAILABLE = False
    AGENTIC_RAG_AVAILABLE = False

    try:
        from optimized_rag import generate_grounded_answer as _g
        generate_grounded_answer = _g
        REGULAR_RAG_AVAILABLE = True
    except ImportError:
        generate_grounded_answer = None

    try:
        from agentic_rag import agentic_answer as _a
        agentic_answer = _a
        AGENTIC_RAG_AVAILABLE = True
    except ImportError:
        agentic_answer = None

    try:
        from SAFETY_AGENT import safety_gate_agent as _sg
        safety_gate_agent = _sg
    except ImportError:
        safety_gate_agent = lambda q, **_: (True, "")

    try:
        from Spotter_AI import chat_text as _ct
        chat_text = _ct
    except ImportError:
        chat_text = None

    _rag_loaded = True


def _load_vision():
    global _vision_loaded
    global chat_vision, load_image, load_video
    if _vision_loaded:
        return
    from Spotter_AI import chat_vision as _cv, load_image as _li, load_video as _lv
    chat_vision  = _cv
    load_image   = _li
    load_video   = _lv
    _vision_loaded = True


# ─────────────────────────────────────────────────────────────
# COMPLEXITY SCORER
# ─────────────────────────────────────────────────────────────
class ComplexityScorer:
    CALC_KEYWORDS    = ['how many','how much','calculate','compute','calories',
                        'pounds','kilograms','weeks','months','percent','bmi',
                        'weight','gain','lose','if i','should i']
    COMPLEX_KEYWORDS = ['and','both','also','compare','versus','vs','plan',
                        'routine','program','schedule','multiple','several',
                        'different','best way','optimal','recommend']
    SIMPLE_KEYWORDS  = ['what is','define','definition','explain','show me',
                        'how to do','form','technique','muscles worked','benefits of']

    @classmethod
    def score(cls, q: str) -> int:
        ql, s = q.lower(), 5
        if any(k in ql for k in cls.CALC_KEYWORDS):    s += 3
        cx = sum(1 for k in cls.COMPLEX_KEYWORDS if k in ql)
        s += (2 if cx >= 2 else 1 if cx == 1 else 0)
        if any(k in ql for k in cls.SIMPLE_KEYWORDS):  s -= 2
        wc = len(q.split())
        if q.count('?') == 1 and wc < 15:              s -= 1
        if wc > 25:                                     s += 1
        elif wc < 10:                                   s -= 1
        if re.search(r'\d+', q):                        s += 1
        return max(0, min(10, s))


# ─────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────
IMAGE_EXTS = {'.jpg','.jpeg','.png','.webp','.gif','.bmp'}
VIDEO_EXTS = {'.mp4','.mov','.avi','.mkv','.webm','.m4v'}

def answer_text(query: str) -> tuple[str, dict]:
    """Run a text query through safety → complexity → RAG. Returns (answer, meta)."""
    _load_rag()

    if safety_gate_agent:
        ok, warning = safety_gate_agent(
            query,
            chat=lambda msgs, **kw: chat_text(msgs, intent="safety",
                                              confidence=0.0, seed=0) if chat_text else ""
        )
        if not ok:
            return warning, {"system": "safety"}

    if not REGULAR_RAG_AVAILABLE and not AGENTIC_RAG_AVAILABLE:
        return ("⚠️ No RAG system loaded. Ensure optimized_rag.py or "
                "agentic_rag.py is in the same directory."), {"system": "none"}

    score      = ComplexityScorer.score(query)
    use_agentic = score >= 7 and AGENTIC_RAG_AVAILABLE

    if use_agentic:
        answer = agentic_answer(query)
        system = "agentic"
    elif REGULAR_RAG_AVAILABLE:
        answer = generate_grounded_answer(query)
        system = "regular"
    else:
        answer = agentic_answer(query)
        system = "agentic"

    return answer, {"system": system, "complexity_score": score}


def answer_media(file_paths: list[str], caption: str = "") -> tuple[str, dict]:
    """Analyse image/video files. Returns (answer, meta)."""
    _load_vision()

    images_list, video_paths, image_paths = [], [], []

    for path in file_paths:
        ext = Path(path).suffix.lower()
        if ext in VIDEO_EXTS:
            video_paths.append(path)
        elif ext in IMAGE_EXTS:
            image_paths.append(path)

    for vp in video_paths:
        images_list.extend(load_video(vp, max_frames=8))

    for ip in image_paths:
        images_list.append(load_image(ip))

    if not images_list:
        return "No valid media found to analyse.", {"system": "vision"}

    n_vids, n_imgs = len(video_paths), len(image_paths)
    user_note = f'\n\nUser note: "{caption}"' if caption.strip() else ""

    if n_vids and n_imgs:
        prompt = (
            f"You are an expert AI spotter reviewing {n_vids} video(s) "
            f"and {n_imgs} photo(s) of an exercise.{user_note}\n\n"
            "1. **Exercise** – what movement is shown?\n"
            "2. **Form Rating** – score 1-10 and justify\n"
            "3. **Rep Consistency** – does form hold across the set?\n"
            "4. **Key Issues** – top 2-3 problems, most critical first\n"
            "5. **Corrections** – specific, actionable fixes\n"
            "6. **Safety** – any injury risks?\n"
        )
    elif n_vids:
        prompt = (
            f"Analyse this exercise video ({len(images_list)} frames).{user_note}\n\n"
            "1. **Exercise** – what is being performed?\n"
            "2. **Form Rating** – 1-10\n"
            "3. **Rep Breakdown** – which rep does form deteriorate?\n"
            "4. **Key Issues** – top 2-3 problems\n"
            "5. **Corrections** – specific fixes\n"
            "6. **Safety** – injury risks?\n"
        )
    elif n_imgs == 1:
        prompt = (
            f"Analyse this exercise photo.{user_note}\n\n"
            "1. **Exercise** – what and what phase?\n"
            "2. **Form Rating** – 1-10\n"
            "3. **Key Issues** – main problems\n"
            "4. **Corrections** – specific fixes\n"
            "5. **Safety** – any risks?\n"
        )
    else:
        prompt = (
            f"Analyse these {n_imgs} exercise photos.{user_note}\n\n"
            "1. **Exercise** – what is shown?\n"
            "2. **Multi-Angle Assessment** – what each view reveals\n"
            "3. **Form Rating** – overall 1-10\n"
            "4. **Key Issues** – priority problems\n"
            "5. **Corrections** – specific fixes\n"
            "6. **Safety** – any concerns?\n"
        )

    content  = [{"type": "image", "image": img} for img in images_list]
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system",  "content": "You are an expert AI spotter analysing exercise form."},
        {"role": "user",    "content": content},
    ]
    max_tokens = 600 if len(images_list) <= 5 else 1000
    answer = chat_vision(messages, max_new_tokens=max_tokens)
    return answer, {"system": "vision", "files": len(file_paths)}


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("\n❌  FastAPI / uvicorn not found.\n"
          "    Install: pip install fastapi uvicorn python-multipart\n")
    sys.exit(1)

app = FastAPI(title="GymBot API")

# Allow your UI's dev server (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="gymbot_"))


# ── GET /info ──────────────────────────────────────────────
@app.get("/info")
async def info():
    """Hardware/model info for your UI's status display."""
    return HW_CONFIG


# ── POST /chat ─────────────────────────────────────────────
@app.post("/chat")
async def chat(message: str = Form(...)):
    """
    Plain text query → RAG answer.

    Request  (multipart or form):
      message: str

    Response:
      { answer: str, meta: { system, complexity_score, time_seconds } }
    """
    t0 = time.time()
    try:
        answer, meta = answer_text(message)
        meta["time_seconds"] = round(time.time() - t0, 2)
        return {"answer": answer, "meta": meta}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── POST /analyse ──────────────────────────────────────────
@app.post("/analyse")
async def analyse(
    files:   list[UploadFile] = File(...),
    caption: str              = Form(""),
):
    """
    Upload images/videos → vision analysis.

    Request  (multipart):
      files:   one or more image/video files
      caption: optional text note about the upload

    Response:
      { answer: str, meta: { system, files, time_seconds } }
    """
    t0    = time.time()
    saved = []
    try:
        for upload in files:
            dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{upload.filename}"
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            saved.append(str(dest))

        answer, meta = answer_media(saved, caption=caption)
        meta["time_seconds"] = round(time.time() - t0, 2)
        return {"answer": answer, "meta": meta}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        for path in saved:
            try: os.remove(path)
            except OSError: pass


# ── POST /chat-with-media ──────────────────────────────────
@app.post("/chat-with-media")
async def chat_with_media(
    message: str              = Form(""),
    files:   list[UploadFile] = File([]),
):
    """
    Combined endpoint — behaves like a modern chat box:
      • files present  → vision analysis  (caption = message)
      • no files       → RAG text answer

    Request  (multipart):
      message: str  (question or caption)
      files:   zero or more image/video files

    Response:
      { answer: str, meta: { system, ..., time_seconds } }
    """
    t0    = time.time()
    saved = []
    try:
        for upload in files:
            dest = UPLOAD_DIR / f"{uuid.uuid4().hex}_{upload.filename}"
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            saved.append(str(dest))

        if saved:
            answer, meta = answer_media(saved, caption=message)
        else:
            if not message.strip():
                return JSONResponse({"error": "No message or files provided."}, status_code=400)
            answer, meta = answer_text(message)

        meta["time_seconds"] = round(time.time() - t0, 2)
        return {"answer": answer, "meta": meta}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        for path in saved:
            try: os.remove(path)
            except OSError: pass


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GymBot API server")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--host",   type=str, default="127.0.0.1")
    parser.add_argument("--reload", action="store_true", help="Dev hot-reload")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  GYMBOT API")
    print("="*50)
    print(f"  Hardware : {HW_CONFIG.get('hw')}")
    print(f"  Device   : {HW_CONFIG.get('device')}")
    print(f"  Model    : {HW_CONFIG.get('model')}")
    print(f"  URL      : http://{args.host}:{args.port}")
    print("="*50)
    print("  Endpoints:")
    print("    GET  /info")
    print("    POST /chat")
    print("    POST /analyse")
    print("    POST /chat-with-media   ← use this one")
    print("="*50 + "\n")

    uvicorn.run(
        "gymbot_api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
