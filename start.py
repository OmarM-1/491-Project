#!/usr/bin/env python3
"""
GymBot — Production Launcher

Boots the full system in one command:
  • FastAPI server (API + auth + chat + profile endpoints)
  • RAG pipeline (standard + agentic)
  • Vision model (photo/video form analysis)
  • Supabase database (profiles, memories, conversations)

Usage:
  python start.py                  # Production: start everything
  python start.py --port 3001      # Custom port
  python start.py --host 0.0.0.0   # Expose to network
  python start.py --dev            # Dev mode: interactive menu, hot-reload
  python start.py --dev rag        # Dev: test RAG only
  python start.py --dev agentic    # Dev: test agentic RAG only
  python start.py --dev cli        # Dev: terminal chat (no server)

Environment (.env):
  SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
  GYMBOT_PORT        (default: 8000)
  GYMBOT_HOST        (default: 127.0.0.1)
  GYMBOT_LOG_LEVEL   (default: info)
  GYMBOT_WORKERS     (default: 1)
"""

import os
import sys
import signal
import time
import platform
import logging
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Logging — set up before anything else
# ─────────────────────────────────────────────────────────────

LOG_LEVEL = os.environ.get("GYMBOT_LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gymbot")


# ─────────────────────────────────────────────────────────────
# Hardware Detection (no interactive prompts)
# ─────────────────────────────────────────────────────────────

def detect_hardware() -> dict:
    """Detect hardware and configure environment. Returns config dict."""
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == "Darwin" and machine in ("arm64", "aarch64")

    config = {"system": system, "machine": machine}

    if is_apple_silicon:
        os.environ.setdefault("DEVICE_MAP", "mps")
        os.environ.setdefault("LOAD_IN_4BIT", "false")
        os.environ.setdefault("SPOTTER_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
        config.update(device="MPS", model="Qwen2-VL-2B", hw="Apple Silicon")

    elif system == "Darwin":
        os.environ.setdefault("DEVICE_MAP", "cpu")
        os.environ.setdefault("LOAD_IN_4BIT", "false")
        os.environ.setdefault("SPOTTER_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
        config.update(device="CPU", model="Qwen2-VL-2B", hw="Intel Mac")

    else:
        try:
            import torch
            if torch.cuda.is_available():
                os.environ.setdefault("DEVICE_MAP", "auto")
                os.environ.setdefault("LOAD_IN_4BIT", "true")
                os.environ.setdefault("SPOTTER_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
                vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                config.update(
                    device="CUDA", model="Qwen2-VL-7B",
                    hw=torch.cuda.get_device_name(0),
                    vram_gb=round(vram, 1),
                )
            else:
                raise RuntimeError("No CUDA")
        except Exception:
            os.environ.setdefault("DEVICE_MAP", "cpu")
            os.environ.setdefault("LOAD_IN_4BIT", "false")
            os.environ.setdefault("SPOTTER_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
            config.update(device="CPU", model="Qwen2-VL-2B", hw="CPU")

    return config


# ─────────────────────────────────────────────────────────────
# Supabase Connection Check
# ─────────────────────────────────────────────────────────────

def check_database() -> dict:
    """Check Supabase connectivity. Returns status dict."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    url = os.environ.get("SUPABASE_URL", "")
    anon = os.environ.get("SUPABASE_ANON_KEY", "")
    service = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not all([url, anon, service]):
        missing = []
        if not url: missing.append("SUPABASE_URL")
        if not anon: missing.append("SUPABASE_ANON_KEY")
        if not service: missing.append("SUPABASE_SERVICE_ROLE_KEY")
        return {"connected": False, "reason": f"Missing: {', '.join(missing)}"}

    try:
        from supabase_client import check_connection, get_db_info
        if check_connection():
            info = get_db_info()
            return {"connected": True, **info}
        else:
            return {"connected": False, "reason": "Connection test failed"}
    except Exception as e:
        return {"connected": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────
# AI System Preloading
# ─────────────────────────────────────────────────────────────

def preload_ai_systems() -> dict:
    """
    Load all AI subsystems eagerly at startup instead of on first request.
    Returns a status dict of what loaded successfully.
    """
    status = {
        "rag": False,
        "agentic_rag": False,
        "vision": False,
        "safety": False,
    }

    # ── RAG Pipeline ──────────────────────────────────────
    try:
        logger.info("Loading RAG pipeline...")
        t0 = time.time()
        from optimized_rag import generate_grounded_answer  # noqa: F401
        status["rag"] = True
        logger.info(f"  RAG ready ({time.time() - t0:.1f}s)")
    except ImportError as e:
        logger.warning(f"  RAG unavailable: {e}")
    except Exception as e:
        logger.error(f"  RAG failed to load: {e}")

    # ── Agentic RAG ───────────────────────────────────────
    try:
        logger.info("Loading agentic RAG...")
        t0 = time.time()
        from agentic_rag import agentic_answer  # noqa: F401
        status["agentic_rag"] = True
        logger.info(f"  Agentic RAG ready ({time.time() - t0:.1f}s)")
    except ImportError as e:
        logger.warning(f"  Agentic RAG unavailable: {e}")
    except Exception as e:
        logger.error(f"  Agentic RAG failed to load: {e}")

    # ── Vision Model ──────────────────────────────────────
    try:
        logger.info("Loading vision model...")
        t0 = time.time()
        from Spotter_AI import chat_vision, load_image, load_video  # noqa: F401
        status["vision"] = True
        logger.info(f"  Vision model ready ({time.time() - t0:.1f}s)")
    except ImportError as e:
        logger.warning(f"  Vision unavailable: {e}")
    except Exception as e:
        logger.error(f"  Vision model failed to load: {e}")

    # ── Safety Gate ───────────────────────────────────────
    try:
        from SAFETY_AGENT import safety_gate_agent  # noqa: F401
        status["safety"] = True
        logger.info("  Safety gate loaded")
    except ImportError:
        logger.warning("  Safety gate unavailable (SAFETY_AGENT.py not found)")
    except Exception as e:
        logger.error(f"  Safety gate failed: {e}")

    return status


# ─────────────────────────────────────────────────────────────
# FastAPI App Builder
# ─────────────────────────────────────────────────────────────

def build_app(hw_config: dict, db_status: dict, ai_status: dict):
    """Construct the FastAPI application with all routes."""
    from fastapi import FastAPI, File, Form, UploadFile, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import shutil
    import tempfile
    import uuid as _uuid

    app = FastAPI(
        title="GymBot API",
        version="2.0",
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5500",
            "http://localhost:5500",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="gymbot_"))

    # ── Health ─────────────────────────────────────────────

    @app.get("/")
    async def health():
        return {
            "status": "running",
            "version": "2.0",
            "model": hw_config.get("model"),
            "device": hw_config.get("device"),
            "database": "connected" if db_status.get("connected") else "disconnected",
            "systems": {k: v for k, v in ai_status.items()},
        }

    @app.get("/health")
    async def health_detailed():
        return {
            "hardware": hw_config,
            "database": db_status,
            "ai_systems": ai_status,
        }

    # ── Auth (requires database) ──────────────────────────

    if db_status.get("connected"):
        from supabase_client import sign_up, sign_in

        class AuthRequest(BaseModel):
            email: str
            password: str
            username: str | None = None

        @app.post("/auth/signup")
        async def signup(req: AuthRequest):
            try:
                result = sign_up(req.email, req.password, req.username)
                session = result.session
                return {
                    "success": True,
                    "user_id": result.user.id,
                    "access_token": session.access_token if session else None,
                }
            except Exception as e:
                return JSONResponse({"success": False, "error": str(e)}, status_code=400)

        @app.post("/auth/login")
        async def login(req: AuthRequest):
            try:
                result = sign_in(req.email, req.password)
                return {
                    "success": True,
                    "user_id": result.user.id,
                    "access_token": result.session.access_token,
                    "refresh_token": result.session.refresh_token,
                    "expires_in": result.session.expires_in,
                }
            except Exception as e:
                return JSONResponse({"success": False, "error": str(e)}, status_code=401)

    # ── Vision helper (shared by /chat and /analyse) ─────

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
    VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}

    def _analyse_media(file_paths: list[str], caption: str = "") -> tuple[str, dict]:
        """
        Run vision analysis on uploaded files.
        Uses Spotter_AI directly — the orchestrator doesn't handle vision.
        """
        from Spotter_AI import chat_vision, load_image, load_video

        images_list = []
        video_paths = []
        image_paths = []

        for path in file_paths:
            ext = Path(path).suffix.lower()
            if ext in VIDEO_EXTS:
                video_paths.append(path)
            elif ext in IMAGE_EXTS:
                image_paths.append(path)

        for vp in video_paths:
            frames = load_video(vp, max_frames=8)
            images_list.extend(frames)

        for ip in image_paths:
            images_list.append(load_image(ip))

        if not images_list:
            return "No valid media found to analyse.", {"system": "vision"}

        n_vids = len(video_paths)
        n_imgs = len(image_paths)
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
                "Keep the tone honest but encouraging."
            )
        elif n_vids:
            prompt = (
                f"Analyse this exercise video ({len(images_list)} frames).{user_note}\n\n"
                "1. **Exercise** – what is being performed?\n"
                "2. **Form Rating** – 1-10\n"
                "3. **Rep Breakdown** – which reps show form deterioration?\n"
                "4. **Key Issues** – top 2-3 problems\n"
                "5. **Corrections** – specific fixes\n"
                "6. **Safety** – any injury risks?"
            )
        else:
            prompt = (
                f"Analyse this exercise photo.{user_note}\n\n"
                "1. **Exercise** – what movement is shown?\n"
                "2. **Form Rating** – score 1-10 and justify\n"
                "3. **Key Issues** – top 2-3 problems\n"
                "4. **Corrections** – specific, actionable fixes\n"
                "5. **Safety** – any injury risks?\n"
                "Keep the tone honest but encouraging."
            )

        answer = chat_vision(images_list, prompt)
        meta = {
            "system": "vision",
            "files": n_imgs + n_vids,
            "images": n_imgs,
            "videos": n_vids,
        }
        return answer, meta

    # ── Chat (text — JSON body, no file upload headaches) ─

    class ChatRequest(BaseModel):
        message: str
        thread_id: str | None = None

    @app.post("/chat")
    async def chat(req: ChatRequest):
        t0 = time.time()

        if not req.message.strip():
            return JSONResponse(
                {"success": False, "error": "Message cannot be empty"},
                status_code=400,
            )

        try:
            from hybrid_orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            result = orchestrator.answer(req.message, return_metadata=True)

            answer = result.get("answer", "")
            meta = {
                "system": result.get("system", "rag"),
                "complexity_score": result.get("complexity_score"),
                "route_reason": result.get("route_reason"),
                "safe": result.get("safe", True),
                "time_seconds": round(time.time() - t0, 2),
            }

            return {"success": True, "answer": answer, "thread_id": req.thread_id, "meta": meta}

        except Exception as e:
            logger.exception("Chat endpoint error")
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=500,
            )

    # ── Chat with media (multipart — use for images/video) ─

    @app.post("/chat-with-media")
    async def chat_with_media(
        message: str = Form(""),
        thread_id: str = Form(None),
        files: list[UploadFile] = File(...),
    ):
        t0 = time.time()
        saved = []

        for upload in files:
            if not upload.filename:
                continue
            dest = UPLOAD_DIR / f"{_uuid.uuid4().hex}_{upload.filename}"
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            saved.append(str(dest))

        try:
            if not saved:
                return JSONResponse(
                    {"success": False, "error": "No valid files uploaded"},
                    status_code=400,
                )
            if not ai_status.get("vision"):
                return JSONResponse(
                    {"success": False, "error": "Vision model not loaded"},
                    status_code=503,
                )

            answer, meta = _analyse_media(saved, caption=message)
            meta["time_seconds"] = round(time.time() - t0, 2)
            return {"success": True, "answer": answer, "thread_id": thread_id, "meta": meta}

        except Exception as e:
            logger.exception("Chat-with-media endpoint error")
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=500,
            )
        finally:
            for path in saved:
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ── Vision-only endpoint ──────────────────────────────

    @app.post("/analyse")
    async def analyse(
        files: list[UploadFile] = File(...),
        caption: str = Form(""),
    ):
        if not ai_status.get("vision"):
            return JSONResponse(
                {"success": False, "error": "Vision model not loaded"},
                status_code=503,
            )
        t0 = time.time()
        saved = []
        try:
            for upload in files:
                if not upload.filename:
                    continue
                dest = UPLOAD_DIR / f"{_uuid.uuid4().hex}_{upload.filename}"
                with dest.open("wb") as fh:
                    shutil.copyfileobj(upload.file, fh)
                saved.append(str(dest))

            answer, meta = _analyse_media(saved, caption=caption)
            meta["time_seconds"] = round(time.time() - t0, 2)
            return {"success": True, "answer": answer, "meta": meta}

        except Exception as e:
            logger.exception("Analyse endpoint error")
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
        finally:
            for path in saved:
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ── Profile / Context / Threads / Memories (DB required)

    if db_status.get("connected"):
        from supabase_service import (
            get_current_user,
            get_profile,
            update_profile,
            build_session_context,
            list_threads,
            get_messages,
            get_all_memories,
        )

        @app.get("/profile")
        async def profile_get(user=Depends(get_current_user)):
            data = get_profile(user.id)
            if not data:
                return JSONResponse({"error": "Profile not found"}, status_code=404)
            return data

        @app.put("/profile")
        async def profile_update(updates: dict, user=Depends(get_current_user)):
            allowed = {
                "username", "display_name", "age", "height_cm", "weight_kg",
                "body_fat_pct", "sex", "fitness_goal", "experience",
                "equipment_access", "injuries", "activity_level",
                "dietary_prefs", "calorie_target", "protein_target_g",
                "carb_target_g", "fat_target_g", "timezone", "onboarding_done",
            }
            filtered = {k: v for k, v in updates.items() if k in allowed}
            if not filtered:
                return JSONResponse({"error": "No valid fields"}, status_code=400)
            return update_profile(user.id, filtered)

        @app.get("/context")
        async def session_context(user=Depends(get_current_user)):
            return build_session_context(user.id)

        @app.get("/threads")
        async def threads_list(user=Depends(get_current_user)):
            return list_threads(user.id)

        @app.get("/threads/{thread_id}/messages")
        async def thread_messages(thread_id: str, user=Depends(get_current_user)):
            return get_messages(thread_id)

        @app.get("/memories")
        async def memories_list(user=Depends(get_current_user)):
            return get_all_memories(user.id)

    # ── Shutdown cleanup ──────────────────────────────────

    @app.on_event("shutdown")
    async def shutdown_cleanup():
        logger.info("Cleaning up temp files...")
        try:
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        except Exception:
            pass
        logger.info("Shutdown complete.")

    return app


# ─────────────────────────────────────────────────────────────
# Startup Banner
# ─────────────────────────────────────────────────────────────

def print_banner(hw: dict, db: dict, ai: dict, host: str, port: int):
    ok = lambda v: "✅" if v else "❌"  # noqa: E731
    W = 52  # box width

    def row(label, value, indent=2):
        content = f"{' ' * indent}{label}: {value}"
        return f"│{content:<{W}}│"

    print()
    print(f"┌{'─' * W}┐")
    print(f"│{'GYMBOT v2.0':^{W}}│")
    print(f"│{'AI-Powered Fitness Coaching':^{W}}│")
    print(f"├{'─' * W}┤")
    print(row("Hardware", hw.get("hw", "Unknown")))
    print(row("Device  ", hw.get("device", "?")))
    print(row("Model   ", hw.get("model", "?")))
    if hw.get("vram_gb"):
        print(row("VRAM    ", f"{hw['vram_gb']} GB"))
    print(f"├{'─' * W}┤")

    db_label = "Supabase (PostgreSQL)" if db.get("connected") else "Not connected"
    print(row("Database", f"{ok(db.get('connected'))}  {db_label}"))
    if not db.get("connected") and db.get("reason"):
        print(row("        ", f"   {db['reason'][:40]}"))

    print(f"├{'─' * W}┤")
    print(row("RAG         ", f"{ok(ai.get('rag'))}  Standard pipeline"))
    print(row("Agentic RAG ", f"{ok(ai.get('agentic_rag'))}  Complex reasoning"))
    print(row("Vision      ", f"{ok(ai.get('vision'))}  Photo/video analysis"))
    print(row("Safety Gate ", f"{ok(ai.get('safety'))}  Content filtering"))
    print(f"├{'─' * W}┤")

    url = f"http://{host}:{port}"
    print(row("Server ", url))
    print(row("Docs   ", f"{url}/docs"))
    print(f"├{'─' * W}┤")

    endpoints = [
        ("POST", "/chat", "Text chat (JSON)"),
        ("POST", "/chat-with-media", "Chat + images/video"),
        ("POST", "/analyse", "Vision-only analysis"),
        ("GET ", "/health", "System status"),
    ]
    if db.get("connected"):
        endpoints.extend([
            ("POST", "/auth/signup", "Create account"),
            ("POST", "/auth/login", "Sign in → JWT"),
            ("GET ", "/profile", "User profile"),
            ("PUT ", "/profile", "Update profile"),
            ("GET ", "/context", "Session context"),
            ("GET ", "/threads", "Conversation list"),
            ("GET ", "/memories", "User memories"),
        ])

    for method, path, desc in endpoints:
        line = f"  {method} {path:<28} {desc}"
        print(f"│{line:<{W}}│")

    print(f"└{'─' * W}┘")
    print()


# ─────────────────────────────────────────────────────────────
# Dev Mode (--dev flag)
# ─────────────────────────────────────────────────────────────

def run_dev_mode(hw_config: dict, sub_mode: str = None):
    """Old interactive/testing modes, preserved behind --dev."""
    import subprocess

    if sub_mode is None:
        print()
        print("  GymBot Dev Mode")
        print("  ───────────────")
        print("  1. Interactive terminal chat")
        print("  2. Test RAG pipeline")
        print("  3. Test agentic RAG")
        print("  4. Vision demo")
        print("  5. Complete system demo")
        print("  6. API server with hot-reload")
        try:
            choice = input("\n  Choice (1-6): ").strip()
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            return
        sub_mode = {
            "1": "cli", "2": "rag", "3": "agentic",
            "4": "vision", "5": "complete", "6": "server",
        }.get(choice, "cli")

    if sub_mode == "cli":
        from hybrid_orchestrator import interactive_mode
        interactive_mode()
    elif sub_mode == "rag":
        subprocess.run([sys.executable, "optimized_rag.py"])
    elif sub_mode == "agentic":
        subprocess.run([sys.executable, "agentic_rag.py"])
    elif sub_mode == "vision":
        subprocess.run([sys.executable, "vision_demo.py"])
    elif sub_mode == "complete":
        subprocess.run([sys.executable, "complete_gymbot.py", "demo"])
    elif sub_mode == "server":
        # Dev server with hot-reload
        import uvicorn
        db_status = check_database()
        ai_status = preload_ai_systems()
        app = build_app(hw_config, db_status, ai_status)
        port = int(os.environ.get("GYMBOT_PORT", "8000"))
        host = os.environ.get("GYMBOT_HOST", "127.0.0.1")
        print_banner(hw_config, db_status, ai_status, host, port)
        print("  🔄 Dev mode — hot-reload enabled\n")
        uvicorn.run(app, host=host, port=port, reload=False, log_level="debug")
    else:
        print(f"  Unknown dev mode: {sub_mode}")
        print("  Options: cli, rag, agentic, vision, complete, server")


# ─────────────────────────────────────────────────────────────
# Graceful Shutdown Handler
# ─────────────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\n  Forced shutdown.")
        sys.exit(1)
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    print(f"\n\n  Received {sig_name} — shutting down gracefully...")
    print("  (Press Ctrl+C again to force quit)\n")
    raise SystemExit(0)


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GymBot — AI Fitness Coach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python start.py                  Start production server
  python start.py --port 3001      Custom port
  python start.py --host 0.0.0.0   Expose to network
  python start.py --dev            Dev mode (interactive menu)
  python start.py --dev cli        Dev: terminal chat only
  python start.py --dev rag        Dev: test RAG pipeline
  python start.py --dev server     Dev: API with hot-reload
        """,
    )
    parser.add_argument(
        "--host", type=str,
        default=os.environ.get("GYMBOT_HOST", "127.0.0.1"),
        help="Server host (default: 127.0.0.1, use 0.0.0.0 for network)",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("GYMBOT_PORT", "8000")),
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("GYMBOT_WORKERS", "1")),
        help="Number of uvicorn workers (default: 1)",
    )
    parser.add_argument(
        "--dev", nargs="?", const="menu", default=None,
        metavar="MODE",
        help="Dev mode: cli, rag, agentic, vision, complete, server",
    )

    args = parser.parse_args()

    # ── Signal handlers ───────────────────────────────────
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # ── Hardware detection ────────────────────────────────
    hw_config = detect_hardware()

    # ── Dev mode shortcut ─────────────────────────────────
    if args.dev is not None:
        sub = None if args.dev == "menu" else args.dev
        run_dev_mode(hw_config, sub)
        return

    # ══════════════════════════════════════════════════════
    # PRODUCTION BOOT SEQUENCE
    # ══════════════════════════════════════════════════════

    boot_start = time.time()

    print()
    print("  Initializing GymBot...")
    print()

    # Step 1: Database connection
    logger.info("Step 1/3 — Checking database...")
    db_status = check_database()
    if db_status.get("connected"):
        logger.info("  Database connected")
    else:
        logger.warning(f"  Database unavailable: {db_status.get('reason', 'unknown')}")
        logger.warning("  Auth, profiles, and memory endpoints will be disabled")

    # Step 2: Preload AI systems
    logger.info("Step 2/3 — Loading AI systems (this may take 30-60s on first run)...")
    ai_status = preload_ai_systems()

    loaded = sum(1 for v in ai_status.values() if v)
    total = len(ai_status)
    if loaded == 0:
        logger.error("No AI systems loaded — cannot start. Check your model files.")
        sys.exit(1)
    logger.info(f"  {loaded}/{total} systems ready")

    # Step 3: Build application
    logger.info("Step 3/3 — Building API server...")
    app = build_app(hw_config, db_status, ai_status)

    boot_time = round(time.time() - boot_start, 1)
    logger.info(f"  Boot complete in {boot_time}s")

    # ── Banner + launch ───────────────────────────────────
    print_banner(hw_config, db_status, ai_status, args.host, args.port)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=LOG_LEVEL.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
