import os
import anyio
from typing import Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

os.environ.setdefault("SPOTTER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
os.environ.setdefault("DEVICE_MAP", "cpu")
os.environ.setdefault("LOAD_IN_4BIT", "0")

import hybrid_orchestrator
import memory_manager
import supabase_service
from supabase_client import get_user_from_token

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_up():
    print("Warming up orchestrator + RAG...")
    await anyio.to_thread.run_sync(hybrid_orchestrator.get_orchestrator)
    print("Warmup complete.")


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    # --- 1. Identify user from JWT (optional — works without login too) ---
    user_id = None
    thread_id = req.thread_id

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        user = get_user_from_token(token)
        if user:
            user_id = user.id

    # --- 2. Get or create conversation thread ---
    if user_id:
        thread = supabase_service.get_or_create_thread(user_id, thread_id)
        thread_id = thread["id"]

    # --- 3. Load message history and build context block ---
    history = []
    context_block = ""
    if thread_id:
        history = supabase_service.get_history_for_context(thread_id, max_messages=10)
        context_block = memory_manager.build_context_block(thread_id, user_id, history)

    # --- 4. Save user message ---
    if thread_id:
        supabase_service.save_message(thread_id, role="user", content=req.message)

    # --- 5. Build prompt with context and run the model ---
    prompt = req.message
    if context_block:
        prompt = f"{context_block}\n\nCurrent question: {req.message}"

    answer = await anyio.to_thread.run_sync(
        lambda p: hybrid_orchestrator.smart_answer(p), prompt
    )

    # --- 6. Save assistant response ---
    if thread_id:
        supabase_service.save_message(thread_id, role="assistant", content=answer)

    return {"answer": answer, "thread_id": thread_id}
