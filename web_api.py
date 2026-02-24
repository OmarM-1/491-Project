import os
import time
import anyio
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import platform

# Mirror start.py defaults
os.environ.setdefault("SPOTTER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # example; use YOUR intended model
os.environ.setdefault("DEVICE_MAP", "auto")
os.environ.setdefault("LOAD_IN_4BIT", "1")

# If you’re on CPU only, you might want:
os.environ.setdefault("DEVICE_MAP", "cpu")
os.environ.setdefault("LOAD_IN_4BIT", "0")

# 1) Import orchestrator AFTER env setup (see section B)
import hybrid_orchestrator

# 2) Preload at startup
@app.on_event("startup")
async def warm_up():
    print("🔥 Warming up orchestrator + RAG…")
    await anyio.to_thread.run_sync(hybrid_orchestrator.get_orchestrator)
    print("✅ Warmup complete.")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    t0 = time.time()
    print(f"➡️  Chat request: {req.message[:80]}")
    reply = await anyio.to_thread.run_sync(hybrid_orchestrator.smart_answer, req.message)
    print(f"✅ Replied in {time.time()-t0:.2f}s")
    return {"reply": reply}