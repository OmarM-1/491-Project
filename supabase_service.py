"""
supabase_service.py — Database Operations Layer

All Supabase queries live here. The API layer (gymbot_api.py) calls these
functions — it never touches supabase directly.

Sections:
  1. Auth dependency (for FastAPI endpoint protection)
  2. Profile operations
  3. Conversation + message operations
  4. Memory operations (personal RAG)
  5. Session context builder
  6. Workout & meal plan operations
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, Request, status

from supabase_client import (
    get_supabase,
    get_supabase_admin,
    get_user_from_token,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# 1. AUTH DEPENDENCY — protects FastAPI endpoints
# ═════════════════════════════════════════════════════════════

async def get_current_user(request: Request):
    """
    FastAPI dependency — extracts and validates the JWT from the
    Authorization header. Use on any endpoint that requires login:

        @app.post("/chat")
        async def chat(user = Depends(get_current_user)):
            user_id = user.id   # UUID from Supabase Auth
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected: Bearer <token>",
        )

    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please sign in again.",
        )

    return user


# ═════════════════════════════════════════════════════════════
# 2. PROFILE OPERATIONS
# ═════════════════════════════════════════════════════════════

def get_profile(user_id: str) -> dict | None:
    """Fetch the user's full profile."""
    admin = get_supabase_admin()
    result = (
        admin.table("profiles")
        .select("*")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )
    return result.data


def update_profile(user_id: str, updates: dict) -> dict:
    """
    Update profile fields. Pass only the fields you want to change:
        update_profile(uid, {"weight_kg": 82.5, "fitness_goal": "build_muscle"})
    """
    admin = get_supabase_admin()
    result = (
        admin.table("profiles")
        .update(updates)
        .eq("id", user_id)
        .execute()
    )
    return result.data[0] if result.data else {}


# ═════════════════════════════════════════════════════════════
# 3. CONVERSATION + MESSAGE OPERATIONS
# ═════════════════════════════════════════════════════════════

def create_thread(user_id: str, title: str = None) -> dict:
    """Start a new conversation thread."""
    admin = get_supabase_admin()
    result = (
        admin.table("conversation_threads")
        .insert({"user_id": user_id, "title": title})
        .execute()
    )
    return result.data[0]


def get_thread(thread_id: str, user_id: str) -> dict | None:
    """Fetch a thread, ensuring it belongs to the user."""
    admin = get_supabase_admin()
    result = (
        admin.table("conversation_threads")
        .select("*")
        .eq("id", thread_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    return result.data


def get_or_create_thread(user_id: str, thread_id: str = None) -> dict:
    """
    If thread_id is provided and valid, return it.
    Otherwise create a new thread.
    """
    if thread_id:
        existing = get_thread(thread_id, user_id)
        if existing:
            return existing
    return create_thread(user_id)


def list_threads(user_id: str, limit: int = 20) -> list[dict]:
    """Fetch the user's most recent conversation threads."""
    admin = get_supabase_admin()
    result = (
        admin.table("conversation_threads")
        .select("id, title, summary, topics, message_count, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def save_message(
    thread_id: str,
    role: str,
    content: str,
    model_used: str = None,
    token_count: int = None,
    latency_ms: int = None,
    has_image: bool = False,
    image_path: str = None,
    retrieved_context: dict = None,
) -> dict:
    """Insert a message into a conversation thread."""
    admin = get_supabase_admin()

    row = {
        "thread_id": thread_id,
        "role": role,
        "content": content,
    }
    if model_used:
        row["model_used"] = model_used
    if token_count is not None:
        row["token_count"] = token_count
    if latency_ms is not None:
        row["latency_ms"] = latency_ms
    if has_image:
        row["has_image"] = True
        row["image_path"] = image_path
    if retrieved_context:
        row["retrieved_context"] = retrieved_context

    result = admin.table("messages").insert(row).execute()

    # Bump the thread's message count and updated_at
    admin.rpc("", {}).execute  # updated_at trigger handles timestamp
    admin.table("conversation_threads").update(
        {"message_count": get_message_count(thread_id)}
    ).eq("id", thread_id).execute()

    return result.data[0] if result.data else {}


def get_messages(thread_id: str, limit: int = 50) -> list[dict]:
    """Fetch messages in a thread, ordered chronologically."""
    admin = get_supabase_admin()
    result = (
        admin.table("messages")
        .select("id, role, content, has_image, image_path, created_at, model_used")
        .eq("thread_id", thread_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_message_count(thread_id: str) -> int:
    """Count messages in a thread."""
    admin = get_supabase_admin()
    result = (
        admin.table("messages")
        .select("id", count="exact")
        .eq("thread_id", thread_id)
        .execute()
    )
    return result.count or 0


def get_history_for_context(thread_id: str, max_messages: int = 10) -> list[dict]:
    """
    Get recent messages formatted for the LLM context window.
    Returns [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    admin = get_supabase_admin()
    result = (
        admin.table("messages")
        .select("role, content")
        .eq("thread_id", thread_id)
        .order("created_at", desc=True)
        .limit(max_messages)
        .execute()
    )
    # Reverse so oldest is first (LLM expects chronological order)
    messages = result.data or []
    messages.reverse()
    return messages


def save_thread_summary(
    thread_id: str,
    summary: str,
    topics: list[str] = None,
    user_sentiment: str = None,
):
    """
    Called after a session ends — stores the LLM-generated summary.
    Your backend generates this by passing the full conversation
    to Qwen with a summarization prompt.
    """
    admin = get_supabase_admin()
    updates = {"summary": summary}
    if topics:
        updates["topics"] = topics
    if user_sentiment:
        updates["user_sentiment"] = user_sentiment

    admin.table("conversation_threads").update(updates).eq("id", thread_id).execute()


# ═════════════════════════════════════════════════════════════
# 4. MEMORY OPERATIONS (Personal RAG)
# ═════════════════════════════════════════════════════════════

def store_memory(
    user_id: str,
    content: str,
    category: str = "other",
    embedding: list[float] = None,
    confidence: float = 1.0,
    source_thread_id: str = None,
) -> dict:
    """
    Insert a new memory fact about the user.
    Call after extracting facts from a completed conversation.
    
    Before calling this, use find_similar_memories() via RPC to check
    for duplicates — see deduplicate_and_store_memory() below.
    """
    admin = get_supabase_admin()
    row = {
        "user_id": user_id,
        "content": content,
        "category": category,
        "confidence": confidence,
    }
    if embedding:
        row["embedding"] = embedding
    if source_thread_id:
        row["source_thread_id"] = source_thread_id

    result = admin.table("user_memories").insert(row).execute()
    return result.data[0] if result.data else {}


def search_memories(
    user_id: str,
    query_embedding: list[float],
    match_count: int = 5,
    threshold: float = 0.65,
    categories: list[str] = None,
) -> list[dict]:
    """
    Semantic search over the user's memories.
    Call this at query time to pull relevant context.
    """
    admin = get_supabase_admin()
    params = {
        "query_embedding": query_embedding,
        "target_user_id": user_id,
        "match_count": match_count,
        "match_threshold": threshold,
    }
    if categories:
        params["filter_categories"] = categories

    result = admin.rpc("match_user_memories", params).execute()
    return result.data or []


def find_duplicates(
    user_id: str,
    embedding: list[float],
    similarity_cutoff: float = 0.92,
) -> list[dict]:
    """
    Check if a memory already exists before inserting.
    Returns similar existing memories above the cutoff.
    """
    admin = get_supabase_admin()
    result = admin.rpc("find_similar_memories", {
        "query_embedding": embedding,
        "target_user_id": user_id,
        "similarity_cutoff": similarity_cutoff,
    }).execute()
    return result.data or []


def supersede_memory(old_memory_id: str, new_memory: dict) -> dict:
    """
    Mark an old memory as inactive and insert the updated version.
    Used when a fact changes (e.g., weight updated, new goal set).
    """
    admin = get_supabase_admin()

    # Deactivate old
    admin.table("user_memories").update(
        {"is_active": False}
    ).eq("id", old_memory_id).execute()

    # Insert new with reference to old
    new_memory["supersedes_id"] = old_memory_id
    result = admin.table("user_memories").insert(new_memory).execute()
    return result.data[0] if result.data else {}


def deduplicate_and_store_memory(
    user_id: str,
    content: str,
    category: str,
    embedding: list[float],
    confidence: float = 1.0,
    source_thread_id: str = None,
) -> dict:
    """
    High-level helper: check for duplicates, then either skip, supersede, or insert.
    This is the function your post-session pipeline should call.
    
    Returns {"action": "skipped"|"superseded"|"inserted", "memory": {...}}
    """
    duplicates = find_duplicates(user_id, embedding)

    if duplicates:
        best_match = duplicates[0]
        if best_match["similarity"] > 0.97:
            # Near-identical — skip
            logger.info(f"Memory skipped (duplicate): {content[:60]}...")
            return {"action": "skipped", "memory": best_match}
        else:
            # Similar but different — this is an update
            logger.info(f"Memory superseded: {best_match['content'][:40]}... → {content[:40]}...")
            new = supersede_memory(
                old_memory_id=best_match["id"],
                new_memory={
                    "user_id": user_id,
                    "content": content,
                    "category": category,
                    "embedding": embedding,
                    "confidence": confidence,
                    "source_thread_id": source_thread_id,
                },
            )
            return {"action": "superseded", "memory": new}
    else:
        # Brand new fact
        logger.info(f"Memory stored: {content[:60]}...")
        new = store_memory(
            user_id=user_id,
            content=content,
            category=category,
            embedding=embedding,
            confidence=confidence,
            source_thread_id=source_thread_id,
        )
        return {"action": "inserted", "memory": new}


def get_all_memories(user_id: str) -> list[dict]:
    """Fetch all active memories for a user, grouped by category."""
    admin = get_supabase_admin()
    result = (
        admin.table("user_memories")
        .select("id, category, content, confidence, created_at")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .order("category")
        .order("confidence", desc=True)
        .execute()
    )
    return result.data or []


# ═════════════════════════════════════════════════════════════
# 5. SESSION CONTEXT — one call to load everything
# ═════════════════════════════════════════════════════════════

def build_session_context(user_id: str) -> dict:
    """
    Call at the start of every conversation.
    Returns a dict with profile, active schedule, meal plan,
    all memories, and recent session summaries.
    
    Inject this into the system prompt so GymBot knows the user.
    """
    admin = get_supabase_admin()
    result = admin.rpc("build_session_context", {
        "target_user_id": user_id
    }).execute()
    return result.data or {}


def get_recent_summaries(user_id: str, count: int = 5) -> list[dict]:
    """Fetch the last N conversation summaries for continuity."""
    admin = get_supabase_admin()
    result = admin.rpc("get_recent_summaries", {
        "target_user_id": user_id,
        "num_sessions": count,
    }).execute()
    return result.data or []


# ═════════════════════════════════════════════════════════════
# 6. WORKOUT & MEAL PLAN OPERATIONS
# ═════════════════════════════════════════════════════════════

def get_active_schedule(user_id: str) -> dict | None:
    """Fetch the user's active workout schedule with all days and exercises."""
    admin = get_supabase_admin()
    result = (
        admin.table("workout_schedules")
        .select("*, workout_days(*, exercises(*))")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .maybe_single()
        .execute()
    )
    return result.data


def get_active_meal_plan(user_id: str) -> dict | None:
    """Fetch the user's active meal plan with all meals."""
    admin = get_supabase_admin()
    result = (
        admin.table("meal_plans")
        .select("*, meals(*)")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .maybe_single()
        .execute()
    )
    return result.data


def create_workout_schedule(user_id: str, name: str, description: str = None) -> dict:
    """Create a new workout schedule (deactivates any existing active one)."""
    admin = get_supabase_admin()

    # Deactivate current active schedule
    admin.table("workout_schedules").update(
        {"is_active": False}
    ).eq("user_id", user_id).eq("is_active", True).execute()

    result = (
        admin.table("workout_schedules")
        .insert({
            "user_id": user_id,
            "name": name,
            "description": description,
            "is_active": True,
        })
        .execute()
    )
    return result.data[0] if result.data else {}


def create_meal_plan(user_id: str, name: str, **macros) -> dict:
    """Create a new meal plan (deactivates any existing active one)."""
    admin = get_supabase_admin()

    # Deactivate current active plan
    admin.table("meal_plans").update(
        {"is_active": False}
    ).eq("user_id", user_id).eq("is_active", True).execute()

    row = {"user_id": user_id, "name": name, "is_active": True}
    for key in ("total_calories", "total_protein_g", "total_carbs_g", "total_fat_g"):
        if key in macros:
            row[key] = macros[key]

    result = admin.table("meal_plans").insert(row).execute()
    return result.data[0] if result.data else {}
