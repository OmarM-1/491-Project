"""
memory_manager.py — Claude-style sliding window memory

How it works (mirrors Claude's context management):

  PROMPT STRUCTURE ON EVERY CALL:
  ┌─────────────────────────────────────────┐
  │  LONG-TERM: Past session summaries      │  ← from Supabase (all past threads)
  │  ROLLING:   Summary of older turns      │  ← generated when current chat > limit
  │  RECENT:    Last N full messages        │  ← always included verbatim
  │  CURRENT:   User's new question         │
  └─────────────────────────────────────────┘

  When the RECENT messages exceed TOKEN_BUDGET:
    1. Take the oldest half of RECENT messages
    2. Summarize them with the LLM → append to ROLLING summary
    3. Store updated ROLLING summary back to the thread in Supabase
    4. Drop those old messages from the window
    → Token count stays bounded forever
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

# Max tokens allowed for history (recent messages) before summarizing.
# ~600 tokens ≈ 6-8 average exchanges. Keeps total prompt well under 2K.
HISTORY_TOKEN_BUDGET = 600

# How many recent messages to always keep verbatim (never summarize these)
MIN_RECENT_MESSAGES = 4

# How many past-session summaries to load at the start of each chat
PAST_SESSIONS_TO_LOAD = 3


# ─────────────────────────────────────────────────────────────
# Token estimation (no processor needed — fast approximation)
# ─────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """~0.75 words per token for English fitness text."""
    return max(1, len(text.split()) * 4 // 3)

def _history_token_count(messages: list[dict]) -> int:
    return sum(_estimate_tokens(m.get("content", "")) for m in messages)


# ─────────────────────────────────────────────────────────────
# Summarization (uses the local Qwen model)
# ─────────────────────────────────────────────────────────────

def _summarize_messages(messages: list[dict]) -> str:
    """Ask the LLM to compress a block of messages into a short summary."""
    from Spotter_AI import chat_text, build_messages

    turns = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Spotter AI'}: {m['content']}"
        for m in messages
    )

    msgs = build_messages(
        system=(
            "You are a summarizer for a fitness chatbot. "
            "Summarize the conversation below into 2-4 bullet points. "
            "Keep every specific fact: numbers, exercises, goals, injuries. "
            "Be brief but lose nothing important."
        ),
        user=f"Conversation to summarize:\n{turns}\n\nSummary:",
    )

    return chat_text(msgs, max_new_tokens=200, temperature=0.2).strip()


# ─────────────────────────────────────────────────────────────
# Rolling summary — stored on the thread row in Supabase
# ─────────────────────────────────────────────────────────────

def _get_rolling_summary(thread_id: str) -> str:
    """Load the current rolling summary for this thread (may be empty)."""
    try:
        from supabase_client import get_supabase_admin
        admin = get_supabase_admin()
        result = (
            admin.table("conversation_threads")
            .select("summary")
            .eq("id", thread_id)
            .maybe_single()
            .execute()
        )
        return (result.data or {}).get("summary") or ""
    except Exception as e:
        logger.warning(f"Could not load rolling summary: {e}")
        return ""


def _save_rolling_summary(thread_id: str, summary: str):
    """Persist the updated rolling summary back to the thread row."""
    try:
        from supabase_client import get_supabase_admin
        admin = get_supabase_admin()
        admin.table("conversation_threads").update(
            {"summary": summary}
        ).eq("id", thread_id).execute()
    except Exception as e:
        logger.warning(f"Could not save rolling summary: {e}")


# ─────────────────────────────────────────────────────────────
# Past-session summaries — from other threads
# ─────────────────────────────────────────────────────────────

def _get_past_session_summaries(user_id: str, current_thread_id: str) -> str:
    """
    Load summaries of past threads (not the current one).
    Returns a formatted string ready to inject into the prompt.
    """
    if not user_id:
        return ""
    try:
        from supabase_service import get_recent_summaries
        summaries = get_recent_summaries(user_id, count=PAST_SESSIONS_TO_LOAD)
        # Filter out current thread
        summaries = [s for s in summaries if s.get("id") != current_thread_id]
        if not summaries:
            return ""
        lines = []
        for s in summaries:
            if s.get("summary"):
                lines.append(f"- {s['summary']}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Could not load past summaries: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def build_context_block(
    thread_id: Optional[str],
    user_id: Optional[str],
    history: list[dict],
) -> str:
    """
    Given the current thread's full message history, return a formatted
    context block to inject into the system/user prompt.

    Automatically summarizes old messages when the token budget is exceeded.

    Args:
        thread_id:  Supabase thread ID (None = no persistence)
        user_id:    Supabase user ID (None = no cross-session memory)
        history:    Full list of {"role": ..., "content": ...} for current thread

    Returns:
        A formatted string with sections:
          [PAST SESSIONS] / [EARLIER IN THIS CHAT] / [RECENT MESSAGES]
    """
    if not history and not thread_id:
        return ""

    sections = []

    # ── 1. Past session summaries (cross-session long-term memory) ────
    if user_id and thread_id:
        past = _get_past_session_summaries(user_id, thread_id)
        if past:
            sections.append(f"[PAST SESSIONS]\n{past}")

    # ── 2. Sliding window — summarize if over budget ───────────────────
    rolling_summary = _get_rolling_summary(thread_id) if thread_id else ""
    recent = list(history)  # copy

    if _history_token_count(recent) > HISTORY_TOKEN_BUDGET:
        # Always preserve the last MIN_RECENT_MESSAGES messages verbatim
        keep_count = max(MIN_RECENT_MESSAGES, len(recent) // 2)
        to_summarize = recent[:-keep_count]
        recent = recent[-keep_count:]

        if to_summarize:
            logger.info(f"Summarizing {len(to_summarize)} older messages...")
            new_chunk_summary = _summarize_messages(to_summarize)

            # Append new chunk to existing rolling summary
            if rolling_summary:
                rolling_summary = rolling_summary + "\n" + new_chunk_summary
            else:
                rolling_summary = new_chunk_summary

            # Persist so next call doesn't re-summarize
            if thread_id:
                _save_rolling_summary(thread_id, rolling_summary)

    if rolling_summary:
        sections.append(f"[EARLIER IN THIS CHAT]\n{rolling_summary}")

    # ── 3. Recent messages (verbatim) ─────────────────────────────────
    if recent:
        lines = []
        for m in recent:
            role = "User" if m["role"] == "user" else "Spotter AI"
            lines.append(f"{role}: {m['content']}")
        sections.append("[RECENT MESSAGES]\n" + "\n".join(lines))

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────
# Post-session: summarize the whole thread for long-term memory
# ─────────────────────────────────────────────────────────────

def finalize_thread(thread_id: str, user_id: str):
    """
    Call this when a thread ends (e.g., user closes chat or starts a new thread).
    Generates a final summary of the whole session and saves it so future
    sessions can reference it via [PAST SESSIONS].
    """
    try:
        from supabase_service import get_messages, save_thread_summary
        messages = get_messages(thread_id)
        if not messages:
            return

        final_summary = _summarize_messages(messages)
        save_thread_summary(thread_id, summary=final_summary)
        logger.info(f"Thread {thread_id} finalized with summary.")
    except Exception as e:
        logger.warning(f"Could not finalize thread: {e}")
