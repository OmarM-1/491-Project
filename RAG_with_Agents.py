# run_spotter_with_rag.py
from typing import List, Dict, Any, Tuple
import traceback

# --- Your modules ---
from Spotter_AI import chat_text, build_messages, deterministic_seed_from
from SAFETY_AGENT import safety_gate_agent

# Try several RAG entry points so this works with your current RAG.py
try:
    # If your RAG exposes a single end-to-end generator:
    from RAG import generate_grounded_answer as rag_generate
except Exception:
    rag_generate = None

# Optional low-level hooks if you have them (hybrid retrieve + rerank)
try:
    from RAG import hybrid_retrieve as rag_hybrid_retrieve  # returns list[{"text", "score", "source"}]
except Exception:
    rag_hybrid_retrieve = None

try:
    from RAG import cross_encoder_rerank as rag_rerank  # (query, docs, k)->reranked
except Exception:
    rag_rerank = None


def get_rag_context(query: str, k: int = 6, max_chars: int = 2000) -> Tuple[str, float, List[str]]:
    """
    Returns (context_text, confidence, sources).
    Confidence is a scalar in [0,1] that we feed into Spotter to dial sampling.
    """
    # Preferred: explicit retrieve (+rerank) → context
    if rag_hybrid_retrieve is not None:
        try:
            docs = rag_hybrid_retrieve(query, k=k)  # [{'text', 'score'?, 'source'?}, ...]
            if rag_rerank is not None:
                docs = rag_rerank(query, docs, k=k)

            # Heuristic confidence from top doc scores if present; else count-based
            top_scores = [float(d.get("score", 0.0)) for d in docs[:3]]
            conf = min(0.99, max(0.0, (sum(top_scores) / max(1, len(top_scores))) if top_scores else 0.65))

            context = "\n\n".join(f"[{i+1}] {d.get('text','')}" for i, d in enumerate(docs))[:max_chars]
            sources = [str(d.get("source", "")) for d in docs]
            return context, float(conf), sources
        except Exception:
            traceback.print_exc()

    # Fallback: an end-to-end grounded draft (no explicit chunks)
    if rag_generate is not None:
        try:
            draft = rag_generate(query)
            # A conservative default confidence; Spotter will lean more deterministic when lower
            return draft, 0.65, []
        except Exception:
            traceback.print_exc()

    # Last resort: no context
    return "", 0.0, []


def answer(user_query: str, intent: str = "knowledge") -> str:
    """
    One-pass orchestration: Safety → RAG → LLM (Spotter).
    """
    # 1) Safety pre-check. We give the agent a minimal chat callable via Spotter itself.
    ok, warn = safety_gate_agent(
        user_query,
        chat=lambda msgs, **kw: chat_text(msgs, intent="safety", confidence=0.0, seed=0)
    )
    if not ok:
        return warn  # Safety agent decided to stop here

    # 2) Retrieve/Rerank (and/or grounded draft)
    context, confidence, sources = get_rag_context(user_query)

    # 3) Build the final LLM prompt with explicit grounding instructions
    system = (
        "You are Spotter, a precise assistant. "
        "Use the provided CONTEXT to answer. If the CONTEXT is insufficient, say so. "
        "Prefer concrete facts, show your steps briefly, and keep advice actionable."
    )
    user = (
        f"QUESTION:\n{user_query}\n\n"
        f"CONTEXT (retrieved):\n{context if context else '(none)'}\n\n"
        f"INSTRUCTIONS:\n- Cite short bracketed refs like [1], [2] corresponding to context blocks when used.\n"
        f"- If context is missing for a claim, say you lack evidence."
    )
    msgs = build_messages(system, user)

    # 4) Deterministic seeding for stable phrasing with sampling turned on
    seed = deterministic_seed_from(user_query, user_id="spotter_orchestrator")

    # 5) Generate. Spotter will tune temperature from the confidence we pass.
    return chat_text(msgs, intent=intent, confidence=float(confidence), seed=seed)


if __name__ == "__main__":
    try:
        print("Type your question (Ctrl+C to exit):")
        while True:
            q = input("> ").strip()
            if not q:
                continue
            print("\n--- Answer ---")
            print(answer(q))
            print("--------------\n")
    except KeyboardInterrupt:
        pass
