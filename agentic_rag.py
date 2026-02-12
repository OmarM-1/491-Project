# agentic_rag.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re

from optimized_rag import get_rag, retrieve, generate_grounded_answer

@dataclass
class AgenticRAGAgent:
    rag: object
    verbose: bool = False

_AGENT: Optional[AgenticRAGAgent] = None

def get_agent(verbose: bool = False) -> AgenticRAGAgent:
    """Hybrid orchestrator imports this."""
    global _AGENT
    if _AGENT is None:
        _AGENT = AgenticRAGAgent(rag=get_rag(), verbose=verbose)
    return _AGENT

def _decompose(query: str) -> List[str]:
    """
    Cheap decomposition:
    Split multi-part questions into sub-questions.
    """
    parts = re.split(r"\b(?:and|then|also|plus)\b", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [query]

def agentic_answer(agent: AgenticRAGAgent, query: str, k: int = 6) -> str:
    """
    Agentic flow:
    1) Decompose into subquestions (if multi-part)
    2) Retrieve evidence per subquestion
    3) Do ONE final generation using combined evidence

    This stays fast: only one final LLM call.
    """
    subqs = _decompose(query)

    # If it's not multi-part, just use your fast end-to-end generator
    if len(subqs) == 1:
        return generate_grounded_answer(query)

    evidence_blocks = []
    for i, sq in enumerate(subqs, 1):
        docs, _conf = retrieve(sq, k=max(3, k // 2))
        if docs:
            context = "\n".join([f"[{j+1}] {d['text']}" for j, d in enumerate(docs)])
        else:
            context = "(No relevant context found)"
        evidence_blocks.append(f"SUBQUESTION {i}: {sq}\nCONTEXT:\n{context}\n")

    synthesis_prompt = (
        f"User question:\n{query}\n\n"
        "You have evidence from multiple retrieval steps.\n"
        "Use it to answer clearly and accurately.\n"
        "Cite sources like [1], [2] within each context block.\n\n"
        + "\n---\n".join(evidence_blocks)
    )

    return generate_grounded_answer(synthesis_prompt)
