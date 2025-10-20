"""
Cores 1–6 RAG for GymBot with dynamic handoff to Spotter_AI.chat_text:
1) Knowledge base + chunking
2) Hybrid retrieval (embeddings+FAISS + BM25)
3) Cross-encoder reranker
4) Confidence meter
5) Prompt/spec
6) End-to-end generate_grounded_answer(query)
"""

from __future__ import annotations
import os, re, json, math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# --- Optional model wrapper import (no circular deps) ---
try:
    from Spotter_AI import chat_text, detect_intent, build_messages  # generator
except Exception:
    chat_text = None
    detect_intent = None
    build_messages = None

# =========================
# CORE 1 — Knowledge Base (sample; call init_kb(--kb) to use your own)
# =========================
SAMPLE_DOCS = [
    {
        "id": "ex_db_press",
        "type": "exercise",
        "title": "Dumbbell Flat Press",
        "text": (
            "Dumbbell Flat Press (chest emphasis). "
            "Cues: set scapula (retract/depress), elbows ~45° from torso, soft lockout, steady tempo. "
            "Common mistakes: flared elbows, bouncing, losing upper-back tension. "
            "Progressions: tempo (3s down), heavier load; Regressions: incline push-up."
        ),
        "metadata": {"muscles":["chest"], "equipment":"dumbbells", "level":"beginner", "source_id":"kb://exercises/db_press"}
    },
    {
        "id": "ex_bench_press",
        "type": "exercise",
        "title": "Barbell Bench Press",
        "text": (
            "Barbell Bench Press (Primary: chest; Secondary: triceps, anterior delts). "
            "Cues: retract/depress scapula, slight arch, feet planted, bar path to lower chest, wrists over elbows. "
            "Common mistakes: elbows flared >75°, bouncing bar, losing upper-back tension. "
            "Progressions: pause bench; Regressions: dumbbell floor press."
        ),
        "metadata": {"muscles":["chest","triceps"], "equipment":"barbell", "level":"intermediate", "source_id":"kb://exercises/bench"}
    },
    {
        "id": "primer_rir",
        "type": "primer",
        "title": "RIR (Reps In Reserve)",
        "text": (
            "RIR = reps you could still perform at the end of a set. "
            "0 RIR = maximal effort; 2 RIR = you stopped with two reps left. "
            "Hypertrophy guidance commonly uses 0–3 RIR depending on movement and fatigue."
        ),
        "metadata": {"topic":"RIR", "source_id":"kb://concepts/rir"}
    },
    {
        "id": "tmpl_push_45",
        "type": "program",
        "title": "Push Day — 45 min Hypertrophy Template",
        "text": (
            "Template: 4–6 exercises; 8–15 reps; 60–90s rest; aim 1–3 RIR. "
            "Flow: Horizontal press → Incline press → Overhead press → Lateral raise → Triceps isolation."
        ),
        "metadata": {"goal":"hypertrophy","day":"push","duration_min":45, "source_id":"kb://templates/push45"}
    },
    {
        "id": "policy_safety",
        "type": "policy",
        "title": "Safety Note",
        "text": (
            "This information is educational and not medical advice. "
            "Consult a qualified professional for injuries or medical conditions."
        ),
        "metadata": {"source_id":"kb://policy/safety"}
    },
]

@dataclass
class Document:
    id: str
    type: str
    title: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class Chunk:
    id: str
    parent_id: str
    text: str
    metadata: Dict[str, Any]

def load_kb(jsonl_path: Optional[str]) -> List[Document]:
    if jsonl_path and os.path.isfile(jsonl_path):
        docs = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docs.append(Document(
                    id=obj["id"], type=obj["type"], title=obj["title"],
                    text=obj["text"], metadata=obj.get("metadata", {})
                ))
        return docs
    return [Document(**d) for d in SAMPLE_DOCS]

# =========================
# CORE 2 — Chunking + Metadata
# =========================
def normalize_text(t: str) -> str:
    import re as _re
    t = _re.sub(r"\\s+", " ", t).strip()
    return t

def chunk_docs(docs: List[Document], chunk_size: int = 420, overlap: int = 60) -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0
    for d in docs:
        full = normalize_text(f"{d.title}. {d.text}")
        toks = full.split()
        start = 0
        while start < len(toks):
            end = min(len(toks), start + chunk_size)
            piece = " ".join(toks[start:end])
            chunks.append(Chunk(
                id=f"{d.id}::chunk{cid}",
                parent_id=d.id,
                text=piece,
                metadata={**d.metadata, "type": d.type, "title": d.title}
            ))
            cid += 1
            if end == len(toks): break
            start = max(0, end - overlap)
    return chunks

# =========================
# CORE 3 — Hybrid Retrieval (FAISS + BM25) + metadata pre-filter
# =========================
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_emb_model = None
_faiss = None
_chunk_embs = None
_bm25 = None
_tokenized_corpus = None
_chunks: List[Chunk] = []
_reranker = None
_INDEX_READY = False

def build_indexes(chunks: List[Chunk]):
    global _emb_model, _faiss, _chunk_embs, _bm25, _tokenized_corpus, _chunks, _reranker, _INDEX_READY
    _chunks = chunks
    _emb_model = SentenceTransformer(EMB_MODEL)
    texts = [c.text for c in chunks]
    _chunk_embs = _emb_model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    d = _chunk_embs.shape[1]
    _faiss = faiss.IndexFlatIP(d)  # cosine via normalized inner product
    _faiss.add(_chunk_embs)
    _tokenized_corpus = [t.lower().split() for t in texts]
    _bm25 = BM25Okapi(_tokenized_corpus)
    _reranker = CrossEncoder(RERANKER)
    _INDEX_READY = True

def init_kb(kb_path: Optional[str] = None, *, chunk_size: int = 420, overlap: int = 60):
    docs = load_kb(kb_path)
    chunks = chunk_docs(docs, chunk_size=chunk_size, overlap=overlap)
    build_indexes(chunks)

def rr_fusion(list1: List[int], list2: List[int], k: int = 60) -> List[int]:
    scores: Dict[int, float] = defaultdict(float)
    for r, i in enumerate(list1): scores[i] += 1.0/(k+r+1)
    for r, i in enumerate(list2): scores[i] += 1.0/(k+r+1)
    return [i for i,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def parse_profile_from_query(q: str) -> Dict[str, Any]:
    t = q.lower()
    equip = []
    for w in ["dumbbell","dumbbells","barbell","machine","cable","bodyweight","kettlebell","bands"]:
        if w in t: equip.append("dumbbells" if w.startswith("dumbbell") else
                                "barbell" if w=="barbell" else
                                "bodyweight" if w=="bodyweight" else w)
    goal = "hypertrophy" if any(k in t for k in ["hypertrophy","muscle","mass","size"]) else \
           "strength" if "strength" in t else None
    day = None
    for d in ["push","pull","legs","upper","lower","full body","full-body","fullbody"]:
        if d in t:
            day = d.replace(" ", "")
            break
    m = re.search(r"(\\d{2,3})\\s*(?:min|minutes?)", t)
    duration_min = int(m.group(1)) if m else None
    return {"equipment": set(equip), "goal": goal, "day": day, "duration_min": duration_min}

def metadata_match(idx: int, prefs: Dict[str,Any]) -> bool:
    md = _chunks[idx].metadata
    if prefs.get("equipment"):
        eq = md.get("equipment")
        if eq and eq not in prefs["equipment"]:
            return False
    if prefs.get("goal"):
        if md.get("goal") and md.get("goal") != prefs["goal"]:
            return False
    if prefs.get("day"):
        if md.get("day") and md.get("day") != prefs["day"]:
            return False
    return True

def retrieval_params(query: str) -> Tuple[int,int,int]:
    L = len(query.split())
    if L < 6:     return 24, 24, 12
    if L < 12:    return 16, 16, 10
    return 12, 12, 10

def hybrid_candidates(query: str, prefs: Dict[str,Any] | None = None):
    global _INDEX_READY
    if not _INDEX_READY:
        init_kb(None)  # build from SAMPLE_DOCS by default
    prefs = prefs or {}
    top_vec, top_kw, pool = retrieval_params(query)
    q_emb = _emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims, ids = _faiss.search(__import__("numpy").expand_dims(q_emb, 0), top_vec)
    vec_ids = ids[0].tolist()
    kw_scores = _bm25.get_scores(query.lower().split())
    kw_ids = list(__import__("numpy").argsort(kw_scores)[::-1][:top_kw])
    merged = rr_fusion(vec_ids, kw_ids, k=60)
    filtered = [i for i in merged if metadata_match(i, prefs)]
    cand_ids = filtered if filtered else merged
    return cand_ids[:max(pool, top_vec, top_kw)], q_emb, kw_scores

# =========================
# CORE 4 — Cross-Encoder Reranker
# =========================
def rerank(query: str, cand_ids: List[int], keep_top: int = 5, pool_extra: int = 10) -> List[Dict[str,Any]]:
    pairs = [(query, _chunks[i].text) for i in cand_ids[:max(keep_top+pool_extra, len(cand_ids))]]
    scores = __import__("numpy").asarray(_reranker.predict(pairs)).tolist()
    ranked = sorted(zip(cand_ids[:len(pairs)], scores), key=lambda x: x[1], reverse=True)
    out = []
    for i, sc in ranked[:keep_top]:
        c = _chunks[i]
        out.append({
            "idx": i,
            "chunk_id": c.id,
            "parent_id": c.parent_id,
            "text": c.text,
            "title": c.metadata.get("title"),
            "type": c.metadata.get("type"),
            "rerank_score": float(sc)
        })
    return out

# =========================
# CORE 5 — Confidence Meter
# =========================
def softmax(xs):
    if not xs: return []
    m = max(xs); exps = [math.exp(x - m) for x in xs]; s = sum(exps) or 1.0
    return [e/s for e in exps]

def compute_confidence(reranked: List[Dict[str,Any]], kw_scores, q_emb, k: int = 5):
    top = reranked[:k]
    if not top:
        return 0.0, {"p_top":0.0, "gap":0.0, "coverage":0.0, "density":0.0, "bm25_flag":0.0}
    cosims, bm25_flag = [], 0
    for h in top:
        emb = _chunk_embs[h["idx"]]
        cos = float(__import__("numpy").dot(q_emb, emb))
        h["cos_sim"] = cos
        h["bm25_score"] = float(kw_scores[h["idx"]])
        cosims.append(cos)
        if h["bm25_score"] >= 1.5:
            bm25_flag = 1
    probs = softmax([h["rerank_score"] for h in top])
    p_top = probs[0]
    gap = p_top - (probs[1] if len(probs) > 1 else 0.0)
    coverage = len({h["parent_id"] for h in top})/len(top)
    density = sum(cosims)/len(cosims)
    conf = 0.45*p_top + 0.25*gap + 0.15*coverage + 0.10*density + 0.05*bm25_flag
    conf = max(0.0, min(1.0, conf))
    return conf, {"p_top":p_top, "gap":gap, "coverage":coverage, "density":density, "bm25_flag":float(bm25_flag)}

def conf_bucket(c: float) -> str:
    if c >= 0.70: return "high"
    if c >= 0.55: return "medium"
    return "low"

# =========================
# CORE 6 — Prompt (“spec”) + Generation
# =========================
SYS_RULES = (
    "You are GymBot, a helpful fitness assistant.\n"
    "Answer ONLY using the provided context chunks; do not invent facts.\n"
    "Cite chunk IDs like [doc::chunkN] after the statements they support.\n"
    "If the user mentions pain/injury, include the safety note if present.\n"
    "Write concise, actionable guidance (bullets OK)."
)

def _format_with_context(user_query: str, top_chunks: list[dict]) -> str:
    ctx = "\\n\\n".join(f"[{c['chunk_id']}] {c['text']}" for c in top_chunks)
    return (
        f"User question: {user_query}\\n\\n"
        f"Context:\\n{ctx}\\n\\n"
        f"Now answer the user. If information is missing, say so briefly."
    )

if build_messages is None:
    def build_messages(system: str, user: str) -> list[dict]:
        return [{"role":"system","content":system},{"role":"user","content":user}]

def injury_flag(q: str) -> bool:
    t = q.lower()
    return any(k in t for k in ["injury","pain","hurt","impingement","strain","sprain","tendonitis","tendinitis","back pain","shoulder pain","knee pain"])

def get_safety_chunk() -> Optional[Dict[str,Any]]:
    for i, c in enumerate(_chunks):
        title = str(c.metadata.get("title","")).lower()
        if c.metadata.get("type") == "policy" or "safety" in title:
            return {
                "idx": i,
                "chunk_id": c.id,
                "parent_id": c.parent_id,
                "text": c.text,
                "title": c.metadata.get("title"),
                "type": c.metadata.get("type","policy"),
                "rerank_score": 0.0
            }
    return None

# --- Public entrypoint ---
def generate_grounded_answer(plain_user_query: str, top_k: int = 5, seed: int | None = None, kb_path: Optional[str] = None) -> dict:
    """
    Run cores 1–6 and return an answer grounded on your KB.
    If kb_path is provided on first call, we load and build indexes from it.
    """
    global _INDEX_READY
    if not _INDEX_READY:
        init_kb(kb_path)

    prefs = parse_profile_from_query(plain_user_query)
    cand_ids, q_emb, kw_scores = hybrid_candidates(plain_user_query, prefs)
    reranked = rerank(plain_user_query, cand_ids, keep_top=top_k)

    if injury_flag(plain_user_query):
        safety = get_safety_chunk()
        if safety and all(safety["chunk_id"] != c["chunk_id"] for c in reranked):
            reranked = [safety] + reranked[:-1] if len(reranked) >= top_k else [safety] + reranked

    confidence, breakdown = compute_confidence(reranked, kw_scores, q_emb, k=min(top_k, len(reranked)))
    bucket = conf_bucket(confidence)

    user_block = _format_with_context(plain_user_query, reranked)
    messages = build_messages(SYS_RULES, user_block)

    if chat_text is None or detect_intent is None:
        return {
            "answer": "(Generator unavailable — grounded prompt follows)\\n\\n" + user_block,
            "intent": "knowledge",
            "confidence": confidence,
            "confidence_bucket": bucket,
            "confidence_breakdown": breakdown,
            "contexts": [{"id": c["chunk_id"], "title": c.get("title"), "type": c.get("type")} for c in reranked],
        }

    intent = detect_intent(plain_user_query)
    answer = chat_text(messages, intent=intent, confidence=confidence, seed=seed)

    return {
        "answer": answer,
        "intent": intent,
        "confidence": confidence,
        "confidence_bucket": bucket,
        "confidence_breakdown": breakdown,
        "contexts": [{"id": c["chunk_id"], "title": c.get("title"), "type": c.get("type")} for c in reranked],
    }
