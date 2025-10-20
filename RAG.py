# gymbot_rag_v2.py
import os, re, json, math, argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# =========================
# CORE 1 — Knowledge Base (sample; supply --kb to use your own)
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
    t = re.sub(r"\s+", " ", t).strip()
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
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_emb_model = None
_faiss = None
_chunk_embs = None
_bm25 = None
_tokenized_corpus = None
_chunks: List[Chunk] = []

def build_indexes(chunks: List[Chunk]):
    global _emb_model, _faiss, _chunk_embs, _bm25, _tokenized_corpus, _chunks
    _chunks = chunks
    _emb_model = SentenceTransformer(EMB_MODEL)
    texts = [c.text for c in chunks]
    _chunk_embs = _emb_model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    d = _chunk_embs.shape[1]
    _faiss = faiss.IndexFlatIP(d)  # cosine via normalized inner product
    _faiss.add(_chunk_embs)
    _tokenized_corpus = [t.lower().split() for t in texts]
    _bm25 = BM25Okapi(_tokenized_corpus)

def rr_fusion(list1: List[int], list2: List[int], k: int = 60) -> List[int]:
    scores: Dict[int, float] = defaultdict(float)
    for r, i in enumerate(list1): scores[i] += 1.0/(k+r+1)
    for r, i in enumerate(list2): scores[i] += 1.0/(k+r+1)
    return [i for i,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# ---- NEW: parse light metadata filters from query ----
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
    m = re.search(r"(\d{2,3})\s*(?:min|minutes?)", t)
    duration_min = int(m.group(1)) if m else None
    return {"equipment": set(equip), "goal": goal, "day": day, "duration_min": duration_min}

def metadata_match(idx: int, prefs: Dict[str,Any]) -> bool:
    md = _chunks[idx].metadata
    # Equipment: if specified, prefer matching; allow pass if doc has no equipment tag
    if prefs["equipment"]:
        eq = md.get("equipment")
        if eq and eq not in prefs["equipment"]:
            return False
    if prefs["goal"]:
        if md.get("goal") and md.get("goal") != prefs["goal"]:
            return False
    if prefs["day"]:
        if md.get("day") and md.get("day") != prefs["day"]:
            return False
    # duration is advisory; we don't hard-filter on it
    return True

def retrieval_params(query: str) -> Tuple[int,int,int]:
    L = len(query.split())
    if L < 6:     return 24, 24, 12   # short/ambiguous → fetch more
    if L < 12:    return 16, 16, 10
    return 12, 12, 10

def hybrid_candidates(query: str, prefs: Dict[str,Any]):
    top_vec, top_kw, pool = retrieval_params(query)
    q_emb = _emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims, ids = _faiss.search(np.expand_dims(q_emb, 0), top_vec)
    vec_ids = ids[0].tolist()
    kw_scores = _bm25.get_scores(query.lower().split())
    kw_ids = list(np.argsort(kw_scores)[::-1][:top_kw])
    merged = rr_fusion(vec_ids, kw_ids, k=60)

    # Metadata filter (soft): if it wipes out everything, fall back to unfiltered
    filtered = [i for i in merged if metadata_match(i, prefs)]
    cand_ids = filtered if filtered else merged
    return cand_ids[:max(pool, top_vec, top_kw)], q_emb, kw_scores

# =========================
# CORE 4 — Cross-Encoder Reranker
# =========================
from sentence_transformers import CrossEncoder
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None
def ensure_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER)

def rerank(query: str, cand_ids: List[int], keep_top: int = 5, pool_extra: int = 10) -> List[Dict[str,Any]]:
    ensure_reranker()
    # score a slightly larger pool before trimming
    pairs = [(query, _chunks[i].text) for i in cand_ids[:max(keep_top+pool_extra, len(cand_ids))]]
    scores = _reranker.predict(pairs).tolist()
    ranked = sorted(zip(cand_ids[:len(pairs)], scores), key=lambda x: x[1], reverse=True)
    out = []
    for i, sc in ranked[:keep_top]:
        out.append({
            "idx": i,
            "chunk_id": _chunks[i].id,
            "parent_id": _chunks[i].parent_id,
            "text": _chunks[i].text,
            "title": _chunks[i].metadata.get("title"),
            "type": _chunks[i].metadata.get("type"),
            "rerank_score": float(sc)
        })
    return out

# =========================
# CORE 5 — Confidence Meter (same formula)
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
        cos = float(np.dot(q_emb, emb))
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
# CORE 6 — Prompt (“spec”) + Generation (Qwen if available)
# =========================
SYS_RULES = (
    "You are GymBot, a helpful fitness assistant.\n"
    "Answer ONLY using the provided context chunks; do not invent facts.\n"
    "Cite chunk IDs like [doc::chunkN] after the statements they support.\n"
    "If the user mentions pain/injury, include the safety note if present.\n"
    "Write concise, actionable guidance (bullets OK)."
)

def build_messages(user_query: str, contexts: List[Dict[str,Any]]) -> List[Dict[str,str]]:
    ctx = "\n\n".join([f"[{c['chunk_id']}] {c['text']}" for c in contexts])
    user = (
        f"User question: {user_query}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Now answer the user. If information is missing, say so briefly."
    )
    return [{"role":"system","content":SYS_RULES},{"role":"user","content":user}]

# Optional: auto-use your Qwen chat if available
_QWEN_CHAT = None
try:
    from Spotter_AI import chat_text as _QWEN_CHAT  # your file
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
_GEN = None
def ensure_generator(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    global _GEN
    if _GEN is not None or _QWEN_CHAT is not None: return
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        _GEN = pipeline("text-generation", model=lm, tokenizer=tok, max_new_tokens=400)
    except Exception:
        _GEN = None

def detect_intent(q: str) -> str:
    t = q.lower()
    if any(k in t for k in ["injury","pain","hurts","impingement","sprain","strain"]): return "safety"
    if any(k in t for k in ["plan","program","template","make me a","session","routine","workout"]): return "plan"
    if any(k in t for k in ["motivate","pep talk","caption","slogan"]): return "creative"
    return "knowledge"

def sampling_params(intent: str, confidence: float) -> Dict[str,Any]:
    if intent in {"knowledge","safety"}: return dict(do_sample=False)
    if confidence < 0.55:              return dict(do_sample=False)
    if intent == "plan":               return dict(do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.05)
    if intent == "creative":           return dict(do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.05)
    return dict(do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.05)

def generate_answer(user_query: str, contexts: List[Dict[str,Any]], intent: str, confidence: float) -> str:
    params = sampling_params(intent, confidence)
    if _QWEN_CHAT is not None:
        msgs = build_messages(user_query, contexts)
        return _QWEN_CHAT(
            msgs,
            max_new_tokens=400,
            do_sample=params.get("do_sample", False),
            temperature=params.get("temperature", 0.0),
            top_p=params.get("top_p", 1.0),
        )
    ensure_generator()
    if _GEN is None:
        # Show grounded prompt so you can paste into any model manually
        ctx_preview = build_messages(user_query, contexts)
        return "(Generator unavailable — grounded prompt follows)\n\n" + json.dumps(ctx_preview, indent=2)
    set_seed(42)
    prompt = SYS_RULES + "\n\nUser: " + build_messages(user_query, contexts)[1]["content"]
    if params["do_sample"]:
        out = _GEN(prompt, do_sample=True, temperature=params["temperature"], top_p=params["top_p"],
                   repetition_penalty=params.get("repetition_penalty", 1.0))[0]["generated_text"]
    else:
        out = _GEN(prompt, do_sample=False)[0]["generated_text"]
    return out[len(prompt):].strip()

# ---- Injury detection: force-include safety note if present ----
def injury_flag(q: str) -> bool:
    t = q.lower()
    return any(k in t for k in ["injury","pain","hurt","impingement","strain","sprain","tendonitis","back pain","shoulder pain"])

def get_safety_chunk() -> Optional[Dict[str,Any]]:
    for i, c in enumerate(_chunks):
        if c.metadata.get("type") == "policy" or "safety" in c.metadata.get("title","").lower():
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

# =========================
# End-to-end RAG
# =========================
def rag_answer(user_query: str, top_k: int = 5) -> Dict[str,Any]:
    prefs = parse_profile_from_query(user_query)
    cand_ids, q_emb, kw_scores = hybrid_candidates(user_query, prefs)
    reranked = rerank(user_query, cand_ids, keep_top=top_k, pool_extra=10)

    # Force include safety note for injury queries
    if injury_flag(user_query):
        safety = get_safety_chunk()
        if safety and all(safety["chunk_id"] != c["chunk_id"] for c in reranked):
            reranked = [safety] + reranked[:-1] if len(reranked) >= top_k else [safety] + reranked

    confidence, breakdown = compute_confidence(reranked, kw_scores, q_emb, k=min(top_k, len(reranked)))
    intent = detect_intent(user_query)
    answer = generate_answer(user_query, reranked, intent=intent, confidence=confidence)

    return {
        "answer": answer,
        "intent": intent,
        "confidence": confidence,
        "confidence_bucket": conf_bucket(confidence),
        "confidence_breakdown": breakdown,
        "contexts": [{"id": c["chunk_id"], "title": c["title"], "type": c["type"]} for c in reranked]
    }

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=str, default=None, help="Path to KB JSONL (id,type,title,text,metadata)")
    parser.add_argument("--query", type=str, required=True, help="User query")
    args = parser.parse_args()

    docs = load_kb(args.kb)
    chunks = chunk_docs(docs, chunk_size=420, overlap=60)
    build_indexes(chunks)

    res = rag_answer(args.query)

    print("\n=== CONFIDENCE ===")
    print(f"{res['confidence']:.2f}  ({res['confidence_bucket']})")
    print(res["confidence_breakdown"])

    print("\n=== CONTEXTS ===")
    for c in res["contexts"]:
        print(c)

    print("\n=== ANSWER ===")
    print(res["answer"])

if __name__ == "__main__":
    import numpy as np
    main()
