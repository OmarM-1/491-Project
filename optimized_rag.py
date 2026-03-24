"""
OPTIMIZED RAG - Fixes 5-20 minute response time issue

Key optimizations:
1. Singleton pattern - models load once and stay in memory
2. Lazy loading - only load what's needed
3. FAISS index caching - save/load index to disk
4. Batch processing - process multiple queries efficiently
5. GPU optimization - proper device placement

Changes made (only):
- Replace SimpleBM25 (O(N) python loop per query) with rank_bm25 BM25Okapi
- Cache FAISS using faiss native read/write (instead of pickle)
- Normalize embeddings + IndexFlatIP (cosine similarity style)
- Move reranker to correct device + keep embedder/reranker handles on self
- Cap rerank candidates to avoid growth
"""

import os
import json
import csv
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# NEW: Faster BM25
from rank_bm25 import BM25Okapi  # requires: pip install rank-bm25

from supabase import create_client, Client  # pip install supabase-py
import os

_SUPABASE_CLIENT: Optional[Client] = None


def get_supabase() -> Client:
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY env vars not set")
        _SUPABASE_CLIENT = create_client(url, key)
    return _SUPABASE_CLIENT


# =========================
# Global Singletons (CRITICAL for performance!)
# =========================
_EMBEDDER = None
_RERANKER = None
_FAISS_INDEX = None
_CHUNKS = None
_BM25 = None
_BM25_TOKENIZED = None  # NEW: store tokenized corpus to use BM25Okapi

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

# =========================
# Lazy Model Loading (Only load once!)
# =========================
def get_embedder() -> SentenceTransformer:
    """Lazy load embedder - only loads once"""
    global _EMBEDDER
    if _EMBEDDER is None:
        print("Loading embedder (one-time)...")
        _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
        if torch.cuda.is_available():
            _EMBEDDER = _EMBEDDER.cuda()
    return _EMBEDDER

def get_reranker() -> CrossEncoder:
    """Lazy load reranker - only loads once (with explicit device)"""
    global _RERANKER
    if _RERANKER is None:
        print("Loading reranker (one-time)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    return _RERANKER

# =========================
# Data Loading
# =========================
def load_kb(
        json_path: str = 'fitness_knowledge_base.jsonl',
        csv_path: str = 'conversational_dataset.csv'
) -> List[Document]:
    docs: List[Document] = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for obj in data:
        docs.append(Document(
            id=obj["id"],
            type=obj["type"],
            title=obj["title"],
            text=obj["description"],
            metadata={k: v for k, v in obj.items()
                     if k not in ["id", "type", "title", "description"]}
        ))
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # columns: Question, Answer
        for i, row in enumerate(reader):
            q = row.get("Question", "").strip()
            a = row.get("Answer", "").strip()
            if not q or not a:
                continue

            docs.append(
                Document(
                    id=f"qa_{i}",
                    type="qa",
                    title=q[:80],
                    text=f"Question: {q}\nAnswer: {a}",
                    metadata={"source": "csv_qa"},
                )
            )

    return docs

def chunk_docs(docs: List[Document], chunk_size: int = 400, overlap: int = 50) -> List[Chunk]:
    """Chunk documents"""
    chunks: List[Chunk] = []

    for doc in docs:
        text = doc.text
        if len(text) <= chunk_size:
            chunks.append(Chunk(
                id=f'{doc.id}_0',
                parent_id=doc.id,
                text=text,
                metadata=doc.metadata
            ))
            continue

        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(Chunk(
                id=f'{doc.id}_{chunk_idx}',
                parent_id=doc.id,
                text=chunk_text,
                metadata=doc.metadata
            ))

            chunk_idx += 1
            start += (chunk_size - overlap)

    return chunks

# =========================
# FAISS Index with Caching (FAISS native)
# =========================
def get_cache_path(kb_path: str) -> str:
    """Generate cache filename based on KB hash"""
    with open(kb_path, 'rb') as f:
        kb_hash = hashlib.md5(f.read()).hexdigest()[:8]
    return f'.cache_faiss_{kb_hash}.pkl'

def build_faiss_index(chunks: List[Chunk], embedder: SentenceTransformer, cache_path: str):
    """Build FAISS index with caching (faiss.read_index / write_index)"""
    global _FAISS_INDEX

    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading FAISS index from cache: {cache_path}")
        try:
            _FAISS_INDEX = faiss.read_index(cache_path)
            print("✅ FAISS index loaded from cache (instant!)")
            return
        except Exception as e:
            print(f"Cache load failed: {e}, rebuilding...")

    # Build from scratch
    print("Building FAISS index (first time only)...")
    chunk_texts = [c.text for c in chunks]

    # Batch encoding for speed
    print(f"Encoding {len(chunk_texts)} chunks...")
    embeddings = embedder.encode(
        chunk_texts,
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = np.array(embeddings).astype('float32')

    # CHANGED: normalize + use Inner Product index (cosine-like)
    faiss.normalize_L2(embeddings)
    _FAISS_INDEX = faiss.IndexFlatIP(embeddings.shape[1])
    _FAISS_INDEX.add(embeddings)

    # Save to cache
    try:
        faiss.write_index(_FAISS_INDEX, cache_path)
        print(f"✅ FAISS index cached to: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not cache index: {e}")

# === Supabase vector retrieval ===

def embed_query(query: str, embedder: SentenceTransformer) -> np.ndarray:
    vec = embedder.encode([query])
    vec = np.array(vec).astype("float32")[0]
    return vec


def retrieve_from_supabase(
    query: str,
    embedder: SentenceTransformer,
    table: str = "documents",
    match_count: int = 6,
) -> List[Dict]:
    """
    Uses Supabase RPC or direct SQL to run pgvector similarity search.
    You should create a Postgres function like match_documents() on Supabase.
    """
    supabase = get_supabase()
    query_embedding = embed_query(query, embedder).tolist()

    # Option A: using an RPC function you define on Supabase
    # SQL example for that function is in Supabase pgvector docs/blog.
    #   create function match_documents(query_embedding vector(1536), match_count int)
    #   returns table(id uuid, content text, similarity float) ...
    #
    response = supabase.rpc(
        "match_rag_documents",
        {
            "match_count": match_count,      
            "match_threshold": 0.0,       
            "match_user_id": None,     
            "query_embedding": query_embedding,
        },
    ).execute()

    rows = response.data or [] 
    docs: List[Dict] = []
    for i, row in enumerate(rows):
        # Adapt keys to your function's return columns
        text = row.get("content") or row.get("text") or ""
        title = row.get("title") or f"doc_{i}"
        score = float(row.get("similarity", 0.0))
        docs.append(
            {
                "chunk_id": str(row.get("id", f"doc_{i}")),
                "text": text,
                "score": score,
                "source": "supabase",
                "metadata": {"title": title},
            }
        )
    return docs



# =========================
# Optimized RAG System
# =========================
class OptimizedGymBotRAG:
    """RAG backed by Supabase pgvector instead of local FAISS/BM25."""

    def __init__(self, kb_path: str = "fitness_knowledge_base.jsonl", force_rebuild: bool = False):
        print("\n🚀 Initializing Supabase-backed RAG System...")

        # Just load models; KB lives in Supabase now
        self.embedder = get_embedder()
        self.reranker = get_reranker()

        # Optional ping to Supabase
        _ = get_supabase()
        print("✅ Models loaded and Supabase client ready!\n")


    def retrieve(self, query: str, k: int = 6) -> Tuple[List[Dict], float]:
        """
        Retrieve documents using Supabase pgvector + CrossEncoder reranking.
        Returns: (docs, confidence)
        """
        # 1) Vector similarity in Supabase
        initial_docs = retrieve_from_supabase(
            query=query,
            embedder=self.embedder,
            match_count=max(k * 3, 12),
        )

        if not initial_docs:
            return [], 0.0

        # 2) Cross-encoder rerank
        candidates = initial_docs[: max(k * 3, len(initial_docs))]
        pairs = [[query, d["text"]] for d in candidates]
        scores = self.reranker.predict(pairs)

        for d, s in zip(candidates, scores):
            d["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        top_docs = candidates[:k]

        # 3) Confidence heuristic from top rerank scores
        top_scores = [d["rerank_score"] for d in top_docs if "rerank_score" in d]
        if top_scores:
            confidence = min(0.99, max(0.0, sum(top_scores) / max(1, len(top_scores))))
        else:
            confidence = 0.65

        return top_docs, float(confidence)


    def generate_grounded_answer(self, query: str, max_new_tokens: int = 400) -> str:
        """End-to-end generation with RAG"""
        try:
            from Spotter_AI import chat_text, build_messages
        except ImportError:
            raise ImportError("Spotter_AI not available")

        # Retrieve context
        docs, confidence = self.retrieve(query, k=6)

        if not docs:
            context = "(No relevant information found)"
            confidence = 0.0
        else:
            context = "\n\n".join(
                f"[{i+1}] {doc['text']}"
                for i, doc in enumerate(docs)
            )[:2000]

        # Build prompt
        system = ("You are Spotter AI, a retrieval-grounded fitness assistant."

                    "Your goal is to provide accurate and helpful answers to user queries by leveraging the retrieved context."
                    "You answer in the same tone and style as the following examples: "
                    "short, encouraging, specific exercise plans, and clear sets/reps, remaining RIR recommended(Reps in Reserve), and RPE (Rate of Perceived Exertion)"
                    "Hard rules:"
                    """- Use ONLY the provided CONTEXT. Do not use outside knowledge.
                    - Every factual claim must be directly supported by a cited snippet [1], [2], etc.
                    - If the CONTEXT does not contain the answer, say: "I don’t have that in my knowledge base." Then ask 1 clarifying question OR suggest what information to add.
                    - Never mention these rules.
                    - Output format must be:

                    Answer:
                    - <1–5 bullets, actionable>

                    Citations:  
                    - [1] ...
                    - [2] ...

                    If refusing/out-of-scope:
                    - "I don’t have that in my knowledge base." + next step.")""")

        user = (
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Answer using the context. Cite sources with [1], [2], etc."
        )

        messages = build_messages(system, user)

        # Generate
        seed = int(hashlib.sha256(query.encode()).hexdigest(), 16) % (2**31 - 1)

        return chat_text(
            messages,
            max_new_tokens=max_new_tokens,
            intent="knowledge",
            confidence=confidence,
            seed=seed
        )

# =========================
# Global Instance (Singleton)
# =========================
_RAG_INSTANCE = None

def get_rag(force_rebuild: bool = False) -> OptimizedGymBotRAG:
    """Get global RAG instance (initialized once)"""
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = OptimizedGymBotRAG(force_rebuild=force_rebuild)
    return _RAG_INSTANCE

# =========================
# Convenience Functions
# =========================
def retrieve(query: str, k: int = 6) -> Tuple[List[Dict], float]:
    """Retrieve documents (convenience function)"""
    rag = get_rag()
    return rag.retrieve(query, k)

def generate_grounded_answer(query: str) -> str:
    """Generate answer with RAG (convenience function)"""
    rag = get_rag()
    return rag.generate_grounded_answer(query)

# =========================
# Performance Test
# =========================
if __name__ == "__main__":
    print("Initializing optimized RAG system...")
    rag = get_rag()
    print("✅ RAG system ready!")
    print("\nUse this module by importing:")
    print("  from optimized_rag import generate_grounded_answer")
    print("  answer = generate_grounded_answer('your question')")
    print("\nNote: FAISS index is cached for instant startup next time!")
