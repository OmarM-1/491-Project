"""
OPTIMIZED RAG - Fixes 5-20 minute response time issue

Key optimizations:
1. Singleton pattern - models load once and stay in memory
2. Lazy loading - only load what's needed
4. Batch processing - process multiple queries efficiently
5. GPU optimization - proper device placement

Changes made (only):
- Replace SimpleBM25 (O(N) python loop per query) with rank_bm25 BM25Okapi
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

# NEW: Faster BM25
from rank_bm25 import BM25Okapi  # requires: pip install rank-bm25

from supabase import create_client, Client  # pip install supabase-py
import os

_SUPABASE_CLIENT: Optional[Client] = None


def get_supabase() -> Client:
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        url = "https://ezowjohfkxvaajeqilkx.supabase.co"
        key = "sb_publishable__ajmqfcEBopf6B-aKLIzuw_XmyvssUq"
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY env vars not set")
        _SUPABASE_CLIENT = create_client(url, key)
    return _SUPABASE_CLIENT


# =========================
# Global Singletons (CRITICAL for performance!)
# =========================
_EMBEDDER = None
_RERANKER = None
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
    """RAG backed by Supabase pgvector"""

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
        # 1) Vector similarity in Supabase — fetch k+2 only, no need for 3x
        initial_docs = retrieve_from_supabase(
            query=query,
            embedder=self.embedder,
            match_count=k + 2,
        )

        if not initial_docs:
            return [], 0.0

        # 2) Cross-encoder rerank on the smaller candidate set
        pairs = [[query, d["text"]] for d in initial_docs]
        scores = self.reranker.predict(pairs)

        for d, s in zip(initial_docs, scores):
            d["rerank_score"] = float(s)

        initial_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        top_docs = initial_docs[:k]

        # 3) Confidence heuristic from top rerank scores
        top_scores = [d["rerank_score"] for d in top_docs if "rerank_score" in d]
        if top_scores:
            confidence = min(0.99, max(0.0, sum(top_scores) / max(1, len(top_scores))))
        else:
            confidence = 0.65

        return top_docs, float(confidence)
    
    def generate_grounded_answer(
        self,
        query: str,
        max_new_tokens: int = 1000,
        history: list = None,
        thread_id: str = None,
        user_id: str = None,
        context_block: str = "",
        profile: dict = None,
    ) -> str:
        """End-to-end generation with RAG"""
        try:
            from Spotter_AI import chat_text
        except ImportError:
            raise ImportError("Spotter_AI not available")

        # Token budget: ~0.75 words per token, leave 20% headroom
        word_budget = int(max_new_tokens * 0.75 * 0.8)

        # Retrieve docs using only the user's actual question (clean embedding)
        docs, confidence = self.retrieve(query, k=6)

        if not docs:
            rag_context = ""
            confidence = 0.0
        else:
            rag_context = "\n\n".join(
                f"[{i+1}] {doc['text']}"
                for i, doc in enumerate(docs)
            )[:2000]

        # Build profile block (placed at TOP of system prompt — small models follow early context best)
        profile_block = ""
        if profile:
            parts = []
            if profile.get("display_name"):
                parts.append(f"Name: {profile['display_name']}")
            if profile.get("age"):
                parts.append(f"Age: {profile['age']}")
            if profile.get("sex"):
                parts.append(f"Sex: {profile['sex']}")
            if profile.get("weight"):
                parts.append(f"Weight: {profile['weight']} lbs")
            if profile.get("height"):
                parts.append(f"Height: {profile['height']} in")
            if profile.get("fitness_goal"):
                parts.append(f"Fitness goal: {profile['fitness_goal']}")
            if parts:
                profile_block = "USER PROFILE:\n" + "\n".join(parts) + "\n\n"

        # RAG context goes in the system prompt — keeps the user message clean
        rag_block = ""
        if rag_context:
            rag_block = f"\n\nRELEVANT FITNESS KNOWLEDGE (cite with [1],[2] etc. when used):\n{rag_context}"

        system = (
            f"{profile_block}"
            f"You are Spotter AI, an expert fitness coach. "
            f"Use the user's profile above to give personalized advice. "
            f"Deny dangerous advice, unproven supplements, and medical diagnoses. "
            f"If asked about anything unrelated to fitness, politely decline. "
            f"Greet the user by name when they say hi/hello. "
            f"Keep responses under {word_budget // 2} words. "
            f"Be precise about exercise equipment: only mention equipment that is genuinely required for the exercise. "
            f"Bodyweight exercises (push-ups, planks, lunges, etc.) require NO equipment — never add barbells or dumbbells to them. "
            f"Format answers in Markdown: bullets for tips, **bold** for exercises, ## for headings."
            f"{rag_block}"
        )

        # Build multi-turn messages: system → last 4 history turns → current question
        messages = [{"role": "system", "content": system}]

        for turn in (history or [])[-4:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})

        temperature = 0.4 if confidence >= 0.6 else 0.5
        return chat_text(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
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

def generate_grounded_answer(query: str, history: list = None, thread_id: str = None, user_id: str = None, profile: dict = None) -> str:
    """Generate answer with RAG (convenience function)"""
    rag = get_rag()
    return rag.generate_grounded_answer(query, history=history, thread_id=thread_id, user_id=user_id, profile=profile)

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
