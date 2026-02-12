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
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# NEW: Faster BM25
from rank_bm25 import BM25Okapi  # requires: pip install rank-bm25

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
def load_kb(json_path: str = 'fitness_knowledge_base.jsonl') -> List[Document]:
    """Load knowledge base"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for obj in data:
        docs.append(Document(
            id=obj["id"],
            type=obj["type"],
            title=obj["title"],
            text=obj["description"],
            metadata={k: v for k, v in obj.items()
                     if k not in ["id", "type", "title", "description"]}
        ))
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
    # CHANGED: use FAISS native .index file
    return f'.cache_faiss_{kb_hash}.index'

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

# =========================
# Optimized RAG System
# =========================
class OptimizedGymBotRAG:
    """Performance-optimized RAG with caching and lazy loading"""

    def __init__(self, kb_path: str = 'fitness_knowledge_base.jsonl', force_rebuild: bool = False):
        global _CHUNKS, _BM25, _FAISS_INDEX, _BM25_TOKENIZED

        print("\n🚀 Initializing Optimized RAG System...")

        # Load data
        print("Loading knowledge base...")
        docs = load_kb(kb_path)
        _CHUNKS = chunk_docs(docs)
        print(f"✅ Loaded {len(docs)} docs → {len(_CHUNKS)} chunks")

        # Lazy load models ONCE (but keep handles on self)
        self.embedder = get_embedder()
        self.reranker = get_reranker()

        # Build/load FAISS index with caching
        cache_path = get_cache_path(kb_path)
        if force_rebuild and os.path.exists(cache_path):
            os.remove(cache_path)
            print("Force rebuilding index...")

        build_faiss_index(_CHUNKS, self.embedder, cache_path)

        # CHANGED: Build BM25 using rank_bm25 (fast)
        print("Building BM25 index...")
        chunk_texts = [c.text for c in _CHUNKS]
        _BM25_TOKENIZED = [t.lower().split() for t in chunk_texts]
        _BM25 = BM25Okapi(_BM25_TOKENIZED)

        print("✅ RAG system ready!\n")

    def retrieve(self, query: str, k: int = 6) -> Tuple[List[Dict], float]:
        """
        Hybrid retrieve + rerank
        Returns: (docs, confidence)
        """
        global _CHUNKS, _BM25, _FAISS_INDEX, _BM25_TOKENIZED

        if _CHUNKS is None or _BM25 is None or _FAISS_INDEX is None:
            raise RuntimeError("RAG not initialized!")

        # Use warm handles (no repeated get_* calls)
        embedder = self.embedder
        reranker = self.reranker

        # 1) FAISS semantic search (cosine-like)
        query_embedding = embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        faiss_scores, faiss_indices = _FAISS_INDEX.search(query_embedding, k * 2)

        # 2) BM25 keyword search (fast)
        q_tokens = query.lower().split()
        bm25_scores = _BM25.get_scores(q_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][: (k * 2)]
        bm25_results = [(int(i), float(bm25_scores[i])) for i in top_bm25_idx]

        # 3) Merge results
        seen = set()
        merged = []

        # Add FAISS results
        for idx, score in zip(faiss_indices[0], faiss_scores[0]):
            if idx < 0:
                continue
            if idx not in seen:
                chunk = _CHUNKS[idx]
                merged.append({
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'score': float(score),  # higher is better now (IP similarity)
                    'source': 'faiss',
                    'metadata': chunk.metadata
                })
                seen.add(idx)

        # Add BM25 results
        # normalize BM25 to a smaller range like before
        for idx, score in bm25_results:
            if idx not in seen:
                chunk = _CHUNKS[idx]
                merged.append({
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'score': float(score / 10.0),
                    'source': 'bm25',
                    'metadata': chunk.metadata
                })
                seen.add(idx)

        if not merged:
            return [], 0.0

        # 4) Cross-encoder reranking (CAPPED candidates)
        rerank_n = min(len(merged), max(20, k * 3))
        candidates = merged[:rerank_n]
        pairs = [[query, doc['text']] for doc in candidates]
        scores = reranker.predict(pairs)

        for doc, score in zip(candidates, scores):
            doc['rerank_score'] = float(score)

        merged.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        top_docs = merged[:k]

        # 5) Calculate confidence
        if top_docs and 'rerank_score' in top_docs[0]:
            top_scores = [doc['rerank_score'] for doc in top_docs[:3] if 'rerank_score' in doc]
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
        system = (
            "You are Spotter AI, a helpful fitness assistant. "
            "Use the provided CONTEXT to answer accurately. "
            "Keep advice practical and actionable."
        )

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
