"""
OPTIMIZED RAG - Fixes 5-20 minute response time issue

Key optimizations:
1. Singleton pattern - models load once and stay in memory
2. Lazy loading - only load what's needed
3. FAISS index caching - save/load index to disk
4. Batch processing - process multiple queries efficiently
5. GPU optimization - proper device placement
"""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# =========================
# Global Singletons (CRITICAL for performance!)
# =========================
_EMBEDDER = None
_RERANKER = None
_FAISS_INDEX = None
_CHUNKS = None
_BM25 = None

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
    """Lazy load reranker - only loads once"""
    global _RERANKER
    if _RERANKER is None:
        print("Loading reranker (one-time)...")
        _RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
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
# BM25 (Lightweight)
# =========================
class SimpleBM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc.split()) for doc in corpus) / self.corpus_size
        
        # Build IDF
        from collections import defaultdict, Counter
        import math
        
        self.idf = {}
        doc_freqs = defaultdict(int)
        
        for doc in corpus:
            words = set(doc.lower().split())
            for word in words:
                doc_freqs[word] += 1
        
        for word, freq in doc_freqs.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def score(self, query: str, doc_idx: int) -> float:
        from collections import Counter
        
        doc = self.corpus[doc_idx]
        doc_len = len(doc.split())
        
        score = 0.0
        query_words = query.lower().split()
        doc_words = doc.lower().split()
        doc_word_counts = Counter(doc_words)
        
        for word in query_words:
            if word not in self.idf:
                continue
            
            word_count = doc_word_counts.get(word, 0)
            numerator = self.idf[word] * word_count * (self.k1 + 1)
            denominator = word_count + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            
            score += numerator / denominator
        
        return score
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# =========================
# FAISS Index with Caching
# =========================
def get_cache_path(kb_path: str) -> str:
    """Generate cache filename based on KB hash"""
    with open(kb_path, 'rb') as f:
        kb_hash = hashlib.md5(f.read()).hexdigest()[:8]
    return f'.cache_faiss_{kb_hash}.pkl'

def build_faiss_index(chunks: List[Chunk], embedder: SentenceTransformer, cache_path: str):
    """Build FAISS index with caching"""
    global _FAISS_INDEX
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading FAISS index from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                _FAISS_INDEX = pickle.load(f)
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
        batch_size=32  # Faster batching
    )
    embeddings = np.array(embeddings).astype('float32')
    
    # Build index
    _FAISS_INDEX = faiss.IndexFlatL2(embeddings.shape[1])
    _FAISS_INDEX.add(embeddings)
    
    # Save to cache
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(_FAISS_INDEX, f)
        print(f"✅ FAISS index cached to: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not cache index: {e}")

# =========================
# Optimized RAG System
# =========================
class OptimizedGymBotRAG:
    """Performance-optimized RAG with caching and lazy loading"""
    
    def __init__(self, kb_path: str = 'fitness_knowledge_base.jsonl', force_rebuild: bool = False):
        global _CHUNKS, _BM25, _FAISS_INDEX
        
        print("\n🚀 Initializing Optimized RAG System...")
        
        # Load data
        print("Loading knowledge base...")
        docs = load_kb(kb_path)
        _CHUNKS = chunk_docs(docs)
        print(f"✅ Loaded {len(docs)} docs → {len(_CHUNKS)} chunks")
        
        # Lazy load models
        embedder = get_embedder()
        
        # Build/load FAISS index with caching
        cache_path = get_cache_path(kb_path)
        if force_rebuild and os.path.exists(cache_path):
            os.remove(cache_path)
            print("Force rebuilding index...")
        
        build_faiss_index(_CHUNKS, embedder, cache_path)
        
        # Build BM25 (fast, no caching needed)
        print("Building BM25 index...")
        chunk_texts = [c.text for c in _CHUNKS]
        _BM25 = SimpleBM25(chunk_texts)
        
        print("✅ RAG system ready!\n")
    
    def retrieve(self, query: str, k: int = 6) -> Tuple[List[Dict], float]:
        """
        Hybrid retrieve + rerank
        Returns: (docs, confidence)
        """
        global _CHUNKS, _BM25, _FAISS_INDEX
        
        if _CHUNKS is None or _BM25 is None or _FAISS_INDEX is None:
            raise RuntimeError("RAG not initialized!")
        
        embedder = get_embedder()
        reranker = get_reranker()
        
        # 1. FAISS semantic search
        query_embedding = embedder.encode([query])[0].astype('float32').reshape(1, -1)
        faiss_distances, faiss_indices = _FAISS_INDEX.search(query_embedding, k * 2)
        
        # 2. BM25 keyword search
        bm25_results = _BM25.search(query, k=k * 2)
        
        # 3. Merge results
        seen = set()
        merged = []
        
        # Add FAISS results
        for idx, dist in zip(faiss_indices[0], faiss_distances[0]):
            if idx not in seen:
                chunk = _CHUNKS[idx]
                merged.append({
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'score': float(1.0 / (1.0 + dist)),
                    'source': 'faiss',
                    'metadata': chunk.metadata
                })
                seen.add(idx)
        
        # Add BM25 results
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
        
        # 4. Cross-encoder reranking
        pairs = [[query, doc['text']] for doc in merged[:k * 3]]
        scores = reranker.predict(pairs)
        
        for doc, score in zip(merged[:k * 3], scores):
            doc['rerank_score'] = float(score)
        
        merged.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        top_docs = merged[:k]
        
        # 5. Calculate confidence
        if top_docs and 'rerank_score' in top_docs[0]:
            top_scores = [doc['rerank_score'] for doc in top_docs[:3]]
            confidence = min(0.99, max(0.0, sum(top_scores) / len(top_scores)))
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

