"""
Simplified RAG for GymBot - fixes bugs and removes redundant components
Uses: FAISS (semantic) + BM25 (keyword) + Cross-encoder reranking
"""

from __future__ import annotations
import os, json, math, hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

# Core dependencies
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Optional Spotter integration
try:
    from Spotter_AI import chat_text, build_messages
except Exception:
    chat_text = None
    build_messages = None

# =========================
# Data Structures
# =========================
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
# Load Knowledge Base (FIXED)
# =========================
def load_kb(json_path: str = 'fitness_knowledge_base.jsonl') -> List[Document]:
    """
    Load knowledge base from JSON file
    Handles both:
    - JSON array format: [{"id": "1", ...}, {"id": "2", ...}]
    - JSONL format: one JSON object per line
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Knowledge base not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Try to parse as JSON array first
    data = None
    try:
        data = json.loads(content)
        print(f"✅ Loaded as JSON array")
    except json.JSONDecodeError as e:
        # If that fails, try JSONL format (one JSON object per line)
        print(f"⚠️  Not a JSON array, trying JSONL format...")
        try:
            data = []
            for i, line in enumerate(content.strip().split('\n'), 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as line_error:
                        print(f"⚠️  Skipping invalid JSON on line {i}: {line_error}")
            
            if data:
                print(f"✅ Loaded as JSONL format")
        except Exception as jsonl_error:
            print(f"❌ Failed to parse as JSONL: {jsonl_error}")
            print(f"\n❌ Original JSON error: {e}")
            print(f"   Problem at line {e.lineno}, column {e.colno}")
            print(f"\n💡 Try running: python fix_json.py")
            raise
    
    if not data:
        raise ValueError("No data loaded from knowledge base")
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list of documents, got {type(data)}")
    
    docs = []
    for i, obj in enumerate(data):
        # Use .get() with defaults to handle missing fields gracefully
        doc_id = obj.get("id", f"doc_{i}")
        doc_type = obj.get("type", "unknown")
        doc_title = obj.get("title", doc_id)  # Fallback to id if no title
        doc_text = obj.get("description", obj.get("text", ""))  # Try "description" then "text"
        
        # Skip documents with no text content
        if not doc_text:
            print(f"⚠️  Skipping document {doc_id}: no text content")
            continue
        
        docs.append(Document(
            id=doc_id,
            type=doc_type,
            title=doc_title,
            text=doc_text,
            metadata={k: v for k, v in obj.items() 
                     if k not in ["id", "type", "title", "description", "text"]}
        ))
    
    print(f"✅ Loaded {len(docs)} documents from knowledge base")
    return docs

# =========================
# Chunking
# =========================
def chunk_docs(docs: List[Document], chunk_size: int = 400, overlap: int = 50) -> List[Chunk]:
    """Split documents into overlapping chunks"""
    chunks: List[Chunk] = []
    
    for doc in docs:
        text = doc.text
        if len(text) <= chunk_size:
            # Document is small enough, keep as-is
            chunks.append(Chunk(
                id=f'{doc.id}_0',
                parent_id=doc.id,
                text=text,
                metadata=doc.metadata
            ))
            continue
        
        # Split into overlapping chunks
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
            start += (chunk_size - overlap)  # Overlap for context
    
    print(f"✅ Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks

# =========================
# BM25 Implementation
# =========================
class SimpleBM25:
    """Lightweight BM25 for keyword-based retrieval"""
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc.split()) for doc in corpus) / self.corpus_size
        
        # Build IDF scores
        self.idf = {}
        doc_freqs = defaultdict(int)
        
        for doc in corpus:
            words = set(doc.lower().split())
            for word in words:
                doc_freqs[word] += 1
        
        for word, freq in doc_freqs.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a document"""
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
        """Return top-k (doc_idx, score) pairs"""
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# =========================
# RAG System (SIMPLIFIED)
# =========================
class GymBotRAG:
    """Hybrid retrieval with FAISS + BM25 + Cross-encoder reranking"""
    
    def __init__(self, kb_path: str = 'fitness_knowledge_base.jsonl'):
        # Load and chunk documents
        self.docs = load_kb(kb_path)
        self.chunks = chunk_docs(self.docs)
        
        # Initialize models
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Build FAISS index
        print("Building FAISS index...")
        chunk_texts = [c.text for c in self.chunks]
        embeddings = self.embedder.encode(chunk_texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        
        # Build BM25 index
        print("Building BM25 index...")
        self.bm25 = SimpleBM25(chunk_texts)
        
        print("✅ RAG system ready!")
    
    def hybrid_retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using both FAISS and BM25, then merge results"""
        
        # FAISS semantic search
        query_embedding = self.embedder.encode([query])[0].astype('float32').reshape(1, -1)
        faiss_distances, faiss_indices = self.faiss_index.search(query_embedding, k)
        
        # BM25 keyword search
        bm25_results = self.bm25.search(query, k=k)
        
        # Merge and deduplicate
        seen = set()
        merged = []
        
        # Add FAISS results (semantic similarity preferred)
        for idx, dist in zip(faiss_indices[0], faiss_distances[0]):
            if idx not in seen:
                chunk = self.chunks[idx]
                merged.append({
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'score': float(1.0 / (1.0 + dist)),  # Convert distance to similarity
                    'source': 'faiss',
                    'metadata': chunk.metadata
                })
                seen.add(idx)
        
        # Add BM25 results
        for idx, score in bm25_results:
            if idx not in seen:
                chunk = self.chunks[idx]
                merged.append({
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'score': float(score / 10.0),  # Normalize roughly
                    'source': 'bm25',
                    'metadata': chunk.metadata
                })
                seen.add(idx)
        
        return merged[:k * 2]  # Return more candidates for reranking
    
    def cross_encoder_rerank(self, query: str, docs: List[Dict], k: int = 6) -> List[Dict]:
        """Rerank retrieved documents using cross-encoder"""
        if not docs:
            return []
        
        # Prepare pairs for reranking
        pairs = [[query, doc['text']] for doc in docs]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Add scores and sort
        for doc, score in zip(docs, scores):
            doc['rerank_score'] = float(score)
        
        docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        return docs[:k]
    
    def retrieve(self, query: str, k: int = 6) -> Tuple[List[Dict], float]:
        """
        Full retrieval pipeline: hybrid retrieve -> rerank
        Returns: (docs, confidence_score)
        """
        # Step 1: Hybrid retrieval
        candidates = self.hybrid_retrieve(query, k=10)
        
        if not candidates:
            return [], 0.0
        
        # Step 2: Cross-encoder reranking
        reranked = self.cross_encoder_rerank(query, candidates, k=k)
        
        # Step 3: Calculate confidence from top scores
        if reranked and 'rerank_score' in reranked[0]:
            top_scores = [doc['rerank_score'] for doc in reranked[:3]]
            confidence = min(0.99, max(0.0, sum(top_scores) / len(top_scores)))
        else:
            confidence = 0.65  # Default moderate confidence
        
        return reranked, float(confidence)
    
    def generate_grounded_answer(self, query: str, max_new_tokens: int = 400) -> str:
        """
        End-to-end: retrieve context -> generate answer with Spotter
        """
        if chat_text is None or build_messages is None:
            raise ImportError("Spotter_AI not available. Import failed.")
        
        # Retrieve context
        docs, confidence = self.retrieve(query, k=6)
        
        if not docs:
            context = "(No relevant information found in knowledge base)"
            confidence = 0.0
        else:
            # Format context with citations
            context = "\n\n".join(
                f"[{i+1}] {doc['text']}" 
                for i, doc in enumerate(docs)
            )[:2000]  # Limit context length
        
        # Build prompt
        system = (
            "You are Spotter AI, a helpful fitness assistant. "
            "Use the provided CONTEXT to answer questions accurately. "
            "If the context doesn't contain enough information, say so clearly. "
            "Keep advice practical and actionable."
        )
        
        user = (
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Use information from the context to answer the question\n"
            f"- Cite sources using [1], [2], etc. when referencing specific information\n"
            f"- If context is insufficient, say 'I don't have enough information about [topic]'\n"
            f"- Keep your answer concise and helpful"
        )
        
        messages = build_messages(system, user)
        
        # Generate with deterministic seed for consistency
        seed = int(hashlib.sha256(query.encode()).hexdigest(), 16) % (2**31 - 1)
        
        return chat_text(
            messages,
            max_new_tokens=max_new_tokens,
            intent="knowledge",
            confidence=confidence,
            seed=seed
        )

# =========================
# Convenience Functions
# =========================
def initialize_rag(kb_path: str = 'fitness_knowledge_base.jsonl') -> GymBotRAG:
    """Initialize the RAG system"""
    return GymBotRAG(kb_path)

# Backwards compatibility with your existing code
def hybrid_retrieve(query: str, k: int = 6) -> List[Dict]:
    """Global function for backwards compatibility"""
    if not hasattr(hybrid_retrieve, '_rag_instance'):
        hybrid_retrieve._rag_instance = initialize_rag()
    docs, _ = hybrid_retrieve._rag_instance.retrieve(query, k)
    return docs

def cross_encoder_rerank(query: str, docs: List[Dict], k: int = 6) -> List[Dict]:
    """Global function for backwards compatibility"""
    if not hasattr(cross_encoder_rerank, '_rag_instance'):
        cross_encoder_rerank._rag_instance = initialize_rag()
    return cross_encoder_rerank._rag_instance.cross_encoder_rerank(query, docs, k)

def generate_grounded_answer(query: str) -> str:
    """Global function for backwards compatibility"""
    if not hasattr(generate_grounded_answer, '_rag_instance'):
        generate_grounded_answer._rag_instance = initialize_rag()
    return generate_grounded_answer._rag_instance.generate_grounded_answer(query)

# =========================
# Main
# =========================
if __name__ == "__main__":
    print("Initializing GymBot RAG system...")
    rag = initialize_rag()
    
    # Test queries
    test_queries = [
       
    ]
    
    print("\n" + "="*60)
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        # Test retrieval
        docs, conf = rag.retrieve(query, k=3)
        print(f"Confidence: {conf:.2f}")
        print(f"Retrieved {len(docs)} documents")
        
        # Test generation
        if chat_text is not None:
            answer = rag.generate_grounded_answer(query)
            print(f"\nAnswer:\n{answer}")
        
        print("="*60)
