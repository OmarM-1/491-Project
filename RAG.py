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

# Optional model wrapper import (no circular deps)
try:
    from Spotter_AI import chat_text, detect_intent, build_messages  # generator
except Exception:
    chat_text = None
    detect_intent = None
    build_messages = None

# =========================
# CORE 1 — Knowledge Base (sample; call init_kb(--kb) to use your own)
# =========================
with open('fitness_knowledge_base.jsonl', 'r') as f:
    documents = json.load(f)

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
    return [Document(**d) for d in 'fitness_knowledge_base']

# =========================
# CORE 2 — Chunking + Metadata
# =========================
def normalize_text(t: str) -> str:
    import re as _re
    t = _re.sub(r"\s+", " ", t).strip()
    return t

def chunk_docs(docs: List[Document], chunk_size: int = 420, overlap: int = 60) -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0
    for d in docs:
        text = d.text
        num_chunks = math.ceil(len(text) / chunk_size)
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                id=f'{d.id}_{cid}', parent_id=d.id, text=chunk_text, metadata=d.metadata
            ))
            cid += 1
    return chunks

# =========================
# FAISS + BM25 Hybrid Retrieval Setup
# =========================

# FAISS Setup (for semantic search)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize Sentence-BERT for document embedding
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for documents
documents_text = [doc['text'] for doc in documents]
document_embeddings = sentence_model.encode(documents_text)

# Convert embeddings to float32 (FAISS format)
document_embeddings = np.array(document_embeddings).astype('float32')

# Create a FAISS index for vector-based retrieval
faiss_index = faiss.IndexFlatL2(document_embeddings.shape[1])
faiss_index.add(document_embeddings)

# Function to retrieve documents using FAISS
def faiss_retrieve(query: str, k=5) -> List[Document]:
    # Encode the query using Sentence-BERT
    query_embedding = sentence_model.encode([query])[0].astype('float32').reshape(1, -1)
    
    # Retrieve the top-k most similar documents from FAISS index
    _, I = faiss_index.search(query_embedding, k)  # I contains the indices of the closest vectors
    
    # Return the documents corresponding to the retrieved indices
    return [documents[i] for i in I[0]]

# =========================
# RAG Setup (Retrieval and Generation)
# =========================

# Initialize RAG model and tokenizer
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model_rag = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# Custom RAG Retriever class to use FAISS
class FAISS_Retriever(RagRetriever):
    def __init__(self, faiss_index, tokenizer, k=5):
        self.index = faiss_index
        self.tokenizer = tokenizer
        self.k = k
    
    def get_relevant_documents(self, query: str):
        # Retrieve the top-k most relevant documents using FAISS
        faiss_results = faiss_retrieve(query, self.k)
        return faiss_results

# Instantiate the custom retriever
retriever = FAISS_Retriever(faiss_index=faiss_index, tokenizer=tokenizer)

# =========================
# Answer Generation (Grounded Answer)
# =========================

def generate_grounded_answer(query: str) -> str:
    # Step 1: Retrieve relevant documents using FAISS
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Step 2: Tokenize the user query and retrieved documents
    inputs = tokenizer(query, return_tensors="pt")
    context_input_ids = tokenizer([doc['text'] for doc in retrieved_docs], return_tensors="pt")['input_ids']
    
    # Step 3: Generate the response using the RAG model
    generated_output = model_rag.generate(input_ids=inputs["input_ids"], 
                                          context_input_ids=context_input_ids)
    
    # Step 4: Decode and return the generated response
    response = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    return response

# =========================
# Final Integration (Main Logic)
# =========================

def process_with_rag_and_agents(user_request: str) -> str:
    # Generate a grounded answer based on the user's query
    return generate_grounded_answer(user_request)

# Example query (this can be dynamic based on user input)
if __name__ == "__main__":
    query = "What are the benefits of dumbbell bench press?"
    response = process_with_rag_and_agents(query)
    print(f"Generated Response: {response}")