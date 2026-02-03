"""
Improved orchestration: Safety -> RAG -> LLM (Spotter)
Cleaner, more maintainable version of RAG_with_Agents.py
"""

from typing import Tuple
import hashlib

# Your modules
from Spotter_AI import chat_text, build_messages
from SAFETY_AGENT import safety_gate_agent
from RAG_simplified import GymBotRAG

# Global RAG instance (initialized once)
_rag_system = None

def get_rag_system() -> GymBotRAG:
    """Lazy initialization of RAG system"""
    global _rag_system
    if _rag_system is None:
        print("Initializing RAG system...")
        _rag_system = GymBotRAG('fitness_knowledge_base.jsonl')
    return _rag_system


def answer(user_query: str, intent: str = "knowledge") -> str:
    """
    Complete pipeline: Safety -> RAG -> LLM
    
    Args:
        user_query: User's question
        intent: Override intent detection (knowledge/plan/creative/safety)
    
    Returns:
        Final answer string
    """
    
    # ====================
    # STEP 1: Safety Check
    # ====================
    ok, warning = safety_gate_agent(
        user_query,
        chat=lambda msgs, **kw: chat_text(
            msgs, 
            intent="safety", 
            confidence=0.0,  # Deterministic for safety
            seed=0
        )
    )
    
    if not ok:
        return warning  # Stop here if safety concerns detected
    
    # ====================
    # STEP 2: RAG Retrieval
    # ====================
    rag = get_rag_system()
    docs, confidence = rag.retrieve(user_query, k=6)
    
    # Format context with citations
    if docs:
        context = "\n\n".join(
            f"[{i+1}] {doc['text']}" 
            for i, doc in enumerate(docs)
        )[:2000]  # Limit to 2000 chars
        
        sources = [doc.get('chunk_id', 'unknown') for doc in docs]
    else:
        context = "(No relevant information found in knowledge base)"
        confidence = 0.0
        sources = []
    
    # ====================
    # STEP 3: Build Prompt
    # ====================
    system = (
        "You are Spotter AI, a knowledgeable fitness assistant. "
        "Use the provided CONTEXT from your knowledge base to answer questions. "
        "Be precise, practical, and actionable. "
        "If the context doesn't provide enough information, clearly state what you don't know."
    )
    
    user = (
        f"QUESTION:\n{user_query}\n\n"
        f"CONTEXT (from knowledge base):\n{context}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Answer based on the context provided\n"
        f"- Cite sources using [1], [2], etc. when making specific claims\n"
        f"- If information is missing, say 'I don't have information about [specific topic]'\n"
        f"- Keep advice clear and actionable\n"
        f"- Don't make up information not in the context"
    )
    
    messages = build_messages(system, user)
    
    # ====================
    # STEP 4: Generate Answer
    # ====================
    # Deterministic seed based on query for reproducible answers
    seed = int(hashlib.sha256(user_query.encode()).hexdigest(), 16) % (2**31 - 1)
    
    answer_text = chat_text(
        messages,
        max_new_tokens=400,
        intent=intent,
        confidence=confidence,  # RAG confidence controls sampling temperature
        seed=seed
    )
    
    return answer_text


def answer_with_metadata(user_query: str, intent: str = "knowledge") -> dict:
    """
    Same as answer() but returns metadata for debugging/analysis
    
    Returns:
        {
            'answer': str,
            'confidence': float,
            'sources': list,
            'intent': str,
            'safety_passed': bool
        }
    """
    
    # Safety check
    ok, warning = safety_gate_agent(
        user_query,
        chat=lambda msgs, **kw: chat_text(msgs, intent="safety", confidence=0.0, seed=0)
    )
    
    if not ok:
        return {
            'answer': warning,
            'confidence': 0.0,
            'sources': [],
            'intent': 'safety',
            'safety_passed': False
        }
    
    # RAG retrieval
    rag = get_rag_system()
    docs, confidence = rag.retrieve(user_query, k=6)
    
    # Build context
    if docs:
        context = "\n\n".join(f"[{i+1}] {doc['text']}" for i, doc in enumerate(docs))[:2000]
        sources = [doc.get('chunk_id', 'unknown') for doc in docs]
    else:
        context = "(No relevant information found)"
        confidence = 0.0
        sources = []
    
    # Build prompt
    system = (
        "You are Spotter AI, a knowledgeable fitness assistant. "
        "Use the provided CONTEXT to answer questions accurately and practically."
    )
    
    user = (
        f"QUESTION:\n{user_query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"Answer the question using the context. Cite sources with [1], [2], etc."
    )
    
    messages = build_messages(system, user)
    seed = int(hashlib.sha256(user_query.encode()).hexdigest(), 16) % (2**31 - 1)
    
    # Generate
    answer_text = chat_text(
        messages,
        max_new_tokens=400,
        intent=intent,
        confidence=confidence,
        seed=seed
    )
    
    return {
        'answer': answer_text,
        'confidence': confidence,
        'sources': sources,
        'intent': intent,
        'safety_passed': True
    }


def interactive_mode():
    """Run interactive CLI for testing"""
    print("="*60)
    print("GymBot RAG System - Interactive Mode")
    print("="*60)
    print("\nInitializing...")
    
    # Pre-initialize RAG
    get_rag_system()
    
    print("\n✅ Ready! Type your questions (Ctrl+C to exit)\n")
    
    try:
        while True:
            query = input("\n> ").strip()
            if not query:
                continue
            
            print("\n" + "-"*60)
            
            # Get answer with metadata
            result = answer_with_metadata(query)
            
            # Display
            print(f"Answer: {result['answer']}")
            print(f"\nConfidence: {result['confidence']:.2f}")
            print(f"Intent: {result['intent']}")
            print(f"Sources: {', '.join(result['sources'][:3])}")
            print("-"*60)
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    # Quick test
    test_query = "What are the benefits of dumbbell bench press?"
    print(f"Test query: {test_query}\n")
    print(answer(test_query))
    
    # Uncomment to run interactive mode
    # interactive_mode()
