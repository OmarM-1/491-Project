"""
HYBRID ORCHESTRATOR - Best of both worlds

Intelligently routes queries between:
- Regular RAG (fast, for simple queries)
- Agentic RAG (smart, for complex queries)

Performance target: <2s for 80% of queries, <10s for complex 20%
"""

from typing import Dict, Any, Tuple
import re
from SAFETY_AGENT import safety_gate_agent
from Spotter_AI import chat_text

# Import both RAG systems
try:
    from optimized_rag import get_rag, generate_grounded_answer
    REGULAR_RAG_AVAILABLE = True
except ImportError:
    REGULAR_RAG_AVAILABLE = False
    print("⚠️  Regular RAG not available")

try:
    from agentic_rag import get_agent, agentic_answer
    AGENTIC_RAG_AVAILABLE = True
except ImportError:
    AGENTIC_RAG_AVAILABLE = False
    print("⚠️  Agentic RAG not available")

# =========================
# Query Complexity Scorer
# =========================
class ComplexityScorer:
    """Determines if a query needs agentic reasoning"""
    
    # Keywords that suggest calculation needs
    CALC_KEYWORDS = [
        'how many', 'how much', 'calculate', 'compute',
        'calories', 'pounds', 'kilograms', 'weeks', 'months', 'days',
        'percent', 'percentage', '%', 'bmi', 'weight', 'gain', 'lose',
        'if i', 'should i'
    ]
    
    # Keywords suggesting multi-part queries
    COMPLEX_KEYWORDS = [
        'and', 'both', 'also', 'compare', 'versus', 'vs',
        'plan', 'routine', 'program', 'schedule',
        'multiple', 'several', 'different', 'various',
        'best way', 'optimal', 'recommend'
    ]
    
    # Keywords for simple factual queries (prefer regular RAG)
    SIMPLE_KEYWORDS = [
        'what is', 'define', 'definition', 'explain',
        'show me', 'how to do', 'form', 'technique',
        'muscles worked', 'benefits of'
    ]
    
    @classmethod
    def score(cls, query: str) -> int:
        """
        Score query complexity 0-10
        
        0-4: Simple factual (use regular RAG)
        5-7: Moderate complexity (use regular RAG, but close)
        8-10: Complex reasoning (use agentic RAG)
        """
        query_lower = query.lower()
        score = 5  # Start at neutral
        
        # Check for calculation needs (+3)
        if any(kw in query_lower for kw in cls.CALC_KEYWORDS):
            score += 3
        
        # Check for multi-part complexity (+2)
        complex_matches = sum(1 for kw in cls.COMPLEX_KEYWORDS if kw in query_lower)
        if complex_matches >= 2:
            score += 2
        elif complex_matches == 1:
            score += 1
        
        # Check for simple queries (-2)
        if any(kw in query_lower for kw in cls.SIMPLE_KEYWORDS):
            score -= 2
        
        # Question marks suggest simpler queries (-1)
        if query.count('?') == 1 and len(query.split()) < 15:
            score -= 1
        
        # Very long queries are usually complex (+1)
        word_count = len(query.split())
        if word_count > 25:
            score += 1
        elif word_count < 10:
            score -= 1
        
        # Has numbers? Likely needs calculation (+1)
        if re.search(r'\d+', query):
            score += 1
        
        # Clamp to 0-10
        return max(0, min(10, score))
    
    @classmethod
    def explain_score(cls, query: str) -> str:
        """Explain why a query got its score"""
        score = cls.score(query)
        query_lower = query.lower()
        
        reasons = []
        
        if any(kw in query_lower for kw in cls.CALC_KEYWORDS):
            reasons.append("needs calculation")
        
        complex_count = sum(1 for kw in cls.COMPLEX_KEYWORDS if kw in query_lower)
        if complex_count > 0:
            reasons.append(f"{complex_count} complexity indicators")
        
        if any(kw in query_lower for kw in cls.SIMPLE_KEYWORDS):
            reasons.append("simple factual query")
        
        if re.search(r'\d+', query):
            reasons.append("contains numbers")
        
        word_count = len(query.split())
        reasons.append(f"{word_count} words")
        
        reason_str = ", ".join(reasons)
        return f"Score {score}/10: {reason_str}"

# =========================
# Hybrid Orchestrator
# =========================
class HybridOrchestrator:
    """
    Routes queries intelligently between RAG systems
    """
    
    def __init__(
        self,
        complexity_threshold: int = 7,
        force_regular: bool = False,
        force_agentic: bool = False,
        verbose: bool = False
    ):
        """
        Args:
            complexity_threshold: Score >= this uses agentic RAG
            force_regular: Always use regular RAG (for testing)
            force_agentic: Always use agentic RAG (for testing)
            verbose: Print routing decisions
        """
        self.threshold = complexity_threshold
        self.force_regular = force_regular
        self.force_agentic = force_agentic
        self.verbose = verbose
        
        # Check availability
        if not REGULAR_RAG_AVAILABLE and not AGENTIC_RAG_AVAILABLE:
            raise RuntimeError("No RAG systems available!")
    
    def _route(self, query: str) -> Tuple[str, str]:
        """
        Decide which RAG system to use
        
        Returns:
            (system_name, reason)
        """
        # Handle force flags
        if self.force_regular:
            return "regular", "forced by flag"
        if self.force_agentic:
            return "agentic", "forced by flag"
        
        # Handle missing systems
        if not REGULAR_RAG_AVAILABLE:
            return "agentic", "regular RAG not available"
        if not AGENTIC_RAG_AVAILABLE:
            return "regular", "agentic RAG not available"
        
        # Score complexity
        score = ComplexityScorer.score(query)
        
        if score >= self.threshold:
            return "agentic", f"complex query (score {score})"
        else:
            return "regular", f"simple query (score {score})"
    
    def answer(
        self,
        query: str,
        return_metadata: bool = False
    ) -> str | Dict[str, Any]:
        """
        Main entry point - routes and answers query
        
        Args:
            query: User question
            return_metadata: If True, return dict with metadata
        
        Returns:
            answer string or dict with metadata
        """
        # Safety check
        ok, warning = safety_gate_agent(
            query,
            chat=lambda msgs, **kw: chat_text(msgs, intent="safety", confidence=0.0, seed=0)
        )
        
        if not ok:
            if return_metadata:
                return {
                    'answer': warning,
                    'system': 'safety',
                    'safe': False,
                    'route_reason': 'safety check failed'
                }
            return warning
        
        # Route
        system, reason = self._route(query)
        
        if self.verbose:
            print(f"\n🔀 Routing to {system.upper()} RAG")
            print(f"   Reason: {reason}")
            if system == "agentic":
                print(f"   Note: This may take 5-10 seconds...")
        
        # Execute
        import time
        start = time.time()
        
        try:
            if system == "regular":
                answer = generate_grounded_answer(query)
                metadata = {
                    'system': 'regular',
                    'complexity_score': ComplexityScorer.score(query),
                }
            else:  # agentic
                if return_metadata:
                    result = get_agent(verbose=self.verbose).answer(query)
                    answer = result['answer']
                    metadata = {
                        'system': 'agentic',
                        'complexity_score': ComplexityScorer.score(query),
                        'steps': result.get('steps', []),
                        'confidence': result.get('confidence', 0.5),
                        'reasoning': result.get('reasoning', ''),
                    }
                else:
                    answer = agentic_answer(query, verbose=self.verbose)
                    metadata = {'system': 'agentic'}
            
            elapsed = time.time() - start
            
            if self.verbose:
                print(f"✅ Answered in {elapsed:.2f}s using {system} RAG")
            
            if return_metadata:
                metadata.update({
                    'answer': answer,
                    'route_reason': reason,
                    'time_seconds': elapsed,
                    'safe': True
                })
                return metadata
            
            return answer
            
        except Exception as e:
            error_msg = f"Error with {system} RAG: {str(e)}"
            print(f"❌ {error_msg}")
            
            if return_metadata:
                return {
                    'answer': "I encountered an error processing your query. Please try again.",
                    'system': system,
                    'error': str(e),
                    'safe': True
                }
            
            return "I encountered an error processing your query. Please try again."

# =========================
# Convenience Functions
# =========================
_ORCHESTRATOR = None

def get_orchestrator(
    complexity_threshold: int = 7,
    verbose: bool = False
) -> HybridOrchestrator:
    """Get global orchestrator instance"""
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = HybridOrchestrator(
            complexity_threshold=complexity_threshold,
            verbose=verbose
        )
    return _ORCHESTRATOR

def smart_answer(query: str, verbose: bool = False) -> str:
    """Simple interface for hybrid RAG"""
    orchestrator = get_orchestrator(verbose=verbose)
    return orchestrator.answer(query)

def analyze_query(query: str) -> None:
    """Analyze a query and show routing decision"""
    print(f"\nQuery: {query}")
    print("="*60)
    
    score = ComplexityScorer.score(query)
    explanation = ComplexityScorer.explain_score(query)
    
    print(f"Complexity: {explanation}")
    
    orchestrator = get_orchestrator()
    system, reason = orchestrator._route(query)
    
    print(f"Route: {system.upper()} RAG")
    print(f"Reason: {reason}")
    
    print(f"\nThreshold: {orchestrator.threshold}")
    print(f"This query: {'ABOVE' if score >= orchestrator.threshold else 'BELOW'} threshold")

# =========================
# Interactive Mode
# =========================
def interactive_mode():
    """Interactive CLI with hybrid routing"""
    print("="*60)
    print("HYBRID GYMBOT - Smart Query Routing")
    print("="*60)
    print("\nCommands:")
    print("  !analyze <query> - Show routing decision without executing")
    print("  !stats - Show routing statistics")
    print("  !regular - Force regular RAG mode")
    print("  !agentic - Force agentic RAG mode")
    print("  !auto - Return to automatic routing")
    print("  !help - Show this help")
    print("\nType your questions (Ctrl+C to exit)\n")
    
    orchestrator = get_orchestrator(verbose=True)
    stats = {'regular': 0, 'agentic': 0, 'safety': 0}
    
    try:
        while True:
            query = input("\n> ").strip()
            if not query:
                continue
            
            # Handle commands
            if query.startswith('!'):
                cmd = query.split()[0].lower()
                
                if cmd == '!analyze':
                    analyze_query(' '.join(query.split()[1:]))
                    continue
                
                elif cmd == '!stats':
                    total = sum(stats.values())
                    print(f"\nQuery Statistics (total: {total}):")
                    for system, count in stats.items():
                        pct = (count / total * 100) if total > 0 else 0
                        print(f"  {system.capitalize()}: {count} ({pct:.1f}%)")
                    continue
                
                elif cmd == '!regular':
                    orchestrator.force_regular = True
                    orchestrator.force_agentic = False
                    print("✅ Forcing regular RAG mode")
                    continue
                
                elif cmd == '!agentic':
                    orchestrator.force_agentic = True
                    orchestrator.force_regular = False
                    print("✅ Forcing agentic RAG mode")
                    continue
                
                elif cmd == '!auto':
                    orchestrator.force_regular = False
                    orchestrator.force_agentic = False
                    print("✅ Automatic routing enabled")
                    continue
                
                elif cmd == '!help':
                    print("\nCommands:")
                    print("  !analyze <query> - Analyze routing without executing")
                    print("  !stats - Show routing statistics")
                    print("  !regular - Force regular RAG")
                    print("  !agentic - Force agentic RAG")
                    print("  !auto - Auto routing")
                    continue
                
                else:
                    print("Unknown command. Type !help for help.")
                    continue
            
            # Process query
            print("\n" + "-"*60)
            result = orchestrator.answer(query, return_metadata=True)
            
            print(f"\n📝 Answer:\n{result['answer']}\n")
            print(f"🔧 System: {result['system']}")
            print(f"⏱️  Time: {result.get('time_seconds', 0):.2f}s")
            
            if 'complexity_score' in result:
                print(f"📊 Complexity: {result['complexity_score']}/10")
            
            # Update stats
            stats[result['system']] = stats.get(result['system'], 0) + 1
            
            print("-"*60)
    
    except KeyboardInterrupt:
        print("\n\nFinal Statistics:")
        total = sum(stats.values())
        for system, count in stats.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {system.capitalize()}: {count} ({pct:.1f}%)")
        print("\nGoodbye!")

# =========================
# Main
# =========================
if __name__ == "__main__":
    interactive_mode()
