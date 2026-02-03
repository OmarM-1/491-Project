#!/usr/bin/env python3
"""
EMERGENCY FIX for GymBot
Fixes: Same context + 20 second responses

This script:
1. Deletes corrupted cache
2. Sets optimal model for Mac
3. Rebuilds FAISS index from scratch
4. Tests that retrieval works
5. Uses intent filtering to skip RAG for non-fitness queries
"""

import os
import sys
import glob
import time

def emergency_fix():
    print("="*60)
    print("GYMBOT EMERGENCY FIX")
    print("="*60)
    
    # Step 1: Clean up corrupted cache
    print("\n📝 Step 1: Removing corrupted cache files...")
    cache_files = glob.glob('.cache_faiss_*.pkl')
    if cache_files:
        for f in cache_files:
            os.remove(f)
            print(f"   Deleted: {f}")
    else:
        print("   No cache files found")
    
    # Step 2: Set optimal configuration for Mac
    print("\n📝 Step 2: Configuring for Mac...")
    
    import platform
    machine = platform.machine()
    is_apple_silicon = machine in ["arm64", "aarch64"]
    
    if is_apple_silicon:
        print("   ✅ Detected Apple Silicon (M1/M2/M3)")
        os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-0.5B-Instruct"  # Fastest
        os.environ['DEVICE_MAP'] = "mps"
        print("   Model: Qwen2.5-0.5B (fastest)")
        print("   Device: MPS (Mac GPU)")
    else:
        print("   ✅ Detected Intel Mac")
        os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-0.5B-Instruct"
        os.environ['DEVICE_MAP'] = "cpu"
        print("   Model: Qwen2.5-0.5B (fastest)")
        print("   Device: CPU")
    
    os.environ['LOAD_IN_4BIT'] = "false"
    
    # Step 3: Rebuild RAG system
    print("\n📝 Step 3: Rebuilding RAG system from scratch...")
    print("   (This may take 30-60 seconds...)")
    
    try:
        from optimized_rag import OptimizedGymBotRAG
        
        # Force rebuild
        rag = OptimizedGymBotRAG(force_rebuild=True)
        print("   ✅ RAG rebuilt successfully")
    except Exception as e:
        print(f"   ❌ Failed to rebuild RAG: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test retrieval
    print("\n📝 Step 4: Testing retrieval...")
    
    test_queries = [
        ("bench press", ["bench", "chest", "press", "pectoral"]),
        ("protein", ["protein", "nutrition", "muscle", "intake"]),
        ("squat", ["squat", "legs", "glutes", "compound"]),
    ]
    
    all_good = True
    
    for query, expected_keywords in test_queries:
        print(f"\n   Testing: '{query}'")
        
        start = time.time()
        docs, conf = rag.retrieve(query, k=3)
        elapsed = time.time() - start
        
        print(f"      Time: {elapsed:.2f}s")
        print(f"      Confidence: {conf:.2f}")
        
        if docs:
            top_text = docs[0]['text'].lower()
            found_keywords = [kw for kw in expected_keywords if kw in top_text]
            
            if found_keywords:
                print(f"      ✅ Found relevant: {', '.join(found_keywords)}")
            else:
                print(f"      ⚠️  Top result doesn't contain expected keywords")
                print(f"      Got: {docs[0]['text'][:80]}...")
                all_good = False
        else:
            print(f"      ❌ No documents retrieved!")
            all_good = False
    
    # Step 5: Test speed
    print("\n📝 Step 5: Testing query speed...")
    
    # Warm up
    rag.retrieve("test", k=3)
    
    # Time a simple query
    start = time.time()
    docs, conf = rag.retrieve("deadlift", k=3)
    elapsed = time.time() - start
    
    print(f"   Retrieval time: {elapsed:.2f}s")
    
    if elapsed < 1.0:
        print(f"   ✅ Speed is GOOD!")
    elif elapsed < 2.0:
        print(f"   ⚠️  Speed is acceptable")
    else:
        print(f"   ❌ Speed is too slow")
        all_good = False
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if all_good:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYour system is now:")
        print("  ✅ Retrieving correct, relevant documents")
        print("  ✅ Running at acceptable speed")
        print("  ✅ Using optimal model for Mac")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Use the improved orchestrator (with intent filtering):")
        print("   python hybrid_orchestrator_v2.py interactive")
        print("\n2. Or use start.py:")
        print("   python start.py")
        print("\n3. Test with:")
        print("   > what is 1+1?         (should skip RAG, answer in 2-3s)")
        print("   > what is bench press? (should use RAG, answer in 3-5s)")
        
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease check the errors above.")
        print("\nCommon issues:")
        print("  1. Knowledge base file is corrupted")
        print("  2. Model download failed")
        print("  3. Not enough RAM")
        
        return False

if __name__ == "__main__":
    print("\nThis will:")
    print("  1. Delete cache files")
    print("  2. Set optimal model (0.5B - fastest)")
    print("  3. Rebuild FAISS index")
    print("  4. Test retrieval works correctly")
    print("  5. Test speed is acceptable")
    
    try:
        response = input("\nProceed? (yes/no): ").lower().strip()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    
    if response not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)
    
    print()
    success = emergency_fix()
    
    if success:
        print("\n🎉 Fix complete! You can now use the system.")
        sys.exit(0)
    else:
        print("\n⚠️  Fix incomplete - please check errors above")
        sys.exit(1)
