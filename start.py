#!/usr/bin/env python3
"""
Quick start script for GymBot
Automatically configures for your hardware and starts the system
"""

import os
import sys
import platform
import subprocess

def quick_start():
    """Quick start with optimal settings for current hardware"""
    
    print("="*60)
    print("GYMBOT QUICK START")
    print("="*60)
    
    # Detect hardware
    system = platform.system()
    machine = platform.machine()
    is_apple_silicon = system == "Darwin" and machine in ["arm64", "aarch64"]
    
    print(f"\n🔍 Detected: {system} {machine}")
    
    # Set optimal defaults for Mac
    if is_apple_silicon:
        print("✅ Apple Silicon detected")
        print("📝 Configuring for Mac with MPS...")
        
        os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-1.5B-Instruct"
        os.environ['DEVICE_MAP'] = "mps"
        os.environ['LOAD_IN_4BIT'] = "false"
        
        print("\n✅ Configuration:")
        print(f"   Model: Qwen2.5-1.5B (fast, good quality)")
        print(f"   Device: MPS (Mac GPU)")
        print(f"   Expected speed: 2-3s per query")
    
    elif system == "Darwin":
        # Intel Mac
        print("✅ Intel Mac detected")
        print("📝 Configuring for CPU...")
        
        os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-0.5B-Instruct"
        os.environ['DEVICE_MAP'] = "cpu"
        os.environ['LOAD_IN_4BIT'] = "false"
        
        print("\n✅ Configuration:")
        print(f"   Model: Qwen2.5-0.5B (fastest for CPU)")
        print(f"   Device: CPU")
        print(f"   Expected speed: 5-8s per query")
    
    else:
        # Linux/Windows
        try:
            import torch
            if torch.cuda.is_available():
                print("✅ NVIDIA GPU detected")
                print("📝 Configuring for CUDA...")
                
                os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-7B-Instruct"
                os.environ['DEVICE_MAP'] = "auto"
                os.environ['LOAD_IN_4BIT'] = "true"
                
                print("\n✅ Configuration:")
                print(f"   Model: Qwen2.5-7B (best quality)")
                print(f"   Device: CUDA (GPU)")
                print(f"   4-bit: Enabled")
                print(f"   Expected speed: 2-4s per query")
            else:
                print("⚠️  No GPU detected")
                print("📝 Configuring for CPU...")
                
                os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-1.5B-Instruct"
                os.environ['DEVICE_MAP'] = "cpu"
                os.environ['LOAD_IN_4BIT'] = "false"
                
                print("\n✅ Configuration:")
                print(f"   Model: Qwen2.5-1.5B")
                print(f"   Device: CPU")
                print(f"   Expected speed: 8-12s per query")
        except ImportError:
            print("⚠️  PyTorch not found, using safe defaults")
            os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-0.5B-Instruct"
            os.environ['DEVICE_MAP'] = "cpu"
            os.environ['LOAD_IN_4BIT'] = "false"
    
    print("\n" + "="*60)
    
    # Choose what to run
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nWhat would you like to run?")
        print("1. Interactive chat (recommended)")
        print("2. Test RAG only")
        print("3. Test agentic RAG")
        print("4. Complete system with vision")
        
        try:
            choice = input("\nChoice (1-4, default=1): ").strip() or "1"
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return
        
        mode = {
            "1": "interactive",
            "2": "rag",
            "3": "agentic",
            "4": "complete"
        }.get(choice, "interactive")
    
    print(f"\n🚀 Starting in {mode} mode...")
    print("="*60 + "\n")
    
    # Run the appropriate script
    try:
        if mode == "interactive":
            # Import and run orchestrator
            from hybrid_orchestrator import interactive_mode
            interactive_mode()
        
        elif mode == "rag":
            # Test RAG
            subprocess.run([sys.executable, "optimized_rag.py"])
        
        elif mode == "agentic":
            # Test agentic
            subprocess.run([sys.executable, "agentic_rag.py"])
        
        elif mode == "complete":
            # Complete system
            subprocess.run([sys.executable, "complete_gymbot.py", "demo"])
        
        else:
            print(f"Unknown mode: {mode}")
            print("Valid modes: interactive, rag, agentic, complete")
    
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure all files are in the current directory:")
        print("  - hybrid_orchestrator.py")
        print("  - optimized_rag.py")
        print("  - agentic_rag.py")
        print("  - Spotter_AI.py")
        print("  - SAFETY_AGENT.py")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_start()
