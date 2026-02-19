#!/usr/bin/env python3
"""
Quick start script for GymBot
Automatically configures for your hardware and starts the system

Usage:
  python start.py              # Text-only mode (fast)
  python start.py --vision     # Enable photo/video analysis
  python start.py interactive  # Direct mode selection
"""

import os
import sys
import platform
import subprocess

def quick_start():
    """Quick start with optimal settings for current hardware"""
    
    # Check for vision flag
    enable_vision = '--vision' in sys.argv
    if enable_vision:
        sys.argv.remove('--vision')  # Remove so it doesn't interfere with mode selection
    
    print("="*60)
    print("GYMBOT QUICK START")
    if enable_vision:
        print("🎬 VISION MODE ENABLED")
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
        
        if enable_vision:
            os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2-VL-2B-Instruct"
            print("\nâœ… Configuration:")
            print(f"   Model: Qwen2-VL-2B (vision-enabled)")
            print(f"   Device: MPS (Mac GPU)")
            print(f"   Features: Text + Photo + Video Analysis")
            print(f"   Expected speed: 2-5s per image, 15-40s per video")
        else:
            os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-1.5B-Instruct"
            print("\nâœ… Configuration:")
            print(f"   Model: Qwen2.5-1.5B (fast, good quality)")
            print(f"   Device: MPS (Mac GPU)")
            print(f"   Features: Text only")
            print(f"   Expected speed: 2-3s per query")
            print(f"   💡 Tip: Add --vision flag for photo/video analysis")
        
        os.environ['DEVICE_MAP'] = "mps"
        os.environ['LOAD_IN_4BIT'] = "false"
    
    elif system == "Darwin":
        # Intel Mac
        print("✅ Intel Mac detected")
        print("📝 Configuring for CPU...")
        
        if enable_vision:
            os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2-VL-2B-Instruct"
            print("\nâœ… Configuration:")
            print(f"   Model: Qwen2-VL-2B (vision-enabled)")
            print(f"   Device: CPU")
            print(f"   Features: Text + Photo + Video Analysis")
            print(f"   Expected speed: 10-15s per image, 60-120s per video")
            print(f"   ⚠️  Vision on CPU is slower - consider using text-only mode")
        else:
            os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-0.5B-Instruct"
            print("\nâœ… Configuration:")
            print(f"   Model: Qwen2.5-0.5B (fastest for CPU)")
            print(f"   Device: CPU")
            print(f"   Features: Text only")
            print(f"   Expected speed: 5-8s per query")
        
        os.environ['DEVICE_MAP'] = "cpu"
        os.environ['LOAD_IN_4BIT'] = "false"
    
    else:
        # Linux/Windows
        try:
            import torch
            if torch.cuda.is_available():
                print("✅ NVIDIA GPU detected")
                print("📝 Configuring for CUDA...")
                
                if enable_vision:
                    os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2-VL-7B-Instruct"
                    print("\nâœ… Configuration:")
                    print(f"   Model: Qwen2-VL-7B (vision-enabled, best quality)")
                    print(f"   Device: CUDA (GPU)")
                    print(f"   4-bit: Enabled")
                    print(f"   Features: Text + Photo + Video Analysis")
                    print(f"   Expected speed: 1-3s per image, 8-20s per video")
                else:
                    os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-7B-Instruct"
                    print("\nâœ… Configuration:")
                    print(f"   Model: Qwen2.5-7B (best quality)")
                    print(f"   Device: CUDA (GPU)")
                    print(f"   4-bit: Enabled")
                    print(f"   Features: Text only")
                    print(f"   Expected speed: 2-4s per query")
                    print(f"   💡 Tip: Add --vision flag for photo/video analysis")
                
                os.environ['DEVICE_MAP'] = "auto"
                os.environ['LOAD_IN_4BIT'] = "true"
            else:
                print("⚠️  No GPU detected")
                print("📝 Configuring for CPU...")
                
                if enable_vision:
                    os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2-VL-2B-Instruct"
                    print("\nâœ… Configuration:")
                    print(f"   Model: Qwen2-VL-2B (vision-enabled)")
                    print(f"   Device: CPU")
                    print(f"   Features: Text + Photo + Video Analysis")
                    print(f"   Expected speed: 10-15s per image, 60-120s per video")
                    print(f"   ⚠️  Vision on CPU is slower")
                else:
                    os.environ['SPOTTER_MODEL'] = "Qwen/Qwen2.5-1.5B-Instruct"
                    print("\nâœ… Configuration:")
                    print(f"   Model: Qwen2.5-1.5B")
                    print(f"   Device: CPU")
                    print(f"   Features: Text only")
                    print(f"   Expected speed: 8-12s per query")
                
                os.environ['DEVICE_MAP'] = "cpu"
                os.environ['LOAD_IN_4BIT'] = "false"
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
        if enable_vision:
            print("4. Vision demo (photo/video analysis)")
            print("5. Complete system")
        else:
            print("4. Complete system")
        
        try:
            max_choice = 5 if enable_vision else 4
            choice = input(f"\nChoice (1-{max_choice}, default=1): ").strip() or "1"
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return
        
        if enable_vision:
            mode = {
                "1": "interactive",
                "2": "rag",
                "3": "agentic",
                "4": "vision",
                "5": "complete"
            }.get(choice, "interactive")
        else:
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
        
        elif mode == "vision":
            # Vision demo
            if not enable_vision:
                print("❌ Vision mode requires --vision flag!")
                print("Run with: python start.py --vision")
                return
            subprocess.run([sys.executable, "vision_demo.py"])
        
        elif mode == "complete":
            # Complete system
            subprocess.run([sys.executable, "complete_gymbot.py", "demo"])
        
        else:
            print(f"Unknown mode: {mode}")
            print("Valid modes: interactive, rag, agentic, vision, complete")
    
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure all files are in the current directory:")
        print("  - hybrid_orchestrator.py")
        print("  - optimized_rag.py")
        print("  - agentic_rag.py")
        print("  - Spotter_AI.py")
        print("  - SAFETY_AGENT.py")
        if enable_vision:
            print("  - vision_demo.py (for vision mode)")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_start()
