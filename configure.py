"""
Auto-detect hardware and set optimal configuration for GymBot
Run this before starting your system to configure for your hardware
"""

import os
import platform
import torch

def detect_hardware():
    """Detect available hardware and recommend settings"""
    
    print("="*60)
    print("HARDWARE DETECTION")
    print("="*60)
    
    system = platform.system()
    machine = platform.machine()
    
    print(f"\nSystem: {system}")
    print(f"Architecture: {machine}")
    
    # Check for Apple Silicon
    is_apple_silicon = system == "Darwin" and machine in ["arm64", "aarch64"]
    
    # Check CUDA
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU Memory: {cuda_memory:.1f} GB")
    
    # Check MPS (Mac GPU)
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if has_mps:
        print(f"✅ MPS (Apple Silicon GPU) available")
    
    # Get RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 RAM: {ram_gb:.1f} GB")
    except ImportError:
        ram_gb = 16  # Assume 16GB if can't detect
        print(f"💾 RAM: Unable to detect (assuming 16GB)")
    
    print("\n" + "="*60)
    print("RECOMMENDED CONFIGURATION")
    print("="*60)
    
    # Determine best settings
    if has_cuda:
        if cuda_memory >= 24:
            # High-end GPU (e.g., RTX 3090, 4090, A100)
            model = "Qwen/Qwen2.5-7B-Instruct"
            device = "auto"
            quant = "false"
            speed = "Fast (2-3s)"
            quality = "Best"
        elif cuda_memory >= 12:
            # Mid-range GPU (e.g., RTX 3060, 4060 Ti)
            model = "Qwen/Qwen2.5-7B-Instruct"
            device = "auto"
            quant = "true"
            speed = "Fast (2-4s)"
            quality = "Very Good"
        else:
            # Low VRAM GPU
            model = "Qwen/Qwen2.5-1.5B-Instruct"
            device = "auto"
            quant = "true"
            speed = "Fast (1-2s)"
            quality = "Good"
    
    elif is_apple_silicon and has_mps:
        # Apple Silicon with MPS
        if ram_gb >= 32:
            model = "Qwen/Qwen2.5-7B-Instruct"
            device = "mps"
            quant = "false"
            speed = "Medium (3-5s)"
            quality = "Best"
        elif ram_gb >= 16:
            model = "Qwen/Qwen2.5-1.5B-Instruct"
            device = "mps"
            quant = "false"
            speed = "Fast (2-3s)"
            quality = "Very Good"
        else:
            model = "Qwen/Qwen2.5-0.5B-Instruct"
            device = "mps"
            quant = "false"
            speed = "Very Fast (1-2s)"
            quality = "Good"
    
    else:
        # CPU only
        if ram_gb >= 32:
            model = "Qwen/Qwen2.5-1.5B-Instruct"
            device = "cpu"
            quant = "false"
            speed = "Slow (8-12s)"
            quality = "Very Good"
        else:
            model = "Qwen/Qwen2.5-0.5B-Instruct"
            device = "cpu"
            quant = "false"
            speed = "Medium (5-8s)"
            quality = "Good"
    
    print(f"\n✅ Recommended Model: {model}")
    print(f"✅ Device: {device}")
    print(f"✅ 4-bit Quantization: {quant}")
    print(f"\n📊 Expected Performance:")
    print(f"   Speed: {speed}")
    print(f"   Quality: {quality}")
    
    # Generate export commands
    print("\n" + "="*60)
    print("CONFIGURATION COMMANDS")
    print("="*60)
    
    shell = os.environ.get('SHELL', '/bin/bash')
    is_fish = 'fish' in shell
    
    if is_fish:
        print("\n# For Fish shell:")
        print(f'set -x SPOTTER_MODEL "{model}"')
        print(f'set -x DEVICE_MAP "{device}"')
        print(f'set -x LOAD_IN_4BIT "{quant}"')
    else:
        print("\n# For Bash/Zsh:")
        print(f'export SPOTTER_MODEL="{model}"')
        print(f'export DEVICE_MAP="{device}"')
        print(f'export LOAD_IN_4BIT="{quant}"')
    
    print("\n# Or run this Python script:")
    print("python configure.py --apply")
    
    return {
        'model': model,
        'device': device,
        'quantization': quant,
        'has_cuda': has_cuda,
        'has_mps': has_mps,
        'is_apple_silicon': is_apple_silicon
    }

def apply_config(config: dict):
    """Apply configuration to environment"""
    os.environ['SPOTTER_MODEL'] = config['model']
    os.environ['DEVICE_MAP'] = config['device']
    os.environ['LOAD_IN_4BIT'] = config['quantization']
    
    print("\n✅ Configuration applied to current session!")
    print("\nYou can now run:")
    print("  python hybrid_orchestrator.py interactive")

def save_config(config: dict):
    """Save configuration to .env file"""
    with open('.env', 'w') as f:
        f.write(f"# GymBot Configuration - Auto-generated\n")
        f.write(f"SPOTTER_MODEL={config['model']}\n")
        f.write(f"DEVICE_MAP={config['device']}\n")
        f.write(f"LOAD_IN_4BIT={config['quantization']}\n")
    
    print("\n✅ Configuration saved to .env file")
    print("\nTo use in future sessions:")
    print("  source .env  # or")
    print("  python configure.py --apply")

if __name__ == "__main__":
    import sys
    
    config = detect_hardware()
    
    if '--apply' in sys.argv:
        apply_config(config)
    elif '--save' in sys.argv:
        save_config(config)
    else:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Apply these settings:")
        
        shell = os.environ.get('SHELL', '/bin/bash')
        is_fish = 'fish' in shell
        
        if is_fish:
            print(f'   set -x SPOTTER_MODEL "{config["model"]}"')
            print(f'   set -x DEVICE_MAP "{config["device"]}"')
            print(f'   set -x LOAD_IN_4BIT "{config["quantization"]}"')
        else:
            print(f'   export SPOTTER_MODEL="{config["model"]}"')
            print(f'   export DEVICE_MAP="{config["device"]}"')
            print(f'   export LOAD_IN_4BIT="{config["quantization"]}"')
        
        print("\n2. Or run:")
        print("   python configure.py --apply")
        print("\n3. Then start the system:")
        print("   python hybrid_orchestrator.py interactive")
        
        print("\n💡 Tip: Run 'python configure.py --save' to save these settings")
