#!/usr/bin/env python3
"""
Installation script for image generation dependencies.
Run this script to install the required packages for the image regeneration feature.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main installation function."""
    print("🎨 Installing Image Generation Dependencies")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: You are not in a virtual environment.")
        print("   It's recommended to activate your virtual environment first:")
        print("   source venv/bin/activate  # On Linux/Mac")
        print("   venv\\Scripts\\activate     # On Windows")
        
        response = input("\nContinue anyway? (y/N): ").lower().strip()
        if response != 'y':
            print("Installation cancelled.")
            return
    
    # Install diffusers and accelerate
    success = True
    
    commands = [
        ("pip install diffusers>=0.30.0", "Installing diffusers"),
        ("pip install accelerate>=0.25.0", "Installing accelerate"),
        ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch (CPU version)")
    ]
    
    # Check if CUDA is available and install GPU version if possible
    try:
        import torch
        if torch.cuda.is_available():
            commands[-1] = ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch (GPU version)")
            print("🚀 CUDA detected! Installing GPU-accelerated PyTorch...")
    except ImportError:
        print("📦 Installing CPU version of PyTorch...")
    
    for command, description in commands:
        if not run_command(command, description):
            success = False
            break
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Restart your backend server: python run_backend.py")
        print("2. The image regeneration feature will be available in the analysis results")
        print("\nNote: First-time model loading may take a few minutes as it downloads the Qwen-Image-Edit-2509 model.")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🚀 GPU acceleration available: {torch.cuda.get_device_name(0)}")
            else:
                print("💻 Running on CPU (GPU acceleration not available)")
        except ImportError:
            pass
            
    else:
        print("\n❌ Installation failed. Please check the error messages above.")
        print("You may need to install dependencies manually:")
        print("pip install diffusers>=0.30.0 accelerate>=0.25.0")

if __name__ == "__main__":
    main()
