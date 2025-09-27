#!/usr/bin/env python3
"""
Alternative entry point for the Multimodal Design Analysis Suite backend.
This file provides compatibility with various deployment platforms.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main function from run_backend.py
from run_backend import main

if __name__ == "__main__":
    main()
