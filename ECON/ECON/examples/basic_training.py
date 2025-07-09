#!/usr/bin/env python3
"""
Basic Training Example for ECON Framework

This example demonstrates the simplest way to start training 
with ECON using default configurations.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def basic_training_example():
    """
    Run basic training with minimal configuration.
    
    This example shows:
    1. How to set up environment variables
    2. Basic configuration loading
    3. Simple training execution
    """
    
    print("🎯 ECON Basic Training Example")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("❌ Error: TOGETHER_API_KEY environment variable not set")
        print("Please set it with: export TOGETHER_API_KEY='your_key_here'")
        return False
    
    print(f"✅ API Key: {api_key[:10]}...")
    
    # Example command to run
    cmd = f"""
    ./run_econ.sh \\
        --api-key {api_key} \\
        --config src/config/config.yaml \\
        --agents 3 \\
        --seed 42 \\
        --experiment-name "basic-training-example"
    """
    
    print("\n📋 Command to run basic training:")
    print(cmd)
    
    print("\n🔧 Configuration details:")
    print("- Dataset: GSM8K (grade school math)")
    print("- Agents: 3 executors + 1 coordinator")
    print("- Model: Llama-3.3-70B-Instruct-Turbo")
    print("- Training: Full dataset traversal")
    print("- Updates: Every 10 steps")
    
    print("\n⏱️ Expected execution:")
    print("- Setup time: 30-60 seconds")
    print("- Per episode: 10-20 seconds")
    print("- Total time: 2-4 hours for full dataset")
    
    return True

if __name__ == "__main__":
    basic_training_example() 