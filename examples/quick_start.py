#!/usr/bin/env python3
"""
ECON Framework Quick Start Example

This script demonstrates how to use ECON for multi-agent coordination
on mathematical reasoning tasks.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train import main, load_config, update_config_with_args
from types import SimpleNamespace

def quick_start_example(api_key: str, n_agents: int = 3, dataset: str = "gsm8k"):
    """
    Run a quick start example with ECON.
    
    Args:
        api_key: Together AI API key
        n_agents: Number of agents to use
        dataset: Dataset name (gsm8k, competition_math, etc.)
    """
    
    print(f"🚀 Starting ECON Quick Start Example")
    print(f"   📊 Dataset: {dataset}")
    print(f"   🤖 Agents: {n_agents}")
    print(f"   🔑 API Key: {api_key[:10]}...")
    print()
    
    # Create a minimal configuration
    config = SimpleNamespace()
    
    # Basic settings
    config.runner = "episode_runner"
    config.mac = "basic_mac"
    config.learner = "q_learner"
    config.n_agents = n_agents
    config.env = "huggingface_dataset_env"
    
    # Quick training settings
    config.t_max = 100000  # Short training for demo
    config.test_interval = 10000
    config.text_embed_dim = 512
    config.belief_dim = 64
    config.batch_size_run = 1
    config.state_shape = 512
    config.n_actions = 2
    config.lr = 0.002
    
    # Environment configuration
    config.env_args = SimpleNamespace()
    config.env_args.hf_dataset_path = dataset
    config.env_args.hf_dataset_config_name = "main"
    config.env_args.dataset_split = "train"
    config.env_args.question_field_name = "question"
    config.env_args.answer_field_name = "answer"
    config.env_args.max_question_length = 512
    config.env_args.max_answer_length = 200
    config.env_args.dataset_streaming = False
    
    # LLM configuration
    config.together_api_key = api_key
    config.llm = SimpleNamespace()
    config.llm.together_api_key = api_key
    config.llm.coordinator_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    config.llm.executor_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    config.llm.max_tokens = 1024
    
    # System configuration
    config.system = SimpleNamespace()
    config.system.use_cuda = True
    config.system.device_num = 0
    config.system.seed = 42
    config.system.debug = False
    
    # Logging configuration
    config.logging = SimpleNamespace()
    config.logging.use_tensorboard = True
    config.logging.log_interval = 1000
    config.logging.save_model = True
    config.logging.save_model_interval = 5000
    config.logging.checkpoint_path = f"./models/quick_start_{n_agents}agents"
    config.logging.log_path = f"./logs/quick_start_{n_agents}agents"
    config.logging.experiment_name = f"quick_start_{n_agents}agents_{dataset}"
    
    # Training configuration
    config.train = SimpleNamespace()
    config.train.episodes_per_task = 25
    config.train.buffer_size = 8
    config.train.batch_size = 4
    config.train.update_interval = 2
    config.train.optimizer = "adam"
    config.train.learning_rate = 0.002
    config.train.coordinator_learning_rate = 0.001
    config.train.gamma = 0.95
    
    # Architecture configuration
    config.arch = SimpleNamespace()
    config.arch.entity_dim = 128
    config.arch.attention_heads = 2
    config.arch.transformer_blocks = 1
    config.arch.key_dim = 64
    config.arch.mlp_hidden_size = 128
    config.arch.feedforward_size = 512
    config.arch.dropout_rate = 0.1
    config.arch.layer_norm_epsilon = 0.00001
    
    # Early stopping
    config.early_stopping = SimpleNamespace()
    config.early_stopping.commitment_threshold = 0.02
    config.early_stopping.loss_threshold = 0.001
    config.early_stopping.reward_threshold = 0.6
    config.early_stopping.patience = 3
    
    # Run the training
    try:
        print("🏃 Starting training...")
        main_with_config(config)
        print("✅ Quick start example completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

def main_with_config(config):
    """Modified main function that accepts a config object directly"""
    from train import setup_experiment, run_training, setup_wandb
    
    # Setup experiment
    runner, mac, learner, logger, device = setup_experiment(config)
    
    # Setup wandb if needed (disabled for quick start)
    # setup_wandb(config, logger)
    
    # Run training
    run_training(config, runner, learner, logger, device)

def demonstrate_multi_agent_coordination():
    """
    Demonstrate the key concepts of multi-agent coordination.
    """
    print("🧠 Multi-Agent Coordination Concepts:")
    print()
    print("1. 🤝 Belief-Based Coordination:")
    print("   - Agents share beliefs instead of explicit messages")
    print("   - Reduces communication overhead significantly")
    print("   - Enables scalable coordination")
    print()
    print("2. 🎯 Nash Equilibrium Framework:")
    print("   - Guarantees convergence to stable solutions")
    print("   - Each agent optimizes given others' strategies")
    print("   - Theoretical foundation for coordination")
    print()
    print("3. 🏗️ Two-Stage Architecture:")
    print("   - Stage 1: Individual agent analysis")
    print("   - Stage 2: Coordinated solution synthesis")
    print("   - BNE-guided belief updates")
    print()
    print("4. 📊 Dynamic Reward System:")
    print("   - Action Likelihood (AL): Consistency rewards")
    print("   - Task Specific (TS): Domain performance")
    print("   - Collaborative Contribution (CC): Coordination quality")
    print()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ECON Quick Start Example')
    parser.add_argument('--api-key', type=str, 
                       help='Together AI API key (required unless --demo-only)')
    parser.add_argument('--agents', type=int, default=3,
                       help='Number of agents (default: 3)')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=['gsm8k', 'competition_math', 'mathqa'],
                       help='Dataset to use (default: gsm8k)')
    parser.add_argument('--demo-only', action='store_true',
                       help='Only show concept demonstration, no training')
    
    args = parser.parse_args()
    
    # Validate that API key is provided unless demo-only
    if not args.demo_only and not args.api_key:
        parser.error("--api-key is required unless --demo-only is specified")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Always show the concept demonstration
    demonstrate_multi_agent_coordination()
    
    if not args.demo_only:
        # Validate API key
        if not args.api_key or len(args.api_key) < 10:
            print("❌ Please provide a valid Together AI API key")
            sys.exit(1)
        
        # Set environment variable
        os.environ['TOGETHER_API_KEY'] = args.api_key
        
        # Run the quick start example
        quick_start_example(
            api_key=args.api_key,
            n_agents=args.agents,
            dataset=args.dataset
        )
    else:
        print("💡 To run actual training, use: python quick_start.py --api-key YOUR_KEY") 