#!/usr/bin/env python3
"""
Custom Configuration Example for ECON Framework

This example shows how to create and modify configurations
for different use cases and experiment setups.
"""

import yaml
import os
from pathlib import Path

def create_custom_config():
    """
    Create a custom configuration file for specific needs.
    """
    
    print("⚙️ ECON Custom Configuration Example")
    print("=" * 50)
    
    # Example 1: High-performance configuration
    high_performance_config = {
        "runner": "episode_runner",
        "mac": "basic_mac", 
        "learner": "q_learner",
        "n_agents": 5,  # More agents for complex problems
        "belief_dim": 256,  # Larger belief representations
        "text_embed_dim": 1024,
        
        # Extended training
        "t_max": 5000000,
        "test_interval": 100000,
        
        # Environment for complex math
        "env": "huggingface_dataset_env",
        "env_args": {
            "hf_dataset_path": "competition_math",
            "dataset_split": "train", 
            "question_field_name": "problem",
            "answer_field_name": "solution",
            "max_question_length": 2048,  # Longer problems
            "max_answer_length": 1024,
            "use_dataset_episode": True
        },
        
        # Advanced coordination
        "bne_max_iterations": 10,  # More coordination rounds
        "bne_convergence_threshold": 0.005,  # Stricter convergence
        "stage2_weight": 0.4,
        
        # Optimized learning rates
        "lr": 0.0005,
        "belief_net_lr": 0.0003,
        "encoder_lr": 0.0005,
        "mixer_lr": 0.0003,
        
        "train": {
            "update_interval": 5,  # More frequent updates
            "batch_size": 32,  # Larger batches
            "buffer_size": 64,
            "learning_rate": 0.0005
        }
    }
    
    # Example 2: Fast experimentation configuration  
    fast_experiment_config = {
        "runner": "episode_runner",
        "mac": "basic_mac",
        "learner": "q_learner", 
        "n_agents": 3,  # Standard number
        "belief_dim": 64,  # Smaller for speed
        "text_embed_dim": 256,
        
        # Quick training
        "t_max": 100000,
        "test_interval": 10000,
        
        # Simple environment
        "env": "huggingface_dataset_env",
        "env_args": {
            "hf_dataset_path": "gsm8k",
            "dataset_split": "train",
            "max_question_length": 512,
            "max_answer_length": 256,
            "use_random_sampling": True,  # Random sampling for speed
            "use_dataset_episode": False  # Episode per question
        },
        
        # Minimal coordination
        "bne_max_iterations": 3,
        "bne_convergence_threshold": 0.02,
        "stage2_weight": 0.2,
        
        # Higher learning rates for fast convergence
        "lr": 0.002,
        "train": {
            "update_interval": 2,
            "batch_size": 8,
            "buffer_size": 16,
            "learning_rate": 0.002
        }
    }
    
    # Save configurations
    configs_dir = Path("examples/configs")
    configs_dir.mkdir(exist_ok=True)
    
    with open(configs_dir / "high_performance.yaml", "w") as f:
        yaml.dump(high_performance_config, f, default_flow_style=False, indent=2)
    
    with open(configs_dir / "fast_experiment.yaml", "w") as f:
        yaml.dump(fast_experiment_config, f, default_flow_style=False, indent=2)
    
    print("✅ Created configuration files:")
    print("- examples/configs/high_performance.yaml")
    print("- examples/configs/fast_experiment.yaml")
    
    print("\n🚀 Usage examples:")
    print("# High-performance training")
    print("./run_econ.sh --config examples/configs/high_performance.yaml --api-key $TOGETHER_API_KEY")
    
    print("\n# Fast experimentation")  
    print("./run_econ.sh --config examples/configs/fast_experiment.yaml --api-key $TOGETHER_API_KEY")

def modify_existing_config():
    """
    Show how to modify existing configurations programmatically.
    """
    
    print("\n🔧 Modifying Existing Configurations")
    print("=" * 40)
    
    # Load base config
    base_config_path = "src/config/config.yaml"
    
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)
        
        print("📁 Loaded base configuration")
        
        # Example modifications
        modifications = [
            ("n_agents", 5, "Increase agents for better coordination"),
            ("belief_dim", 256, "Larger belief representations"),
            ("train.update_interval", 5, "More frequent updates"),
            ("bne_max_iterations", 8, "More coordination rounds")
        ]
        
        print("\n📝 Applying modifications:")
        for key, value, description in modifications:
            print(f"- {key}: {value} ({description})")
            
            # Handle nested keys
            if "." in key:
                keys = key.split(".")
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        # Save modified config
        modified_path = "examples/configs/modified_base.yaml"
        with open(modified_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"\n✅ Saved modified configuration to: {modified_path}")
    else:
        print(f"❌ Base configuration not found: {base_config_path}")

if __name__ == "__main__":
    create_custom_config()
    modify_existing_config() 