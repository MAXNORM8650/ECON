# ECON Framework Examples

This directory contains practical examples and tutorials for using the ECON framework.

## 🚀 Quick Start

### 1. Basic Example
Run the quick start script to see ECON in action:

```bash
# Run concept demonstration only
python examples/quick_start.py --demo-only

# Run actual training with 3 agents
python examples/quick_start.py --api-key YOUR_API_KEY --agents 3

# Try with different datasets
python examples/quick_start.py --api-key YOUR_API_KEY --dataset competition_math
```

### 2. Using Configuration Files

```bash
# Large-scale coordination (8 agents)
./run_econ.sh --config examples/configs/large_scale.yaml --api-key YOUR_API_KEY

# Fast training for experiments
./run_econ.sh --config examples/configs/fast_training.yaml --api-key YOUR_API_KEY
```

## 📁 Directory Structure

```
examples/
├── README.md                    # This file
├── quick_start.py              # Interactive quick start script
├── configs/                    # Configuration examples
│   ├── large_scale.yaml       # 8-agent coordination setup
│   └── fast_training.yaml     # Quick training configuration
└── notebooks/                 # Jupyter notebooks (future)
    ├── getting_started.ipynb  # Basic usage tutorial
    ├── advanced_coordination.ipynb
    └── performance_analysis.ipynb
```

## 🔧 Configuration Examples

### Large-Scale Coordination (`configs/large_scale.yaml`)

Optimized for complex problems requiring many agents:
- **8 agents** for diverse perspectives
- **Larger embeddings** (2048D) for complex representations
- **Extended training** (3M steps) for convergence
- **Lower learning rates** (0.0005) for stability

**Use case**: Complex mathematical proofs, multi-step reasoning

```bash
./run_econ.sh --config examples/configs/large_scale.yaml --api-key YOUR_KEY
```

### Fast Training (`configs/fast_training.yaml`)

Optimized for quick experimentation:
- **3 agents** for standard coordination
- **Smaller embeddings** (512D) for speed
- **Short training** (500K steps) for rapid results
- **Higher learning rates** (0.002) for fast convergence

**Use case**: Algorithm development, hyperparameter tuning

```bash
./run_econ.sh --config examples/configs/fast_training.yaml --api-key YOUR_KEY
```

## 🎓 Learning Path

### Beginner
1. **Concept Demo**: `python examples/quick_start.py --demo-only`
2. **First Training**: `python examples/quick_start.py --api-key YOUR_KEY`
3. **Configuration**: Try different configs in `examples/configs/`

### Intermediate
1. **Custom Configs**: Modify existing YAML files
2. **Different Datasets**: Try GSM8K, MATH, MathQA
3. **Agent Scaling**: Experiment with 3, 5, 8 agents

### Advanced
1. **Custom Environments**: Create new problem types
2. **Custom Agents**: Implement specialized reasoning
3. **Analysis Tools**: Build evaluation metrics

## 📊 Expected Results

### Fast Training (3 agents, 500K steps)
- **Training time**: ~2-4 hours (with GPU)
- **Memory usage**: ~8-12 GB GPU RAM
- **Convergence**: Usually by 300K steps

### Large Scale (8 agents, 3M steps)
- **Training time**: ~12-24 hours (with GPU)
- **Memory usage**: ~16-24 GB GPU RAM
- **Convergence**: Usually by 2M steps

## 🛠️ Customization Examples

### Adding New Datasets

```python
# In your config file
env_args:
  hf_dataset_path: "your_dataset_name"
  dataset_split: "train"
  question_field_name: "problem"  # Adjust field names
  answer_field_name: "solution"
```

### Scaling Agents

```yaml
# For more agents, adjust these parameters:
n_agents: 10
text_embed_dim: 2048  # Larger embeddings
belief_dim: 512       # Larger belief states
arch:
  attention_heads: 8  # More attention heads
  transformer_blocks: 4  # Deeper networks
```

### Custom Reward Weights

```yaml
reward:
  al_weight: 0.3   # Action Likelihood
  ts_weight: 0.5   # Task Specific  
  cc_weight: 0.2   # Collaborative Contribution
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   export TOGETHER_API_KEY="your_key_here"
   ```

2. **Memory Issues**
   - Reduce `n_agents` or `text_embed_dim`
   - Use `fast_training.yaml` configuration

3. **Slow Training**
   - Increase learning rates
   - Reduce `episodes_per_task`
   - Use smaller embeddings

4. **Import Errors**
   - Ensure you're in the project root directory
   - Check that all dependencies are installed

### Performance Tips

1. **GPU Optimization**
   - Use CUDA if available
   - Monitor GPU memory usage
   - Adjust batch sizes accordingly

2. **Training Efficiency**
   - Start with fast configuration
   - Use early stopping
   - Monitor convergence metrics

3. **Debugging**
   - Enable debug mode in config
   - Use frequent logging
   - Check tensorboard outputs

## 📈 Monitoring Training

### Weights & Biases Integration

```bash
./run_econ.sh \
  --config examples/configs/fast_training.yaml \
  --api-key YOUR_KEY \
  --wandb \
  --wandb-project "ECON-Experiments"
```

### Key Metrics to Watch

- **Loss convergence**: Should decrease steadily
- **Reward trends**: Task-specific rewards should improve
- **Coordination efficiency**: CC rewards indicate cooperation
- **Belief convergence**: Agents should reach consensus

## 🤝 Contributing Examples

We welcome community contributions! To add examples:

1. Fork the repository
2. Add your example to `examples/`
3. Update this README
4. Submit a pull request

### Example Types Needed
- Domain-specific applications
- Novel coordination strategies
- Performance optimization techniques
- Evaluation methodologies

---

For more help, check the main [README](../README.md) or open an issue on GitHub. 