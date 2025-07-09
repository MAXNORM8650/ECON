#!/usr/bin/env python3
"""
Performance Analysis Tutorial for ECON Framework

This example shows how to analyze training results, 
coordination patterns, and system performance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def analyze_training_logs():
    """
    Analyze training logs for performance insights.
    """
    
    print("📊 ECON Performance Analysis Tutorial")
    print("=" * 50)
    
    print("🔍 Log Analysis Steps:")
    print("1. Parse training logs")
    print("2. Extract key metrics")
    print("3. Identify patterns and trends")
    print("4. Generate performance reports")
    
    # Simulate training data analysis
    simulate_training_analysis()

def simulate_training_analysis():
    """
    Simulate analysis of training data.
    """
    
    print("\n📈 Training Progress Analysis")
    print("-" * 35)
    
    # Simulate training metrics
    episodes = np.arange(1, 101)
    
    # Task-specific rewards (should increase)
    ts_rewards = 0.3 + 0.5 * (1 - np.exp(-episodes/30)) + np.random.normal(0, 0.05, 100)
    ts_rewards = np.clip(ts_rewards, 0, 1)
    
    # Action likelihood rewards (should stabilize high)
    al_rewards = 0.2 + 0.6 * (1 - np.exp(-episodes/20)) + np.random.normal(0, 0.03, 100)
    al_rewards = np.clip(al_rewards, 0, 1)
    
    # Collaborative contribution rewards
    cc_rewards = 0.1 + 0.4 * (1 - np.exp(-episodes/40)) + np.random.normal(0, 0.04, 100)
    cc_rewards = np.clip(cc_rewards, 0, 1)
    
    # Total rewards
    total_rewards = 0.5 * ts_rewards + 0.3 * al_rewards + 0.2 * cc_rewards
    
    print("Performance Summary (last 10 episodes):")
    print(f"  TS Reward: {np.mean(ts_rewards[-10:]):.3f} ± {np.std(ts_rewards[-10:]):.3f}")
    print(f"  AL Reward: {np.mean(al_rewards[-10:]):.3f} ± {np.std(al_rewards[-10:]):.3f}")  
    print(f"  CC Reward: {np.mean(cc_rewards[-10:]):.3f} ± {np.std(cc_rewards[-10:]):.3f}")
    print(f"  Total:     {np.mean(total_rewards[-10:]):.3f} ± {np.std(total_rewards[-10:]):.3f}")
    
    # Analyze convergence
    analyze_convergence(episodes, total_rewards)
    
    # Create performance plots
    create_performance_plots(episodes, ts_rewards, al_rewards, cc_rewards, total_rewards)

def analyze_convergence(episodes, rewards):
    """
    Analyze training convergence patterns.
    """
    
    print("\n🎯 Convergence Analysis")
    print("-" * 25)
    
    # Calculate moving average
    window_size = 10
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    # Find convergence point (when improvement becomes minimal)
    improvement_threshold = 0.01
    convergence_episode = None
    
    for i in range(len(moving_avg) - 5):
        recent_improvement = moving_avg[i+5] - moving_avg[i]
        if recent_improvement < improvement_threshold:
            convergence_episode = i + window_size
            break
    
    if convergence_episode:
        print(f"✅ Training converged around episode {convergence_episode}")
        print(f"   Final performance: {moving_avg[-1]:.3f}")
        print(f"   Convergence speed: {'Fast' if convergence_episode < 30 else 'Normal'}")
    else:
        print("⚠️ Training still improving - consider longer training")
    
    # Stability analysis
    recent_std = np.std(rewards[-20:])
    print(f"   Performance stability: {recent_std:.4f} (lower is better)")
    
    if recent_std < 0.05:
        print("   ✅ Stable performance achieved")
    else:
        print("   ⚠️ Performance still fluctuating")

def create_performance_plots(episodes, ts_rewards, al_rewards, cc_rewards, total_rewards):
    """
    Create performance visualization plots.
    """
    
    print("\n📊 Creating Performance Plots")
    print("-" * 32)
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Individual reward components
        ax1.plot(episodes, ts_rewards, label='TS Reward', alpha=0.7)
        ax1.plot(episodes, al_rewards, label='AL Reward', alpha=0.7)
        ax1.plot(episodes, cc_rewards, label='CC Reward', alpha=0.7)
        ax1.set_title('Individual Reward Components')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Total reward with moving average
        ax2.plot(episodes, total_rewards, alpha=0.5, label='Total Reward')
        window_size = 10
        if len(total_rewards) >= window_size:
            moving_avg = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=2, label=f'{window_size}-Episode Average')
        ax2.set_title('Total Reward Progress')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Reward distribution
        ax3.hist(total_rewards, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(total_rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(total_rewards):.3f}')
        ax3.set_title('Reward Distribution')
        ax3.set_xlabel('Total Reward')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance improvement over time
        cumulative_mean = np.cumsum(total_rewards) / np.arange(1, len(total_rewards) + 1)
        ax4.plot(episodes, cumulative_mean, 'g-', linewidth=2)
        ax4.set_title('Cumulative Average Performance')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Average Reward')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        output_dir = Path("examples/outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "performance_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance plots saved to: examples/outputs/performance_analysis.png")
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping plot creation")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")

def analyze_coordination_patterns():
    """
    Analyze multi-agent coordination effectiveness.
    """
    
    print("\n🤝 Coordination Pattern Analysis")
    print("-" * 35)
    
    # Simulate coordination metrics
    n_episodes = 50
    n_agents = 3
    
    # Belief convergence over time
    convergence_times = np.random.gamma(3, 2, n_episodes) + 1
    
    # Agent agreement rates
    agreement_rates = np.random.beta(8, 2, n_episodes)
    
    # Coordination efficiency
    efficiency_scores = np.random.beta(6, 3, n_episodes)
    
    print("Coordination Metrics Summary:")
    print(f"  Average convergence time: {np.mean(convergence_times):.2f} iterations")
    print(f"  Agent agreement rate: {np.mean(agreement_rates):.3f}")
    print(f"  Coordination efficiency: {np.mean(efficiency_scores):.3f}")
    
    # Identify coordination patterns
    if np.mean(convergence_times) < 5:
        print("  ✅ Fast convergence - effective coordination")
    elif np.mean(convergence_times) < 10:
        print("  ⚠️ Moderate convergence - room for improvement")
    else:
        print("  ❌ Slow convergence - coordination issues")
    
    if np.mean(agreement_rates) > 0.8:
        print("  ✅ High agent agreement - consistent reasoning")
    else:
        print("  ⚠️ Low agent agreement - check belief sharing")

def generate_performance_report():
    """
    Generate a comprehensive performance report.
    """
    
    print("\n📋 Performance Report Generation")
    print("-" * 35)
    
    # Simulated performance data
    metrics = {
        "training_episodes": 1000,
        "final_accuracy": 0.847,
        "convergence_episode": 687,
        "average_reward": 0.783,
        "coordination_efficiency": 0.812,
        "belief_convergence_time": 4.2,
        "agent_agreement_rate": 0.885
    }
    
    # Generate report content
    report_content = f"""
# ECON Training Performance Report

## Training Summary
- **Total Episodes**: {metrics['training_episodes']}
- **Final Accuracy**: {metrics['final_accuracy']:.1%}
- **Convergence Episode**: {metrics['convergence_episode']}
- **Average Reward**: {metrics['average_reward']:.3f}

## Coordination Analysis
- **Efficiency Score**: {metrics['coordination_efficiency']:.3f}
- **Belief Convergence Time**: {metrics['belief_convergence_time']:.1f} iterations
- **Agent Agreement Rate**: {metrics['agent_agreement_rate']:.1%}

## Performance Assessment
"""
    
    # Add assessment based on metrics
    if metrics['final_accuracy'] > 0.8:
        report_content += "✅ **Excellent Performance** - High accuracy achieved\n"
    elif metrics['final_accuracy'] > 0.6:
        report_content += "⚠️ **Good Performance** - Acceptable accuracy with room for improvement\n"
    else:
        report_content += "❌ **Poor Performance** - Significant improvement needed\n"
    
    if metrics['coordination_efficiency'] > 0.75:
        report_content += "✅ **Effective Coordination** - Agents working well together\n"
    else:
        report_content += "⚠️ **Coordination Issues** - Consider adjusting BNE parameters\n"
    
    # Save report
    output_dir = Path("examples/outputs")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "performance_report.md", "w") as f:
        f.write(report_content)
    
    print("✅ Performance report saved to: examples/outputs/performance_report.md")
    print("\n📊 Key Insights:")
    print(f"- Training converged after {metrics['convergence_episode']} episodes")
    print(f"- Final system accuracy: {metrics['final_accuracy']:.1%}")
    print(f"- Coordination effectiveness: {metrics['coordination_efficiency']:.1%}")

def troubleshooting_guide():
    """
    Provide troubleshooting guidance based on common issues.
    """
    
    print("\n🔧 Performance Troubleshooting Guide")
    print("-" * 40)
    
    issues_and_solutions = [
        {
            "issue": "Low Task-Specific (TS) Rewards",
            "symptoms": ["TS rewards below 0.3", "Poor answer accuracy"],
            "solutions": [
                "Check answer format validation (\\boxed{} requirement)",
                "Verify API key and model access",
                "Review dataset quality and question complexity",
                "Adjust coordinator prompt templates"
            ]
        },
        {
            "issue": "Low Action Likelihood (AL) Rewards", 
            "symptoms": ["AL rewards below 0.5", "High agent disagreement"],
            "solutions": [
                "Increase coordination iterations (bne_max_iterations)",
                "Adjust belief sharing mechanisms",
                "Check agent response consistency",
                "Review strategy generation quality"
            ]
        },
        {
            "issue": "Slow Convergence",
            "symptoms": ["Training doesn't stabilize", "Rewards keep fluctuating"],
            "solutions": [
                "Reduce learning rates",
                "Increase gradient update interval",
                "Add early stopping criteria",
                "Use smaller batch sizes"
            ]
        }
    ]
    
    for issue_data in issues_and_solutions:
        print(f"\n🚨 {issue_data['issue']}")
        print("   Symptoms:")
        for symptom in issue_data['symptoms']:
            print(f"   - {symptom}")
        print("   Solutions:")
        for solution in issue_data['solutions']:
            print(f"   • {solution}")

if __name__ == "__main__":
    analyze_training_logs()
    analyze_coordination_patterns()
    generate_performance_report()
    troubleshooting_guide() 