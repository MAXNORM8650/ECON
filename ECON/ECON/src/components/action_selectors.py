# components/action_selectors.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class MultinomialActionSelector:
    """
    Selects actions using a multinomial distribution over q-values.
    Supports epsilon-greedy exploration.
    """
    def __init__(self, args):
        self.args = args
        self.schedule_start = getattr(args, "epsilon_start", 1.0)
        self.schedule_finish = getattr(args, "epsilon_finish", 0.05)
        self.schedule_timesteps = getattr(args, "epsilon_anneal_time", 50000)
        self.epsilon = self.schedule_start
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs: torch.Tensor,
                    avail_actions: torch.Tensor,
                    t_env: int,
                    test_mode: bool = False) -> torch.Tensor:
        """
        Select actions based on multinomial sampling.
        
        Args:
            agent_inputs: Q-values or policy outputs
            avail_actions: Available actions mask
            t_env: Current environment timestep
            test_mode: Whether in testing mode
            
        Returns:
            Selected actions
        """
        masked_q_values = self._mask_actions(agent_inputs, avail_actions)
        
        # Epsilon schedule
        if test_mode and self.test_greedy:
            epsilon = 0.0
        else:
            epsilon = self.epsilon if t_env <= self.schedule_timesteps else self.schedule_finish
            
        # Random exploration
        random_numbers = torch.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < epsilon).long()
        
        # Get random and greedy actions
        random_actions = self._get_random_actions(avail_actions)
        greedy_actions = masked_q_values.max(dim=2)[1]
        
        # Combine random and greedy actions
        chosen_actions = pick_random * random_actions + (1 - pick_random) * greedy_actions
        
        return chosen_actions

    def _mask_actions(self, agent_inputs: torch.Tensor,
                     avail_actions: torch.Tensor) -> torch.Tensor:
        """Apply action masking."""
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")
        return masked_q_values

    def _get_random_actions(self, avail_actions: torch.Tensor) -> torch.Tensor:
        """Sample random available actions."""
        # Replace zeros with very small value to avoid division by zero
        avail_actions_nonzero = avail_actions + 1e-10
        
        # Normalize to create probability distribution
        probs = avail_actions_nonzero / avail_actions_nonzero.sum(dim=-1, keepdim=True)
        
        # Sample from distribution
        return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[0], -1)

    def epsilon_decay(self, t_env: int):
        """Update epsilon according to schedule."""
        if t_env <= self.schedule_timesteps:
            self.epsilon = self.schedule_start - (self.schedule_start - self.schedule_finish) * (
                t_env / self.schedule_timesteps
            )

class GaussianActionSelector:
    """
    Selects actions using a Gaussian distribution.
    Suitable for continuous action spaces.
    """
    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)
        self.gaussian_std = getattr(args, "gaussian_std", 0.1)
        self.std_decay_rate = getattr(args, "std_decay_rate", 0.99)
        self.min_std = getattr(args, "min_std", 0.02)

    def select_action(self, agent_inputs: torch.Tensor,
                     avail_actions: Optional[torch.Tensor] = None,
                     t_env: Optional[int] = None,
                     test_mode: bool = False) -> torch.Tensor:
        """
        Select actions using Gaussian noise.
        
        Args:
            agent_inputs: Mean actions from policy
            avail_actions: Not used in continuous case
            t_env: Current environment timestep
            test_mode: Whether in testing mode
            
        Returns:
            Selected actions with noise
        """
        if test_mode and self.test_greedy:
            return agent_inputs
            
        # Add Gaussian noise
        noise = torch.randn_like(agent_inputs) * self.gaussian_std
        actions = agent_inputs + noise
        
        # Clip actions if needed
        if hasattr(self.args, "action_range"):
            actions = torch.clamp(actions, 
                                min=self.args.action_range[0],
                                max=self.args.action_range[1])
        
        return actions

    def update_std(self, t_env: int):
        """Decay standard deviation over time."""
        self.gaussian_std = max(
            self.gaussian_std * self.std_decay_rate,
            self.min_std
        )

REGISTRY = {
    "multinomial": MultinomialActionSelector,
    "gaussian": GaussianActionSelector
}