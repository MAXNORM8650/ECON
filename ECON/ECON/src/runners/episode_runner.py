import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from utils.logging import Logger
from dataclasses import dataclass
from loguru import logger

@dataclass
class EpisodeMetrics:
    """Container for episode-specific metrics."""
    llm_responses: List[str] = None
    strategies: List[str] = None
    commitments: List[str] = None
    rewards: List[float] = None
    belief_states: List[torch.Tensor] = None
    rewards_al: List[float] = None
    rewards_ts: List[float] = None
    rewards_cc: List[float] = None
    
    def __post_init__(self):
        """Initialize empty lists."""
        self.llm_responses = []
        self.strategies = []
        self.commitments = []
        self.rewards = []
        self.belief_states = []
        self.rewards_al = []
        self.rewards_ts = []
        self.rewards_cc = []
    
    def add_step_data(self, extra_info: Dict[str, Any], 
                      reward: float, reward_al: float, reward_ts: float, reward_cc: float):
        """Add data from a single step."""
        if 'llm_responses' in extra_info:
            self.llm_responses.append(extra_info['llm_responses'])
        if 'strategy' in extra_info:
            self.strategies.append(extra_info['strategy'])
        if 'commitment' in extra_info:
            if isinstance(extra_info['commitment'], str):
                 self.commitments.append(extra_info['commitment'])
            elif isinstance(extra_info['commitment'], list):
                 self.commitments.extend(extra_info['commitment'])

        if 'belief_states' in extra_info:
            self.belief_states.append(extra_info['belief_states'])
        
        self.rewards.append(reward)
        self.rewards_al.append(reward_al)
        self.rewards_ts.append(reward_ts)
        self.rewards_cc.append(reward_cc)

class EpisodeRunner:
    """
    Episode runner for LLM-based MARL training.
    
    Handles episode execution, data collection, and coordination between
    environment interactions, LLM responses, and data storage.
    """
    def __init__(self, args: Any, logger: Logger):
        """
        Initialize episode runner.
        
        Args:
            args: Configuration arguments
            logger: Logger instance
        """
        self.args = args
        self.logger = logger
        
        # Environment and batch information
        self.env = None
        self.env_info = None
        self.batch = None
        
        # Training state
        self.t = 0  # Current timestep within episode
        self.t_env = 0  # Total timesteps across all episodes
        self.t_episodes = 0  # 添加episode计数器
        
        # Testing state
        self.test_returns = []
        self.train_returns = []
        self.last_test_t = 0
        self.last_save_t = 0
        
        # MAC and processing components
        self.mac = None
        self.batch_handler = None
        
        # Statistics tracking
        self.train_stats = {}
        self.test_stats = {}
        
        # Episode management
        self.episode_limit = 1  # Single step per episode for LLM environments
        self.n_agents = args.n_agents
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1, "Batch size for episode runner is usually 1."
        
        # Initialize environment using the registry and env_args from config
        self.env = self._init_environment()
        self.env_info = self.env.get_env_info()
        self.episode_limit = self.env_info["episode_limit"]
        self.obs_shape = self.env_info["obs_shape"]
        self.t = 0 # Step within the current episode
        
        # Initialize batch handling
        # max_seq_length for EpisodeBatch will be self.episode_limit + 1.
        # If episode_limit is 1 (one data sample = one step), max_seq_length is 2.
        self.new_batch = self._init_batch_handler()
        self.batch = self.new_batch()

    def setup(self, scheme: Dict, groups: Dict, preprocess: Any, mac: Any):
        """Setup with MAC."""
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.mac = mac  # 添加MAC
    
    def _init_environment(self):
        """Initialize and return the environment from the registry."""
        try:
            env_key = self.args.env
            # Prepare environment arguments including reward configuration
            # Convert SimpleNamespace to dict
            if hasattr(self.args.env_args, '__dict__'):
                env_kwargs = vars(self.args.env_args)
            else:
                env_kwargs = dict(self.args.env_args)
            
            # Add reward configuration if it exists
            if hasattr(self.args, 'reward'):
                env_kwargs['reward_config'] = self.args.reward
            
            return env_REGISTRY[env_key](**env_kwargs)
        except KeyError:
            self.logger.error(f"Environment '{self.args.env}' not found in registry. Available: {list(env_REGISTRY.keys())}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize environment '{self.args.env}': {e}")
            raise
    
    def _init_batch_handler(self):
        """Initialize and return the batch handler."""
        return partial(
            EpisodeBatch,
            scheme=self._build_scheme(),
            groups=self._build_groups(),
            batch_size=self.batch_size,
            max_seq_length=self.episode_limit + 1,
            device=self.args.device
        )

    def run(self, test_mode: bool = False) -> EpisodeBatch:
        """
        Run a complete episode (processing one data sample from the dataset).
        
        Args:
            test_mode: Whether in testing mode
            
        Returns:
            Collected episode data for the processed sample.
        """
        try:
            # Reset environment and MAC hidden state
            # For HuggingFaceDatasetEnv, reset() loads the next data sample (e.g., a question)
            # and sets self.env.current_question and self.env.current_sample.
            # The observation returned by self.env.reset() is self.env.current_question.
            current_obs, env_step_info = self.env.reset() # env_step_info contains the full sample
            self.reset_runner_state() # Resets self.batch and self.t
            
            if current_obs is None: # Should be handled by env.reset() raising StopIteration
                self.logger.warning("Environment reset returned None observation. Stopping run.")
                return self.batch # Return empty or partially filled batch

            episode_return = 0
            if hasattr(self.mac, 'init_hidden'):
                self.mac.init_hidden(batch_size=self.batch_size)
            
            metrics = EpisodeMetrics()
            
            # Since episode_limit is 1 for dataset-based env, this loop runs once.
            # self.t is 0 initially.
            # pre_transition_data uses current_obs from env.reset()
            pre_transition_data = self._get_pre_transition_data(current_obs)
            self.batch.update(pre_transition_data, ts=self.t) # ts=0
            
            # Get actions (LLM responses) and other info from MAC
            # MAC's select_actions will use self.batch at ts=0 which contains the tokenized obs
            # Pass the raw current_obs text as well for LLM prompt generation inside MAC
            discrete_actions, mac_extra_info = self._get_actions(test_mode, raw_observation_text=current_obs)
            
            if not test_mode:
                # 'discrete_actions' from MAC might be placeholder if actions are text-based.
                # We need the actual LLM answer generated by MAC, likely in mac_extra_info['llm_responses']
                # or a specific field like mac_extra_info['final_answer_from_coordinator']
                # For now, assuming mac_extra_info['llm_responses'] contains a list of answers from agents,
                # and we need a way to get the single "action" for env.step().
                # Let's assume for now the coordinator's final commitment or a primary agent's response is the action.
                # This needs to align with how LLMBasicMAC.select_actions structures its output for env interaction.
                
                # --- Determine action for env.step --- 
                # If env expects a single string answer, we need to extract it.
                # If mac_extra_info contains per-agent responses in 'llm_responses' (List[str]),
                # and a 'commitment_text' (str from coordinator),
                # we likely pass 'commitment_text' or a primary agent's response to env.step.
                # Let's assume env.step() expects the coordinator's commitment text.
                action_for_env_step = mac_extra_info.get("commitment_text", "") 
                if not action_for_env_step and mac_extra_info.get('llm_responses'):
                    # Fallback if no commitment text, use first agent's response (example)
                    action_for_env_step = mac_extra_info['llm_responses'][0] if mac_extra_info['llm_responses'] else ""

                # --- Prepare extra info for comprehensive reward calculation ---
                step_extra_info = {
                    'agent_responses': mac_extra_info.get('llm_responses', []),
                    'commitment_text': mac_extra_info.get('commitment_text', ''),
                    'agent_log_probs': mac_extra_info.get('agent_log_probs'),
                    'prompt_embeddings': mac_extra_info.get('prompt_embeddings'),
                    'belief_states': mac_extra_info.get('belief_states')
                }

                # --- Execute environment step with the LLM's answer and extra info ---
                # env.step() in HuggingFaceDatasetEnv will compare action_for_env_step with ground truth.
                # It returns: next_obs (dummy), reward (total), terminated (True), truncated (False), info_dict
                _next_obs, reward_total_float, terminated, _truncated, env_step_info = self.env.step(
                    action_for_env_step, extra_info=step_extra_info
                )

                # --- Parse rewards from env_step_info ---
                # HuggingFaceDatasetEnv.step() returns info with reward_ts, reward_al, reward_cc.
                reward_ts = env_step_info.get("reward_ts", 0.0)
                reward_al = env_step_info.get("reward_al", 0.0)
                reward_cc = env_step_info.get("reward_cc", 0.0)
                # For Q-learning, the reward in the buffer should be the main reward used for TD updates.
                # Here, it's reward_total_float from env.step().
                # The components (ts, al, cc) are also stored for potential auxiliary losses (like L_dr).

                # 环境返回的是单个奖励值，需要复制给所有智能体
                rewards_al_list = [reward_al] * self.n_agents
                rewards_ts_list = [reward_ts] * self.n_agents
                rewards_cc_list = [reward_cc] * self.n_agents
                
                episode_return += reward_total_float # Episode return is based on the single step
                
                metrics.add_step_data(
                    mac_extra_info, # Pass MAC's extra info for logging things like strategies, raw LLM responses
                    reward_total_float, 
                    reward_al, 
                    reward_ts, 
                    reward_cc
                )
                
                actions_for_batch_storage = discrete_actions[0] if discrete_actions.ndim > 1 else discrete_actions
                
                # Data from MAC's forward pass via _get_actions()
                current_commitment_embedding = mac_extra_info.get('commitment_embedding') 
                current_q_values = mac_extra_info.get('q_values')
                current_agent_prompt_embeddings = mac_extra_info.get('prompt_embeddings') 
                current_group_representation = mac_extra_info.get('group_repr')  # 修复：使用正确的键名
                current_belief_states = mac_extra_info.get('belief_states')

                post_data = self._get_post_transition_data(
                    actions_for_batch_storage, 
                    reward_total_float,  # 传递单个值
                    terminated, # Should be True
                    env_step_info, # Pass info from env.step()
                    rewards_al_list,
                    rewards_ts_list,
                    rewards_cc_list,
                    current_commitment_embedding, 
                    current_q_values, 
                    current_agent_prompt_embeddings,
                    current_group_representation,
                    current_belief_states
                )
                self.batch.update(post_data, ts=self.t) # Store post-transition data at t=0 because episode_limit=1
            
            # Advance timestep (symbolic in this 1-step-per-episode setup)
            self.t += 1 # self.t becomes 1
            # 移除双重计数：train.py会负责更新t_env
            # self.t_env += 1 # 注释掉避免与train.py重复计数
            
            # Episode always terminates after one step (one data sample)
            if not test_mode:
                self._handle_episode_end(metrics, episode_return, test_mode)
                self.t_episodes += 1  # 增加episode计数器
            # No explicit break needed as the loop was symbolic for one pass
            
            # Add final data (next observation) to the batch if episode_limit > 0
            # For episode_limit = 1, max_seq_length is 2. Batch stores at ts=0 and ts=1.
            # We need to store the (dummy) next observation for s_1 at ts=self.t (which is 1)
            if self.episode_limit > 0:
                 self._add_final_data(_next_obs if not test_mode else current_obs) # current_obs as placeholder for test_mode
            
            if not test_mode:
                 self._add_llm_data_to_batch(metrics) # If this method does something useful
            
            return self.batch
            
        except StopIteration: # Raised by self.env.reset() if dataset is exhausted
            self.logger.info(f"Dataset exhausted after {self.t_env} samples.")
            # Potentially return the last partially filled batch or a special signal
            return self.batch # Or None, or raise further to signal completion
        except Exception as e:
            logger.error(f"Error during episode execution: {str(e)}")
            logger.exception("Exception details:")
            raise

    def _get_pre_transition_data(self, current_observation_text: str) -> Dict:
        """Get pre-transition data (current observation)."""
        # Preprocess (tokenize) the observation text using the MAC's preprocessor
        # Ensure obs_tensor is on the correct device (preprocess_observation should handle this)
        # The shape of obs_tensor should be (max_token_length,)
        obs_tensor = self.mac.preprocess_observation(current_observation_text) 

        # Other fields for scheme if needed (often placeholders for text envs)
        default_state_vshape = self.env_info.get("state_shape", (1,))
        default_avail_actions_vshape = (self.env_info.get("n_actions", 1),)

        return {
            # obs_tensor will be grouped by agent and batched by EpisodeBuffer.
            # For bs=1, EpisodeBuffer.update expects data for "obs" to be a list of tensors,
            # one for each agent, or a single tensor if "group" is not "agents".
            # If scheme["obs"] has "group": "agents", then obs_tensor should be provided for each agent.
            # Since the observation is global, we replicate it for each agent.
            "obs": [obs_tensor for _ in range(self.n_agents)],
            "state": [torch.zeros(*default_state_vshape, device=self.args.device)], 
            "avail_actions": [torch.ones(*default_avail_actions_vshape, device=self.args.device, dtype=torch.int64) for _ in range(self.n_agents)]
        }

    def _get_actions(self, test_mode: bool, raw_observation_text: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        """Get actions and extra info from MAC."""
        # self.batch here contains the pre_transition_data at self.t (which is 0)
        # self.mac.select_actions expects the whole batch and current timestep t_ep.
        return self.mac.select_actions(
            self.batch, # Pass the current episode batch (contains tokenized obs at ts=0)
            t_ep=self.t,  # Current step in the episode (0)
            t_env=self.t_env, # Global step counter
            raw_observation_text=raw_observation_text, # Pass raw text for LLM prompts
            test_mode=test_mode
        )

    def _get_post_transition_data(self, discrete_actions_for_agents: torch.Tensor, 
                                reward_total: float,  # 修改为单个值
                                terminated: bool, env_info: Dict,
                                rewards_al: List[float], 
                                rewards_ts: List[float], 
                                rewards_cc: List[float],
                                commitment_embedding: Optional[torch.Tensor],
                                q_values_per_agent: Optional[torch.Tensor], # New: (n_agents, 1)
                                prompt_embeddings_per_agent: Optional[torch.Tensor], # New: (n_agents, 2)
                                group_representation: Optional[torch.Tensor],
                                belief_states: Optional[torch.Tensor]
                                ) -> Dict:
        """Get post-transition data."""
        
        # actions should be a tensor of shape (self.n_agents, expected_action_vshape_in_scheme)
        # scheme[actions][vshape] is (1,) for discrete actions.
        # So, actions should be (self.n_agents, 1)
        # Ensure `actions` (processed_actions from `run`) has this shape.
        # If `actions` from MAC is (n_agents, ), we might need to .view(-1, 1) if scheme expects (1,)

        # 对于全局奖励，创建标量张量
        final_reward_scalar = torch.tensor([reward_total], dtype=torch.float32, device=self.args.device)  # (1,)
        
        # 对于per-agent奖励，创建张量
        rewards_al_tensor = torch.tensor(rewards_al, dtype=torch.float32, device=self.args.device).view(self.n_agents, 1)
        rewards_ts_tensor = torch.tensor(rewards_ts, dtype=torch.float32, device=self.args.device).view(self.n_agents, 1)
        rewards_cc_tensor = torch.tensor(rewards_cc, dtype=torch.float32, device=self.args.device).view(self.n_agents, 1)

        # discrete_actions_for_agents should be a tensor of shape (self.n_agents, scheme_action_vshape)
        # If scheme_action_vshape is (1,), then (self.n_agents, 1)
        # If discrete_actions_for_agents is (self.n_agents, ), then .view(self.n_agents, 1)
        if discrete_actions_for_agents.ndim == 1:
             actions_for_batch = discrete_actions_for_agents.view(self.n_agents, 1)
        else:
             actions_for_batch = discrete_actions_for_agents
        actions_for_batch = actions_for_batch.to(device=self.args.device, dtype=torch.long)

        post_data_dict = {
            "actions": actions_for_batch, 
            "reward": final_reward_scalar, 
            "terminated": torch.tensor([terminated], dtype=torch.uint8, device=self.args.device),
            "reward_al": rewards_al_tensor,
            "reward_ts": rewards_ts_tensor,
            "reward_cc": rewards_cc_tensor,
            "filled": torch.tensor([1], dtype=torch.long, device=self.args.device)
        }

        if commitment_embedding is not None:
            if commitment_embedding.ndim == 1: 
                processed_commitment_embedding = commitment_embedding.unsqueeze(0).to(self.args.device)
            elif commitment_embedding.ndim == 2 and commitment_embedding.shape[0] == 1: # Already (1, embed_dim)
                processed_commitment_embedding = commitment_embedding.to(self.args.device)
            else:
                self.logger.warning(f"Unexpected commitment_embedding shape from MAC: {commitment_embedding.shape}. Expected (embed_dim,) or (1, embed_dim). Not adding to batch.")
                processed_commitment_embedding = None 
            
            if processed_commitment_embedding is not None:
                 post_data_dict["commitment_embedding"] = processed_commitment_embedding
        
        if q_values_per_agent is not None: # Expected shape (n_agents, 1) or (1, n_agents, 1)
            if q_values_per_agent.shape == (self.n_agents, 1):
                post_data_dict["q_values"] = q_values_per_agent.to(self.args.device) # Shape: (n_agents, 1)
            elif q_values_per_agent.shape == (1, self.n_agents, 1):
                post_data_dict["q_values"] = q_values_per_agent.squeeze(0).to(self.args.device) # Remove batch dim: (n_agents, 1)
            else:
                self.logger.warning(f"Unexpected q_values_per_agent shape: {q_values_per_agent.shape}. Expected ({self.n_agents}, 1) or (1, {self.n_agents}, 1). Not adding to batch.")

        if prompt_embeddings_per_agent is not None: # Expected shape (n_agents, 2) or (1, n_agents, 2)
            if prompt_embeddings_per_agent.shape == (self.n_agents, 2):
                post_data_dict["prompt_embeddings"] = prompt_embeddings_per_agent.to(self.args.device) # Shape: (n_agents, 2)
            elif prompt_embeddings_per_agent.shape == (1, self.n_agents, 2):
                post_data_dict["prompt_embeddings"] = prompt_embeddings_per_agent.squeeze(0).to(self.args.device) # Remove batch dim: (n_agents, 2)
            else:
                self.logger.warning(f"Unexpected prompt_embeddings_per_agent shape: {prompt_embeddings_per_agent.shape}. Expected ({self.n_agents}, 2) or (1, {self.n_agents}, 2). Not adding to batch.")

        if group_representation is not None:
            if group_representation.ndim == 1: 
                processed_group_representation = group_representation.to(self.args.device) # Shape: (embed_dim,)
            elif group_representation.ndim == 2 and group_representation.shape[0] == 1: # Shape: (1, embed_dim)
                processed_group_representation = group_representation.squeeze(0).to(self.args.device) # Remove batch dim: (embed_dim,)
            else:
                self.logger.warning(f"Unexpected group_representation shape from MAC: {group_representation.shape}. Expected (embed_dim,) or (1, embed_dim). Not adding to batch.")
                processed_group_representation = None 
            
            if processed_group_representation is not None:
                 post_data_dict["group_representation"] = processed_group_representation
        
        if belief_states is not None:
            expected_belief_dim = getattr(self.args, 'belief_dim', 64) # Get belief_dim from args, with a fallback
            if belief_states.shape == (self.n_agents, expected_belief_dim):
                post_data_dict["belief_states"] = belief_states.to(self.args.device) # Shape: (n_agents, belief_dim)
            elif belief_states.shape == (1, self.n_agents, expected_belief_dim):
                post_data_dict["belief_states"] = belief_states.squeeze(0).to(self.args.device) # Remove batch dim: (n_agents, belief_dim)
            else:
                self.logger.warning(f"Unexpected belief_states shape: {belief_states.shape}. Expected ({self.n_agents}, {expected_belief_dim}) or (1, {self.n_agents}, {expected_belief_dim}). Not adding to batch.")

        return post_data_dict

    def _handle_episode_end(self, metrics: EpisodeMetrics, 
                          episode_return: float, test_mode: bool):
        """Handle end of episode processing."""
        self._save_episode_metrics(metrics, test_mode)
        
        if test_mode:
            self.test_returns.append(episode_return)
            self.logger.log_stat("test_return", episode_return, self.t_env)
        else:
            self.train_returns.append(episode_return)
            self.logger.log_stat("train_return", episode_return, self.t_env)

    def _add_final_data(self, next_observation_text: str):
        """Add final (next) observations to batch at self.t (which is 1)."""
        # Preprocess the next (dummy) observation text
        next_obs_tensor = self.mac.preprocess_observation(next_observation_text)

        default_state_vshape = self.env_info.get("state_shape", (1,))
        default_avail_actions_vshape = (self.env_info.get("n_actions", 1),)

        last_data = {
            "obs": [next_obs_tensor for _ in range(self.n_agents)],
            "state": [torch.zeros(*default_state_vshape, device=self.args.device)], 
            "avail_actions": [torch.ones(*default_avail_actions_vshape, device=self.args.device, dtype=torch.int64) for _ in range(self.n_agents)],
            "filled": torch.tensor([0], dtype=torch.long, device=self.args.device)  # 最终状态标记为无效
        }
        self.batch.update(last_data, ts=self.t) # self.t is 1 here

    def reset_runner_state(self):
        """Reset runner's per-episode state (batch and timestep t)."""
        self.batch = self.new_batch() # Get a fresh batch from the handler
        self.t = 0 # Reset episode timestep

    def _build_scheme(self) -> Dict:
        """
        Build data scheme for episode batch.
        
        Returns:
            Data scheme dictionary
        """
        commitment_dim = getattr(self.args, 'commitment_embedding_dim', 768)
        belief_dim = getattr(self.args, 'belief_dim')
        # Max question length here refers to max token length after tokenization
        # It should come from env_args, which HuggingFaceDatasetEnv also uses.
        max_token_len = getattr(self.args.env_args, "max_question_length", 512)

        scheme = {
            "state": {"vshape": self.env_info["state_shape"]}, # Usually (1,) for these envs
            # obs is now token IDs, per agent
            "obs": {"vshape": (max_token_len,), "group": "agents", "dtype": torch.long},
            "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long}, # Symbolic actions
            "avail_actions": {
                "vshape": (self.env_info["n_actions"],), # n_actions usually 1
                "group": "agents",
                "dtype": torch.int64, # Changed from torch.int for consistency
            },
            "reward": {"vshape": (1,)}, # Global reward, will be unsqueezed by buffer if group="agents"
            "terminated": {"vshape": (1,), "dtype": torch.uint8},
            "filled": {"vshape": (1,), "dtype": torch.long},  # 添加filled字段，标记有效的时间步
            
            # Fields per agent (these are fine)
            "q_values": {"vshape": (1,), "group": "agents", "dtype": torch.float32}, 
            "prompt_embeddings": {"vshape": (2,), "group": "agents", "dtype": torch.float32}, 
            "belief_states": {"vshape": (belief_dim,), "group": "agents"},
            "reward_al": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
            "reward_ts": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
            "reward_cc": {"vshape": (1,), "group": "agents", "dtype": torch.float32},

            # Global fields (these are fine)
            "commitment_embedding": {"vshape": (commitment_dim,), "dtype": torch.float32},
            "group_representation": {"vshape": (belief_dim,), "dtype": torch.float32}
        }
        return scheme

    def _build_groups(self) -> Dict:
        """
        Build groups for episode batch.
        
        Returns:
            Group definitions
        """
        return {
            "agents": self.args.n_agents
        }

    def _save_episode_metrics(self, metrics: EpisodeMetrics, test_mode: bool):
        """
        Save episode metrics.
        
        Args:
            metrics: Collected metrics
            test_mode: Whether in testing mode
        """
        stats = self.test_stats if test_mode else self.train_stats
        
        # Calculate average reward
        if metrics.rewards:
            stats['mean_reward'] = np.mean(metrics.rewards)
        
        # Calculate LLM response diversity
        if metrics.llm_responses:
            unique_responses = len(set(map(str, metrics.llm_responses)))
            stats['response_diversity'] = unique_responses / len(metrics.llm_responses)
        
        # Log statistics
        prefix = 'test_' if test_mode else 'train_'
        for k, v in stats.items():
            self.logger.log_stat(f"{prefix}{k}", v, self.t_env)

    def _add_llm_data_to_batch(self, metrics: EpisodeMetrics):
        """
        Add LLM-related data to episode batch.
        
        Args:
            metrics: Collected LLM metrics
        """
        self.logger.info("_add_llm_data_to_batch called. Current logic mostly commented out or for text logging only.")
        # try:
            # Only stack if there are items and they are stackable (e.g. tensors)
            # llm_data_to_add = {}
            # if metrics.llm_responses and all(isinstance(x, torch.Tensor) for x in metrics.llm_responses):
            #     llm_data_to_add["llm_responses"] = torch.stack(metrics.llm_responses) 
            
            # if metrics.belief_states: # Already added per step
            #    pass 

            # if llm_data_to_add: 
            #    self.batch.update(llm_data_to_add) 
            #    self.logger.info("_add_llm_data_to_batch: Added to batch - " + str(list(llm_data_to_add.keys())))

        # except Exception as e:
        #     logger.error(f"Error in _add_llm_data_to_batch: {str(e)}. Data types might be incompatible.")
        #     raise

    def reset(self):
        """Reset the runner state."""
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def get_env_info(self) -> Dict:
        """Get environment information."""
        return self.env_info

    def save_replay(self):
        """Save replay buffer."""
        self.env.save_replay()

    def close_env(self):
        """Close environment."""
        self.env.close()

    def log_train_stats_t(self):
        """Log training statistics."""
        self.logger.print_recent_stats()