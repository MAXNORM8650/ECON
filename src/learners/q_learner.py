import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from components.episode_buffer import EpisodeBatch
from modules.mixer.mix_llm import LLMQMixer
from modules.belief_encoder import BeliefEncoder
from typing import Dict, List, Tuple, Optional, Any
import os

"""
Q-Learning algorithm with multi-agent coordination.

Learner for the ECON framework.

Implements Q-learning with:
- Multi-agent coordination
- LLM-based belief networks
- Mixing networks for global Q-values
- Dynamic reward systems
- Two-stage belief coordination
- Bayesian Nash Equilibrium (BNE) updates
"""

class ECONLearner:
    """
    Learner for the ECON framework.
    Handles the optimization of individual BeliefNetworks, the BeliefEncoder,
    and the CentralizedMixingNetwork (LLMQMixer).
    """
    
    def __init__(self, mac: Any, scheme: Dict, logger: Any, args: Any):
        self.args = args
        self.logger = logger
        self.mac = mac
        # Correctly access use_cuda attribute
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        
        self.last_target_update_episode = 0
        self.log_stats_t = -getattr(args, "learner_log_interval", 100) - 1

        # Initialize ECON Network Components
        self.mixer: Optional[LLMQMixer] = None
        self.target_mixer: Optional[LLMQMixer] = None
        self.belief_encoder: Optional[BeliefEncoder] = None
        self.target_belief_encoder: Optional[BeliefEncoder] = None
        self.target_mac = None

        # Parameter Groups
        self.belief_net_params: List = []
        self.encoder_params: List = []
        self.mixer_params: List = []

        # Optimizers
        self.belief_optimizer: Optional[Adam] = None
        self.encoder_optimizer: Optional[Adam] = None
        self.mixer_optimizer: Optional[Adam] = None

        # Loss Weights
        self.gamma = getattr(args, "gamma", 0.99)
        self.lambda_e = getattr(args, "lambda_e", 0.1)
        self.lambda_sd = getattr(args, "lambda_sd", 0.1)
        self.lambda_m = getattr(args, "lambda_m", 0.1)
        self.lambda_belief = getattr(args.loss, "belief_weight", 0.1) if hasattr(args, 'loss') else 0.1
        
        # BNE协调参数
        self.bne_max_iterations = getattr(args, "bne_max_iterations", 5)
        self.bne_convergence_threshold = getattr(args, "bne_convergence_threshold", 0.01)
        self.stage2_weight = getattr(args, "stage2_weight", 0.3)  # Stage 2在总损失中的权重
        
        # Initialize networks and optimizers
        self._initialize_networks_and_optimizers(args)

    def _initialize_networks_and_optimizers(self, args: Any):
        # Initialize Mixer (CentralizedMixingNetwork)
        if getattr(args, "use_mixer", True):
            self.mixer = LLMQMixer(args)
            self.target_mixer = LLMQMixer(args)
            self.mixer_params = list(self.mixer.parameters())
            self.logger.info(f"Mixer initialized with {len(self.mixer_params)} parameters.")
        else:
            self.mixer = None
            self.target_mixer = None
            self.mixer_params = []
            self.logger.info("Mixer is disabled.")

        # Initialize Belief Encoder
        if hasattr(self.mac, 'belief_encoder') and self.mac.belief_encoder is not None:
            self.belief_encoder = self.mac.belief_encoder
            self.logger.info("Using BeliefEncoder from MAC.")
        elif getattr(args, "use_belief_encoder", True):
            self.belief_encoder = BeliefEncoder(args)
            self.logger.info("Initialized standalone BeliefEncoder.")
        else:
            self.belief_encoder = None
            self.logger.info("BeliefEncoder is disabled.")
        
        if self.belief_encoder is not None:
            self.encoder_params = list(self.belief_encoder.parameters())
            self.target_belief_encoder = copy.deepcopy(self.belief_encoder)
            self.logger.info(f"BeliefEncoder has {len(self.encoder_params)} parameters.")
        else:
            self.encoder_params = []
            self.target_belief_encoder = None
            
        # Initialize Target MAC
        self.target_mac = copy.deepcopy(self.mac)

        # Collect parameters for Individual Belief Networks
        self.belief_net_params = []
        if hasattr(self.mac, 'agents') and (isinstance(self.mac.agents, list) or isinstance(self.mac.agents, nn.ModuleList)):
            for agent_module in self.mac.agents:
                if hasattr(agent_module, 'belief_network') and agent_module.belief_network is not None:
                    self.belief_net_params.extend(list(agent_module.belief_network.parameters()))
                else:
                    self.logger.warning("An agent module in mac.agents is missing 'belief_network' or it's None.")
        elif hasattr(self.mac, 'agent') and hasattr(self.mac.agent, 'belief_network') and self.mac.agent.belief_network is not None: 
            self.logger.info("Treating mac.agent as the single BeliefNetwork provider.")
            self.belief_net_params.extend(list(self.mac.agent.belief_network.parameters()))
        else:
            self.logger.error("ECONLearner: Could not find belief_network parameters in MAC structure. BeliefNetwork losses might not work.")

        # Initialize Optimizers
        self.belief_optimizer = None
        if self.belief_net_params:
            self.belief_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.belief_net_params),
                lr=getattr(args, "belief_net_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )
        
        self.encoder_optimizer = None
        if self.encoder_params and self.belief_encoder:
            self.encoder_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.encoder_params),
                lr=getattr(args, "encoder_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )
        
        self.mixer_optimizer = None
        if self.mixer_params and self.mixer:
            self.mixer_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.mixer_params),
                lr=getattr(args, "mixer_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )

        if self.mixer is None:
            self.logger.warning("ECONLearner: Mixer is None. Global Q-value calculation and related losses will be skipped during training.")
        if self.belief_encoder is None:
            self.logger.warning("ECONLearner: BeliefEncoder is None. Group representation E and related losses will be skipped.")
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> Dict:
        """
        Train the ECON framework using the provided batch data with two-stage coordination.
        
        Args:
            batch: Episode batch data
            t_env: Current environment timestep
            episode_num: Current episode number
            
        Returns:
            Dictionary containing training statistics
        """
        rewards = batch["reward"][:, :-1].to(self.device)
        terminated = batch["terminated"][:, :-1].float().to(self.device)
        mask = batch["filled"][:, :-1].float().to(self.device)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        if self.mixer is None:
            self.logger.warning("Mixer is None, training will be skipped.")
            return {"status": "skipped_mixer_none"}

        if hasattr(self.mac, 'init_hidden'):
            self.mac.init_hidden(batch.batch_size)
        if hasattr(self.target_mac, 'init_hidden'):
            self.target_mac.init_hidden(batch.batch_size)

        # ===========================================
        # Stage 1: Individual Belief Formation
        # ===========================================
        
        # Collect data from forward passes - Stage 1
        list_belief_states_stage1, list_prompt_embeddings_stage1, list_local_q_values_stage1, list_group_repr_stage1 = [], [], [], []
        list_belief_states_stage1_next, list_prompt_embeddings_stage1_next, list_local_q_values_stage1_next, list_group_repr_stage1_next = [], [], [], []
        
        # Store commitment features if available in batch
        list_commitment_features_t = [] 
        has_commitment_features_in_batch = "commitment_embedding" in batch.scheme
        
        self.logger.debug(f"Commitment embedding in batch scheme: {has_commitment_features_in_batch}")
        if has_commitment_features_in_batch:
            self.logger.debug(f"Commitment embedding scheme: {batch.scheme['commitment_embedding']}")

        # Stage 1: Forward pass through time steps for individual belief formation
        self.logger.debug("Starting Stage 1: Individual belief formation")
        for t in range(batch.max_seq_length - 1):
            _, mac_info_t = self.mac.forward(batch, t, train_mode=True)
            list_belief_states_stage1.append(mac_info_t["belief_states"])
            list_prompt_embeddings_stage1.append(mac_info_t["prompt_embeddings"])
            list_local_q_values_stage1.append(mac_info_t["q_values"])
            list_group_repr_stage1.append(mac_info_t["group_repr"])

            # 修复：正确处理commitment_embedding
            if has_commitment_features_in_batch:
                if t < batch.max_seq_length - 1:  # 确保时间步有效
                    try:
                        commitment_emb_t = batch["commitment_embedding"][:, t]
                        list_commitment_features_t.append(commitment_emb_t)
                        self.logger.debug(f"Added commitment_embedding at t={t}, shape: {commitment_emb_t.shape}")
                    except (KeyError, IndexError) as e:
                        self.logger.warning(f"Failed to get commitment_embedding at t={t}: {e}")
                        # 创建dummy commitment embedding
                        dummy_emb = torch.zeros(batch.batch_size, self.args.commitment_embedding_dim, device=self.device)
                        list_commitment_features_t.append(dummy_emb)
                        self.logger.debug(f"Created dummy commitment_embedding at t={t}")

            _, target_mac_info_t_next = self.target_mac.forward(batch, t + 1, train_mode=True)
            list_belief_states_stage1_next.append(target_mac_info_t_next["belief_states"])
            list_prompt_embeddings_stage1_next.append(target_mac_info_t_next["prompt_embeddings"])
            list_local_q_values_stage1_next.append(target_mac_info_t_next["q_values"])
            list_group_repr_stage1_next.append(target_mac_info_t_next["group_repr"])

        # Stack temporal data for Stage 1
        belief_states_stage1_stacked = torch.stack(list_belief_states_stage1, dim=1)
        prompt_embeddings_stage1_stacked = torch.stack(list_prompt_embeddings_stage1, dim=1)
        local_q_values_stage1_stacked = torch.stack(list_local_q_values_stage1, dim=1)
        group_representation_stage1_stacked = torch.stack(list_group_repr_stage1, dim=1)

        belief_states_stage1_next_stacked = torch.stack(list_belief_states_stage1_next, dim=1)
        local_q_values_stage1_next_stacked = torch.stack(list_local_q_values_stage1_next, dim=1)

        # ===========================================
        # Stage 2: BNE Coordination
        # ===========================================
        
        self.logger.debug("Starting Stage 2: BNE coordination")
        belief_states_stage2, prompt_embeddings_stage2, local_q_values_stage2, group_representation_stage2 = self._perform_bne_coordination(
            belief_states_stage1_stacked,
            prompt_embeddings_stage1_stacked,
            local_q_values_stage1_stacked,
            group_representation_stage1_stacked,
            batch
        )

        # 修复：确保commitment_features正确处理
        commitment_features_t_stacked = None
        if has_commitment_features_in_batch and list_commitment_features_t:
            try:
                commitment_features_t_stacked = torch.stack(list_commitment_features_t, dim=1)
                self.logger.debug(f"Stacked commitment_features shape: {commitment_features_t_stacked.shape}")
            except Exception as e:
                self.logger.warning(f"Failed to stack commitment_features: {e}")
                # 创建dummy commitment features
                commitment_features_t_stacked = torch.zeros(
                    batch.batch_size, batch.max_seq_length - 1, self.args.commitment_embedding_dim, 
                    device=self.device
                )
                self.logger.debug(f"Created dummy commitment_features_t_stacked shape: {commitment_features_t_stacked.shape}")
        elif has_commitment_features_in_batch:
            # 如果scheme中有commitment_embedding但list为空，创建dummy
            commitment_features_t_stacked = torch.zeros(
                batch.batch_size, batch.max_seq_length - 1, self.args.commitment_embedding_dim, 
                device=self.device
            )
            self.logger.debug(f"Created dummy commitment_features (empty list) shape: {commitment_features_t_stacked.shape}")

        bs_x_seq_len = batch.batch_size * (batch.max_seq_length - 1)

        # ===========================================
        # Loss Calculation
        # ===========================================

        # 使用Stage 2的结果进行mixer计算
        prompt_embeddings_stage2_flat = prompt_embeddings_stage2.reshape(bs_x_seq_len, self.n_agents, -1)
        local_q_values_stage2_flat = local_q_values_stage2.reshape(bs_x_seq_len, self.n_agents)
        group_representation_stage2_flat = group_representation_stage2.reshape(bs_x_seq_len, -1)

        # Target values using Stage 1 (more stable)
        local_q_values_stage1_next_flat = local_q_values_stage1_next_stacked.reshape(bs_x_seq_len, self.n_agents)

        # 修复：正确处理commitment_features_flat
        commitment_features_flat = None
        if commitment_features_t_stacked is not None:
            commitment_features_flat = commitment_features_t_stacked.reshape(bs_x_seq_len, -1)
            self.logger.debug(f"Flattened commitment_features shape: {commitment_features_flat.shape}")

        # Forward pass through mixer using Stage 2 coordinated values
        mixer_results_stage2 = self.mixer(
            local_q_values=local_q_values_stage2_flat,
            prompt_embeddings=prompt_embeddings_stage2_flat,
            group_representation=group_representation_stage2_flat
        )
        q_total_stage2_flat = mixer_results_stage2["Q_tot"] 

        # Target mixer forward pass using Stage 1 next values
        # 需要计算target的group representation
        target_group_repr_next = self.target_belief_encoder(belief_states_stage1_next_stacked.reshape(bs_x_seq_len, self.n_agents, -1)).reshape(bs_x_seq_len, -1)
        target_prompt_embeddings_next_flat = torch.stack(list_prompt_embeddings_stage1_next, dim=1).reshape(bs_x_seq_len, self.n_agents, -1)
        
        target_mixer_results_next = self.target_mixer(
            local_q_values=local_q_values_stage1_next_flat,
            prompt_embeddings=target_prompt_embeddings_next_flat,
            group_representation=target_group_repr_next
        )
        q_total_target_next_flat = target_mixer_results_next["Q_tot"].detach()

        # Prepare reward and termination data
        rewards_flat = rewards.reshape(bs_x_seq_len, 1)
        terminated_flat = terminated.reshape(bs_x_seq_len, 1)
        mask_flat = mask.reshape(bs_x_seq_len, 1)

        # Calculate target Q-values
        target_q_total_flat = rewards_flat + self.gamma * (1 - terminated_flat) * q_total_target_next_flat

        # ===========================================
        # BeliefNetwork Loss Calculation
        # ===========================================
        
        belief_loss = self._calculate_belief_network_loss(
            belief_states_stage1_stacked,
            belief_states_stage2,
            local_q_values_stage1_stacked,
            local_q_values_stage2,
            target_q_total_flat.reshape(batch.batch_size, batch.max_seq_length - 1),
            rewards.squeeze(-1),
            mask.squeeze(-1)
        )

        # ===========================================
        # Mixer Loss Calculation
        # ===========================================
        
        F_i_for_LSD = mixer_results_stage2.get("F_i_for_LSD")
        
        # 调试信息
        self.logger.debug(f"F_i_for_LSD is None: {F_i_for_LSD is None}")
        self.logger.debug(f"commitment_features_flat is None: {commitment_features_flat is None}")
        self.logger.debug(f"lambda_sd: {self.lambda_sd}")
        
        total_mix_loss, loss_components = self.mixer.calculate_mix_loss(
            Q_tot=q_total_stage2_flat,
            local_q_values=local_q_values_stage2_flat,
            F_i_for_LSD=F_i_for_LSD,
            commitment_text_features=commitment_features_flat,
            target_Q_tot=target_q_total_flat,
            rewards_total=rewards_flat,
            gamma=self.gamma,
            lambda_sd=self.lambda_sd,
            lambda_m=self.lambda_m,
            terminated=terminated_flat,
            mask_flat=mask_flat
        )

        # ===========================================
        # BeliefEncoder Loss
        # ===========================================
        
        encoder_loss = self._calculate_encoder_loss(
            belief_states_stage1_stacked,
            belief_states_stage2,
            group_representation_stage1_stacked,
            group_representation_stage2
        )

        # ===========================================
        # Network Optimization
        # ===========================================

        # 1. Optimize BeliefNetworks
        if self.belief_optimizer:
            self.belief_optimizer.zero_grad()
            belief_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.belief_net_params, 10.0)
            self.belief_optimizer.step()

        # 2. Optimize BeliefEncoder
        if self.encoder_optimizer:
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.encoder_params, 10.0)
            self.encoder_optimizer.step()

        # 3. Optimize Mixer
        if self.mixer_optimizer:
            self.mixer_optimizer.zero_grad()
            total_mix_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mixer_params, 10.0)
            self.mixer_optimizer.step()

        # Update target networks periodically
        if episode_num - self.last_target_update_episode >= getattr(self.args, "target_update_interval", 200):
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Prepare training statistics
        train_stats = {
            "loss_total": (belief_loss + encoder_loss + total_mix_loss).item(),
            "loss_belief": belief_loss.item(),
            "loss_encoder": encoder_loss.item(),
            "loss_mixer": total_mix_loss.item(),
            "q_total_stage1_mean": torch.stack(list_local_q_values_stage1).mean().item(),
            "q_total_stage2_mean": local_q_values_stage2.mean().item(),
            "reward_mean": rewards_flat.mean().item(),
        }
        
        # Add individual loss components
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                train_stats[f"mixer_{key}"] = value.item()
            else:
                train_stats[f"mixer_{key}"] = value

        return train_stats

    def _perform_bne_coordination(self, belief_states_stage1: torch.Tensor, 
                                 prompt_embeddings_stage1: torch.Tensor,
                                 local_q_values_stage1: torch.Tensor,
                                 group_representation_stage1: torch.Tensor,
                                 batch: EpisodeBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行贝叶斯纳什均衡协调，实现Stage 2的belief更新
        
        Args:
            belief_states_stage1: Stage 1的belief states
            prompt_embeddings_stage1: Stage 1的prompt embeddings
            local_q_values_stage1: Stage 1的local Q values
            group_representation_stage1: Stage 1的group representation
            batch: Episode batch data
            
        Returns:
            Tuple of (belief_states_stage2, prompt_embeddings_stage2, local_q_values_stage2, group_representation_stage2)
        """
        batch_size, seq_len, n_agents, belief_dim = belief_states_stage1.shape
        
        # 初始化Stage 2的状态为Stage 1的状态
        belief_states_current = belief_states_stage1.clone()
        prompt_embeddings_current = prompt_embeddings_stage1.clone()
        local_q_values_current = local_q_values_stage1.clone()
        group_representation_current = group_representation_stage1.clone()
        
        # BNE迭代更新
        for iteration in range(self.bne_max_iterations):
            belief_states_prev = belief_states_current.clone()
            
            # 为每个时间步计算BNE更新
            for t in range(seq_len):
                # 当前时间步的状态
                current_beliefs_t = belief_states_current[:, t]  # (batch, n_agents, belief_dim)
                current_group_repr_t = group_representation_current[:, t]  # (batch, group_dim)
                
                # 计算agent间的互动影响
                agent_interactions = self._calculate_agent_interactions(
                    current_beliefs_t, current_group_repr_t
                )
                
                # 更新belief states (BNE step)
                updated_beliefs_t = self._update_beliefs_bne(
                    current_beliefs_t, agent_interactions, batch, t
                )
                
                # 重新计算prompt embeddings和Q values
                updated_prompt_emb_t, updated_q_vals_t = self._recompute_agent_outputs(
                    updated_beliefs_t, batch, t
                )
                
                # 更新group representation
                updated_group_repr_t = self.belief_encoder(updated_beliefs_t)
                
                # 存储更新的状态
                belief_states_current[:, t] = updated_beliefs_t
                prompt_embeddings_current[:, t] = updated_prompt_emb_t
                local_q_values_current[:, t] = updated_q_vals_t
                group_representation_current[:, t] = updated_group_repr_t
            
            # 检查收敛性
            belief_change = torch.norm(belief_states_current - belief_states_prev).item()
            if belief_change < self.bne_convergence_threshold:
                self.logger.debug(f"BNE converged after {iteration + 1} iterations, change: {belief_change:.6f}")
                break
        
        return belief_states_current, prompt_embeddings_current, local_q_values_current, group_representation_current

    def _calculate_agent_interactions(self, beliefs: torch.Tensor, group_repr: torch.Tensor) -> torch.Tensor:
        """
        计算智能体之间的互动影响矩阵
        
        Args:
            beliefs: (batch, n_agents, belief_dim)
            group_repr: (batch, group_dim)
            
        Returns:
            interaction matrix: (batch, n_agents, n_agents)
        """
        batch_size, n_agents, belief_dim = beliefs.shape
        
        # 计算agent间的相似性矩阵
        beliefs_normalized = F.normalize(beliefs, p=2, dim=-1)
        similarity_matrix = torch.bmm(beliefs_normalized, beliefs_normalized.transpose(-2, -1))
        
        # 加入group representation的影响
        group_influence = group_repr.unsqueeze(1).expand(-1, n_agents, -1)  # (batch, n_agents, group_dim)
        
        # 简化的互动权重计算
        interaction_weights = torch.softmax(similarity_matrix, dim=-1)
        
        return interaction_weights

    def _update_beliefs_bne(self, beliefs: torch.Tensor, interactions: torch.Tensor, 
                           batch: EpisodeBatch, t: int) -> torch.Tensor:
        """
        使用BNE机制更新belief states
        
        Args:
            beliefs: (batch, n_agents, belief_dim)
            interactions: (batch, n_agents, n_agents)
            batch: Episode batch
            t: Time step
            
        Returns:
            updated beliefs: (batch, n_agents, belief_dim)
        """
        batch_size, n_agents, belief_dim = beliefs.shape
        
        # BNE更新：每个agent考虑其他agent的影响
        updated_beliefs = beliefs.clone()
        
        for i in range(n_agents):
            # 当前agent的belief
            current_belief_i = beliefs[:, i]  # (batch, belief_dim)
            
            # 其他agents对agent i的影响
            other_agents_influence = torch.zeros_like(current_belief_i)
            for j in range(n_agents):
                if i != j:
                    interaction_weight = interactions[:, i, j].unsqueeze(-1)  # (batch, 1)
                    other_agents_influence += interaction_weight * beliefs[:, j]
            
            # BNE更新规则：当前belief + 其他agents的加权影响
            bne_update_rate = 0.1  # 学习率，可以作为超参数
            updated_beliefs[:, i] = current_belief_i + bne_update_rate * other_agents_influence
        
        return updated_beliefs

    def _recompute_agent_outputs(self, updated_beliefs: torch.Tensor, 
                               batch: EpisodeBatch, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于更新的belief states重新计算agent outputs
        
        Args:
            updated_beliefs: (batch, n_agents, belief_dim)
            batch: Episode batch
            t: Time step
            
        Returns:
            Tuple of (prompt_embeddings, q_values)
        """
        batch_size, n_agents, belief_dim = updated_beliefs.shape
        
        # 获取原始观察
        obs_tokens = batch["obs"][:, t]  # (batch_size, n_agents, max_token_len)
        inputs = obs_tokens.reshape(batch_size * n_agents, -1)
        
        # 通过belief network重新计算输出
        if hasattr(self.mac.agent, 'belief_network'):
            # 创建一个虚拟的mask
            mask = torch.zeros(inputs.shape, dtype=torch.bool, device=self.device)
            
            # 重新前向传播
            belief_outputs = self.mac.agent.belief_network(inputs, mask)
            
            # 重塑输出
            prompt_embeddings = belief_outputs['prompt_embedding'].view(batch_size, n_agents, -1)
            q_values = belief_outputs['q_value'].view(batch_size, n_agents, -1).squeeze(-1)
            
            return prompt_embeddings, q_values
        else:
            # 备用：如果无法重新计算，返回基于belief的近似值
            prompt_embeddings = torch.randn(batch_size, n_agents, 2, device=self.device)
            q_values = torch.mean(updated_beliefs, dim=-1)  # 简化的Q值计算
            
            return prompt_embeddings, q_values

    def _calculate_belief_network_loss(self, belief_states_stage1: torch.Tensor,
                                     belief_states_stage2: torch.Tensor,
                                     q_values_stage1: torch.Tensor,
                                     q_values_stage2: torch.Tensor,
                                     target_q_total: torch.Tensor,
                                     rewards: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
        """
        计算BeliefNetwork的损失
        包括：
        1. Stage 1的TD损失
        2. Stage 2的BNE一致性损失
        3. Belief状态的正则化
        
        Args:
            belief_states_stage1/stage2: (batch, seq_len, n_agents, belief_dim)
            q_values_stage1/stage2: (batch, seq_len, n_agents)
            target_q_total: (batch, seq_len)
            rewards: (batch, seq_len)
            mask: (batch, seq_len)
            
        Returns:
            total belief loss
        """
        batch_size, seq_len, n_agents = q_values_stage1.shape
        
        # 1. Stage 1 TD Loss (个体学习)
        target_q_expanded = target_q_total.unsqueeze(-1).expand(-1, -1, n_agents)
        td_error_stage1 = (q_values_stage1 - target_q_expanded.detach()) * mask.unsqueeze(-1)
        loss_td_stage1 = (td_error_stage1 ** 2).sum() / mask.sum().clamp(min=1e-6)
        
        # 2. Stage 2 BNE Consistency Loss (协调一致性)
        # 衡量Stage 2中agents之间的Q值一致性
        q_mean_stage2 = q_values_stage2.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        consistency_error = (q_values_stage2 - q_mean_stage2) * mask.unsqueeze(-1)
        loss_bne_consistency = (consistency_error ** 2).sum() / mask.sum().clamp(min=1e-6)
        
        # 3. Belief Evolution Loss (衡量Stage 1到Stage 2的合理演化)
        belief_evolution = belief_states_stage2 - belief_states_stage1
        evolution_norm = torch.norm(belief_evolution, p=2, dim=-1)  # (batch, seq_len, n_agents)
        # 希望演化不要太剧烈，但也不要完全不变
        target_evolution_norm = 0.1  # 期望的演化幅度
        evolution_loss = ((evolution_norm - target_evolution_norm) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1e-6)
        
        # 4. Belief Regularization (防止belief过度复杂)
        belief_reg_stage1 = torch.norm(belief_states_stage1, p=2, dim=-1).mean()
        belief_reg_stage2 = torch.norm(belief_states_stage2, p=2, dim=-1).mean()
        
        # 总的BeliefNetwork损失
        total_belief_loss = (
            loss_td_stage1 + 
            self.stage2_weight * loss_bne_consistency + 
            0.1 * evolution_loss + 
            0.01 * (belief_reg_stage1 + belief_reg_stage2)
        )
        
        return total_belief_loss

    def _calculate_encoder_loss(self, belief_states_stage1: torch.Tensor,
                              belief_states_stage2: torch.Tensor,
                              group_repr_stage1: torch.Tensor,
                              group_repr_stage2: torch.Tensor) -> torch.Tensor:
        """
        计算BeliefEncoder的损失
        
        Args:
            belief_states_stage1/stage2: (batch, seq_len, n_agents, belief_dim)
            group_repr_stage1/stage2: (batch, seq_len, group_dim)
            
        Returns:
            encoder loss
        """
        # 1. Representation Consistency Loss
        # 确保group representation能够很好地总结individual beliefs
        batch_size, seq_len, n_agents, belief_dim = belief_states_stage1.shape
        
        # 重新计算group representation以确保一致性
        beliefs_stage1_flat = belief_states_stage1.reshape(-1, n_agents, belief_dim)
        beliefs_stage2_flat = belief_states_stage2.reshape(-1, n_agents, belief_dim)
        
        recomputed_group_repr_stage1 = self.belief_encoder(beliefs_stage1_flat).reshape(batch_size, seq_len, -1)
        recomputed_group_repr_stage2 = self.belief_encoder(beliefs_stage2_flat).reshape(batch_size, seq_len, -1)
        
        # 一致性损失
        consistency_loss_stage1 = F.mse_loss(recomputed_group_repr_stage1, group_repr_stage1)
        consistency_loss_stage2 = F.mse_loss(recomputed_group_repr_stage2, group_repr_stage2)
        
        # 2. Evolution Smoothness Loss
        # 确保group representation的演化是平滑的
        evolution_loss = F.mse_loss(group_repr_stage2, group_repr_stage1)
        
        # 3. Representation Diversity Loss
        # 确保不同的belief组合产生不同的group representation
        group_repr_stage2_norm = F.normalize(group_repr_stage2.reshape(-1, group_repr_stage2.shape[-1]), p=2, dim=-1)
        diversity_matrix = torch.mm(group_repr_stage2_norm, group_repr_stage2_norm.t())
        diversity_loss = torch.mean(torch.abs(diversity_matrix - torch.eye(diversity_matrix.shape[0], device=self.device)))
        
        total_encoder_loss = (
            consistency_loss_stage1 + consistency_loss_stage2 + 
            0.1 * evolution_loss + 
            0.01 * diversity_loss
        )
        
        return total_encoder_loss

    def _update_targets(self):
        """Update target networks with current network parameters."""
        if self.target_mixer and self.mixer:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.target_belief_encoder and self.belief_encoder:
            self.target_belief_encoder.load_state_dict(self.belief_encoder.state_dict())
        if self.target_mac and self.mac:
            self.target_mac.load_state_dict(self.mac.state_dict())

    def cuda(self):
        """Move all components to CUDA."""
        self.mac.cuda()
        if self.target_mac:
            self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
        if self.target_mixer is not None:
            self.target_mixer.cuda()
        if self.belief_encoder is not None: 
            self.belief_encoder.cuda()
        if self.target_belief_encoder is not None: 
            self.target_belief_encoder.cuda()

    def save_models(self, path: str):
        """Save all model components."""
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), f"{path}/mixer.th")
        if self.belief_encoder is not None and not hasattr(self.mac, 'belief_encoder'):
             torch.save(self.belief_encoder.state_dict(), f"{path}/belief_encoder.th")
        
        # Save optimizers for checkpointing
        if self.belief_optimizer:
            torch.save(self.belief_optimizer.state_dict(), f"{path}/belief_opt.pth")
        if self.encoder_optimizer:
            torch.save(self.encoder_optimizer.state_dict(), f"{path}/encoder_opt.pth")
        if self.mixer_optimizer:
            torch.save(self.mixer_optimizer.state_dict(), f"{path}/mixer_opt.pth")

    def load_models(self, path: str):
        """Load all model components."""
        self.mac.load_models(path)
        if self.mixer is not None and os.path.exists(f"{path}/mixer.th"):
            self.mixer.load_state_dict(torch.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        
        if self.belief_encoder is not None and not hasattr(self.mac, 'belief_encoder') and os.path.exists(f"{path}/belief_encoder.th"):
            self.belief_encoder.load_state_dict(torch.load(f"{path}/belief_encoder.th", map_location=lambda storage, loc: storage))

        self._update_targets()

        # Load optimizers if they exist
        if self.belief_optimizer and os.path.exists(f"{path}/belief_opt.pth"):
            self.belief_optimizer.load_state_dict(torch.load(f"{path}/belief_opt.pth"))
        if self.encoder_optimizer and os.path.exists(f"{path}/encoder_opt.pth"):
            self.encoder_optimizer.load_state_dict(torch.load(f"{path}/encoder_opt.pth"))
        if self.mixer_optimizer and os.path.exists(f"{path}/mixer_opt.pth"):
            self.mixer_optimizer.load_state_dict(torch.load(f"{path}/mixer_opt.pth"))