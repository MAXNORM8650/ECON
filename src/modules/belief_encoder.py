import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any


class BeliefEncoder(nn.Module):
    """
    共享置信编码器，聚合所有智能体的置信状态以产生群体表征。
    
    根据 ECON 框架，此模块接收所有智能体的置信状态 b_i，使用多头注意力
    机制处理它们，并输出一个群体级表征 E。
    """
    
    def __init__(self, belief_dim: int, n_agents: int, n_heads: int = 4, 
                 key_dim: int = 64, device: torch.device = None):
        """
        初始化置信编码器。
        
        Args:
            belief_dim: 置信状态的维度
            n_agents: 智能体数量
            n_heads: 注意力头数量
            key_dim: 每个注意力头的维度
            device: 计算设备
        """
        super(BeliefEncoder, self).__init__()
        
        self.belief_dim = belief_dim
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=belief_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # 输出投影层
        self.out_proj = nn.Linear(belief_dim, belief_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(belief_dim)
        
        # 前馈网络
        self.feedforward = nn.Sequential(
            nn.Linear(belief_dim, 4 * belief_dim),
            nn.ReLU(),
            nn.Linear(4 * belief_dim, belief_dim)
        )
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(belief_dim)
        
    def forward(self, belief_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播，聚合智能体置信状态产生群体表征。
        
        Args:
            belief_states: 所有智能体的置信状态 [batch_size, n_agents, belief_dim]
            
        Returns:
            群体表征 E [batch_size, belief_dim]
        """
        batch_size = belief_states.shape[0]
        
        # 应用多头注意力
        # belief_states: [batch_size, n_agents, belief_dim]
        attn_output, _ = self.multihead_attn(
            query=belief_states,
            key=belief_states,
            value=belief_states
        )
        # attn_output: [batch_size, n_agents, belief_dim]
        
        # 残差连接和层归一化
        attn_output = belief_states + attn_output
        attn_output = self.layer_norm(attn_output)
        
        # 前馈网络
        ff_output = self.feedforward(attn_output)
        
        # 残差连接和层归一化
        ff_output = attn_output + ff_output
        ff_output = self.final_layer_norm(ff_output)
        
        # 聚合所有智能体的表征，生成群体表征
        # 使用平均池化
        group_repr = ff_output.mean(dim=1)  # [batch_size, belief_dim]
        
        # 输出投影
        group_repr = self.out_proj(group_repr)
        
        return group_repr
    
    def compute_loss(self, td_loss_tot: torch.Tensor, 
                    td_losses_i: List[torch.Tensor], 
                    lambda_e: float) -> torch.Tensor:
        """
        计算置信编码器的正则化损失 L_e。
        
        根据ECON框架论文，此损失为:
        L_e(θ_e) = L_TD^tot(φ) + λ_e Σ_i L_TD^i(θ_i^B)
        
        Args:
            td_loss_tot: 全局 TD 损失 (L_TD^tot(φ))
            td_losses_i: 各个智能体的局部 TD 损失列表 (L_TD^i(θ_i^B))
            lambda_e: 编码器正则化权重
            
        Returns:
            编码器损失 L_e
        """
        # 根据 ECON 框架论文: 
        # L_e(θ_e) = L_TD^tot(φ) + λ_e Σ_i L_TD^i(θ_i^B)
        sum_local_td_losses = sum(td_losses_i)
        
        encoder_loss = td_loss_tot + lambda_e * sum_local_td_losses
        
        return encoder_loss 