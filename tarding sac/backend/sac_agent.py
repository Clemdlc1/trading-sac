"""
SAC EUR/USD Trading System - SAC Agent
=======================================

This module implements the Soft Actor-Critic (SAC) reinforcement learning agent
with Adaptive Normalization (AN-SAC).

Features:
- Actor-Critic architecture with Spectral Normalization
- Layer Normalization after spectral norm
- Adaptive entropy tuning
- Recency-weighted replay buffer with stratification
- Support for HMM regime detection (Agent 3)
- Checkpoint save/load with full state
- Re-training from checkpoint support

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import pickle
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


@dataclass
class SACConfig:
    """Configuration for SAC agent."""
    
    # Network architecture
    state_dim: int = 30
    action_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    
    # Learning parameters
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005
    
    # Replay buffer
    buffer_capacity: int = 100000
    batch_size: int = 512
    warmup_steps: int = 10000
    
    # Regularization
    weight_decay: float = 1e-3
    actor_dropout: float = 0.1
    critic_dropout: float = 0.2
    
    # Entropy tuning
    auto_entropy_tuning: bool = True
    target_entropy: float = -1.0  # -dim(action)
    
    # Training
    update_ratio: float = 1.0  # 1 gradient update per env step
    max_grad_norm: float = 1.0
    
    # Adaptive Normalization (AN-SAC)
    use_adaptive_norm: bool = True
    reward_scale: float = 1.0
    
    # HMM support (for Agent 3)
    use_regime_qfuncs: bool = False
    state_dim_with_regime: int = 32  # 30 + 2 regime features
    
    # Checkpointing
    models_dir: Path = Path("models/checkpoints")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)


class SpectralNormLinear(nn.Module):
    """Linear layer with Spectral Normalization."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = spectral_norm(nn.Linear(in_features, out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Actor(nn.Module):
    """
    Actor network with Spectral + Layer Normalization.
    
    Outputs mean and log_std for Gaussian policy.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            # Spectral Normalized Linear
            layers.append(SpectralNormLinear(input_dim, hidden_dim))
            # Layer Normalization
            layers.append(nn.LayerNorm(hidden_dim))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads (no spectral norm on output)
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (mean, log_std)
        """
        features = self.feature_extractor(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clip log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            
            # Apply tanh squashing
            action = torch.tanh(z)
            
            # Calculate log probability
            log_prob = normal.log_prob(z)
            # Correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """
    Critic network (Q-function) with Spectral + Layer Normalization.
    
    Takes state and action as input, outputs Q-value.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Build network
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            # Spectral Normalized Linear
            layers.append(SpectralNormLinear(input_dim, hidden_dim))
            # Layer Normalization
            layers.append(nn.LayerNorm(hidden_dim))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output head (no spectral norm)
        self.q_head = nn.Linear(input_dim, 1)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Q-value tensor [batch_size, 1]
        """
        x = torch.cat([state, action], dim=-1)
        features = self.network(x)
        q_value = self.q_head(features)
        return q_value


class ReplayBuffer:
    """
    Replay buffer with recency-weighted sampling and stratification.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        recency_weight: float = 0.00005,
        stratify_ratio: Dict[str, float] = None
    ):
        self.capacity = capacity
        self.recency_weight = recency_weight
        self.stratify_ratio = stratify_ratio or {
            'winning': 0.2,
            'losing': 0.2,
            'neutral': 0.6
        }
        
        self.buffer = deque(maxlen=capacity)
        self.insert_count = 0
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'insert_step': self.insert_count
        })
        self.insert_count += 1
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample batch with recency weighting and stratification.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary of batched transitions
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate recency weights
        current_step = self.insert_count
        ages = np.array([current_step - t['insert_step'] for t in self.buffer])
        weights = np.exp(-self.recency_weight * ages)
        weights = weights / weights.sum()
        
        # Stratified sampling
        # Classify transitions
        winning_indices = []
        losing_indices = []
        neutral_indices = []
        
        for i, transition in enumerate(self.buffer):
            reward = transition['reward']
            if reward > 0.01:
                winning_indices.append(i)
            elif reward < -0.01:
                losing_indices.append(i)
            else:
                neutral_indices.append(i)
        
        # Sample from each category
        n_winning = int(batch_size * self.stratify_ratio['winning'])
        n_losing = int(batch_size * self.stratify_ratio['losing'])
        n_neutral = batch_size - n_winning - n_losing
        
        sampled_indices = []
        
        # Sample winning (with replacement if needed)
        if len(winning_indices) > 0:
            winning_weights = weights[winning_indices]
            winning_weights = winning_weights / winning_weights.sum()
            sampled_winning = np.random.choice(
                winning_indices,
                size=min(n_winning, len(winning_indices)),
                replace=False,
                p=winning_weights
            )
            sampled_indices.extend(sampled_winning)
        
        # Sample losing
        if len(losing_indices) > 0:
            losing_weights = weights[losing_indices]
            losing_weights = losing_weights / losing_weights.sum()
            sampled_losing = np.random.choice(
                losing_indices,
                size=min(n_losing, len(losing_indices)),
                replace=False,
                p=losing_weights
            )
            sampled_indices.extend(sampled_losing)
        
        # Sample neutral
        if len(neutral_indices) > 0:
            neutral_weights = weights[neutral_indices]
            neutral_weights = neutral_weights / neutral_weights.sum()
            sampled_neutral = np.random.choice(
                neutral_indices,
                size=min(n_neutral, len(neutral_indices)),
                replace=False,
                p=neutral_weights
            )
            sampled_indices.extend(sampled_neutral)
        
        # If not enough samples, fill with random weighted sampling
        if len(sampled_indices) < batch_size:
            remaining = batch_size - len(sampled_indices)
            all_indices = list(range(len(self.buffer)))
            available_indices = [i for i in all_indices if i not in sampled_indices]
            if len(available_indices) > 0:
                available_weights = weights[available_indices]
                available_weights = available_weights / available_weights.sum()
                additional = np.random.choice(
                    available_indices,
                    size=min(remaining, len(available_indices)),
                    replace=False,
                    p=available_weights
                )
                sampled_indices.extend(additional)
        
        # Get transitions
        batch = [self.buffer[i] for i in sampled_indices]
        
        # Stack into arrays
        states = np.array([t['state'] for t in batch])
        actions = np.array([t['action'] for t in batch])
        rewards = np.array([t['reward'] for t in batch])
        next_states = np.array([t['next_state'] for t in batch])
        dones = np.array([t['done'] for t in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def trim_old(self, keep_percentage: float = 0.7):
        """
        Trim oldest transitions.
        
        Args:
            keep_percentage: Percentage of buffer to keep
        """
        if len(self.buffer) < self.capacity * 0.9:
            return
        
        # Sort by insert step
        sorted_buffer = sorted(self.buffer, key=lambda x: x['insert_step'], reverse=True)
        keep_count = int(len(sorted_buffer) * keep_percentage)
        self.buffer = deque(sorted_buffer[:keep_count], maxlen=self.capacity)
        
        logger.info(f"Trimmed buffer from {len(sorted_buffer)} to {len(self.buffer)}")
    
    def __len__(self) -> int:
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic agent with Adaptive Normalization.
    """
    
    def __init__(
        self,
        config: Optional[SACConfig] = None,
        agent_id: int = 1
    ):
        self.config = config or SACConfig()
        self.agent_id = agent_id
        
        # Networks
        if self.config.use_regime_qfuncs:
            # Agent 3: Separate Q-functions for each regime
            state_dim = self.config.state_dim_with_regime
            self.actor = Actor(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.actor_dropout
            ).to(device)
            
            self.critic1_low = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            self.critic2_low = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            self.critic1_high = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            self.critic2_high = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            # Target networks
            self.critic1_low_target = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            self.critic2_low_target = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            self.critic1_high_target = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            self.critic2_high_target = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            # Copy parameters
            self.critic1_low_target.load_state_dict(self.critic1_low.state_dict())
            self.critic2_low_target.load_state_dict(self.critic2_low.state_dict())
            self.critic1_high_target.load_state_dict(self.critic1_high.state_dict())
            self.critic2_high_target.load_state_dict(self.critic2_high.state_dict())
            
        else:
            # Agents 1 & 2: Standard Q-functions
            state_dim = self.config.state_dim_with_regime if agent_id in [1, 2] else self.config.state_dim
            
            self.actor = Actor(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.actor_dropout
            ).to(device)
            
            self.critic1 = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            self.critic2 = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            # Target networks
            self.critic1_target = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            self.critic2_target = Critic(
                state_dim,
                self.config.action_dim,
                self.config.hidden_dims,
                self.config.critic_dropout
            ).to(device)
            
            # Copy parameters
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr
        )
        
        if self.config.use_regime_qfuncs:
            self.critic_optimizer = optim.Adam(
                list(self.critic1_low.parameters()) +
                list(self.critic2_low.parameters()) +
                list(self.critic1_high.parameters()) +
                list(self.critic2_high.parameters()),
                lr=self.config.critic_lr,
                weight_decay=self.config.weight_decay
            )
        else:
            self.critic_optimizer = optim.Adam(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                lr=self.config.critic_lr,
                weight_decay=self.config.weight_decay
            )
        
        # Automatic entropy tuning
        if self.config.auto_entropy_tuning:
            self.target_entropy = self.config.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(0.2, device=device)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            recency_weight=0.00005
        )
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
        
        # Adaptive Normalization state
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_m2 = 0.0
        self.reward_count = 0
        
        logger.info(f"SAC Agent {agent_id} initialized on {device}")
        logger.info(f"  State dim: {state_dim}")
        logger.info(f"  Action dim: {self.config.action_dim}")
        logger.info(f"  Hidden dims: {self.config.hidden_dims}")
        logger.info(f"  Gamma: {self.config.gamma}")
        logger.info(f"  Use regime Q-functions: {self.config.use_regime_qfuncs}")
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        regime: Optional[str] = None
    ) -> np.ndarray:
        """
        Select action from policy.
        
        Args:
            state: Current state
            deterministic: If True, return mean action
            regime: Current regime (for Agent 3)
            
        Returns:
            Action array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, deterministic=deterministic)
        
        action = action.cpu().numpy()[0]
        return action
    
    def update(
        self,
        regime: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Update networks with one batch.
        
        Args:
            regime: Current regime (for Agent 3)
            
        Returns:
            Dictionary of losses
        """
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(device)
        actions = torch.FloatTensor(batch['actions']).to(device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(batch['next_states']).to(device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(device)
        
        # Adaptive reward normalization
        if self.config.use_adaptive_norm:
            rewards = self._normalize_rewards(rewards)
        
        # Update critics
        critic_loss = self._update_critics(
            states, actions, rewards, next_states, dones, regime
        )
        
        # Update actor
        actor_loss, alpha_loss = self._update_actor(states, regime)
        
        # Update target networks
        self._soft_update_targets()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss if alpha_loss is not None else 0.0,
            'alpha': self.alpha.item()
        }
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics."""
        # Update running statistics (Welford's algorithm)
        for reward in rewards:
            reward_val = reward.item()
            self.reward_count += 1
            delta = reward_val - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = reward_val - self.reward_mean
            self.reward_m2 += delta * delta2
        
        if self.reward_count > 1:
            self.reward_std = np.sqrt(self.reward_m2 / self.reward_count)
        
        # Normalize
        normalized = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        return normalized
    
    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        regime: Optional[str] = None
    ) -> float:
        """Update critic networks."""
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute target Q-values
            if self.config.use_regime_qfuncs:
                # Agent 3: Use regime-specific critics
                if regime == 'low_vol':
                    target_q1 = self.critic1_low_target(next_states, next_actions)
                    target_q2 = self.critic2_low_target(next_states, next_actions)
                else:
                    target_q1 = self.critic1_high_target(next_states, next_actions)
                    target_q2 = self.critic2_high_target(next_states, next_actions)
            else:
                target_q1 = self.critic1_target(next_states, next_actions)
                target_q2 = self.critic2_target(next_states, next_actions)
            
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - self.alpha * next_log_probs
            
            # Bellman backup
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        # Compute critic losses
        if self.config.use_regime_qfuncs:
            if regime == 'low_vol':
                current_q1 = self.critic1_low(states, actions)
                current_q2 = self.critic2_low(states, actions)
            else:
                current_q1 = self.critic1_high(states, actions)
                current_q2 = self.critic2_high(states, actions)
        else:
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic1.parameters() if not self.config.use_regime_qfuncs else
            list(self.critic1_low.parameters()) + list(self.critic1_high.parameters()),
            self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(
        self,
        states: torch.Tensor,
        regime: Optional[str] = None
    ) -> Tuple[float, Optional[float]]:
        """Update actor network and alpha (if auto-tuning)."""
        # Sample actions
        actions, log_probs = self.actor.sample(states)
        
        # Compute Q-values
        if self.config.use_regime_qfuncs:
            if regime == 'low_vol':
                q1 = self.critic1_low(states, actions)
                q2 = self.critic2_low(states, actions)
            else:
                q1 = self.critic1_high(states, actions)
                q2 = self.critic2_high(states, actions)
        else:
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
        
        q = torch.min(q1, q2)
        
        # Actor loss
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.max_grad_norm
        )
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = None
        if self.config.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()
        
        return actor_loss.item(), alpha_loss
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        tau = self.config.tau
        
        if self.config.use_regime_qfuncs:
            # Update low volatility targets
            for param, target_param in zip(
                self.critic1_low.parameters(),
                self.critic1_low_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            
            for param, target_param in zip(
                self.critic2_low.parameters(),
                self.critic2_low_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            
            # Update high volatility targets
            for param, target_param in zip(
                self.critic1_high.parameters(),
                self.critic1_high_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            
            for param, target_param in zip(
                self.critic2_high.parameters(),
                self.critic2_high_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
        else:
            for param, target_param in zip(
                self.critic1.parameters(),
                self.critic1_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            
            for param, target_param in zip(
                self.critic2.parameters(),
                self.critic2_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
    
    def save(self, filename: str) -> Path:
        """
        Save agent checkpoint.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.config.models_dir / filename
        
        checkpoint = {
            'config': self.config,
            'agent_id': self.agent_id,
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_m2': self.reward_m2,
            'reward_count': self.reward_count
        }
        
        if self.config.use_regime_qfuncs:
            checkpoint.update({
                'critic1_low_state_dict': self.critic1_low.state_dict(),
                'critic2_low_state_dict': self.critic2_low.state_dict(),
                'critic1_high_state_dict': self.critic1_high.state_dict(),
                'critic2_high_state_dict': self.critic2_high.state_dict(),
                'critic1_low_target_state_dict': self.critic1_low_target.state_dict(),
                'critic2_low_target_state_dict': self.critic2_low_target.state_dict(),
                'critic1_high_target_state_dict': self.critic1_high_target.state_dict(),
                'critic2_high_target_state_dict': self.critic2_high_target.state_dict(),
            })
        else:
            checkpoint.update({
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'critic1_target_state_dict': self.critic1_target.state_dict(),
                'critic2_target_state_dict': self.critic2_target.state_dict(),
            })
        
        if self.config.auto_entropy_tuning:
            checkpoint.update({
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
            })
        
        torch.save(checkpoint, output_path)
        logger.info(f"Agent {self.agent_id} checkpoint saved to {output_path}")
        
        return output_path
    
    def load(self, filename: str):
        """
        Load agent checkpoint.
        
        Args:
            filename: Input filename
        """
        input_path = self.config.models_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {input_path}")
        
        checkpoint = torch.load(input_path, map_location=device)
        
        # Load networks
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        
        if self.config.use_regime_qfuncs:
            self.critic1_low.load_state_dict(checkpoint['critic1_low_state_dict'])
            self.critic2_low.load_state_dict(checkpoint['critic2_low_state_dict'])
            self.critic1_high.load_state_dict(checkpoint['critic1_high_state_dict'])
            self.critic2_high.load_state_dict(checkpoint['critic2_high_state_dict'])
            self.critic1_low_target.load_state_dict(checkpoint['critic1_low_target_state_dict'])
            self.critic2_low_target.load_state_dict(checkpoint['critic2_low_target_state_dict'])
            self.critic1_high_target.load_state_dict(checkpoint['critic1_high_target_state_dict'])
            self.critic2_high_target.load_state_dict(checkpoint['critic2_high_target_state_dict'])
        else:
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
            self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        # Load optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.config.auto_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp()
        
        # Load training state
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        self.reward_mean = checkpoint['reward_mean']
        self.reward_std = checkpoint['reward_std']
        self.reward_m2 = checkpoint['reward_m2']
        self.reward_count = checkpoint['reward_count']
        
        logger.info(f"Agent {self.agent_id} checkpoint loaded from {input_path}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Episodes: {self.episode_count}")


def main():
    """Example usage of SAC agent."""
    from backend.data_pipeline import DataPipeline
    from backend.feature_engineering import FeaturePipeline
    from backend.trading_env import TradingEnvironment, TradingEnvConfig
    
    # Load data
    logger.info("Loading data and features...")
    data_pipeline = DataPipeline()
    train_data, val_data, test_data = data_pipeline.get_processed_data()
    
    feature_pipeline = FeaturePipeline()
    train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
        train_data, val_data, test_data
    )
    
    # Create environment
    logger.info("Creating environment...")
    env_config = TradingEnvConfig()
    env = TradingEnvironment(
        data=train_data['EURUSD'],
        features=train_features,
        config=env_config,
        eval_mode=False
    )
    
    # Create agent
    logger.info("Creating SAC agent...")
    sac_config = SACConfig(
        state_dim=30,
        action_dim=1,
        gamma=0.95,
        hidden_dims=[256, 256]
    )
    agent = SACAgent(config=sac_config, agent_id=1)
    
    # Training loop
    logger.info("\nStarting training...")
    num_episodes = 5
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        logger.info(f"{'='*80}")
        
        while not done:
            # Select action (random during warmup)
            if agent.total_steps < agent.config.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent
            if agent.total_steps >= agent.config.warmup_steps:
                losses = agent.update()
                
                if agent.total_steps % 1000 == 0 and losses:
                    logger.info(
                        f"Step {agent.total_steps}: "
                        f"Critic Loss={losses['critic_loss']:.4f}, "
                        f"Actor Loss={losses['actor_loss']:.4f}, "
                        f"Alpha={losses['alpha']:.4f}"
                    )
            
            episode_reward += reward
            state = next_state
            agent.total_steps += 1
        
        agent.episode_count += 1
        
        # Episode summary
        metrics = env.get_episode_metrics()
        logger.info(f"\nEpisode {episode + 1} Complete:")
        logger.info(f"  Episode Reward: {episode_reward:.2f}")
        logger.info(f"  Final Equity: ${metrics['final_equity']:.2f}")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Total Steps: {agent.total_steps}")
    
    # Save agent
    logger.info("\nSaving agent...")
    agent.save(f"sac_agent_{agent.agent_id}_test.pt")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
