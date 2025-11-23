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
    alpha_lr: float = 3e-5
    gamma: float = 0.95
    tau: float = 0.005

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_decay_factor: float = 0.995  # Decay LR by this factor every 1000 steps
    min_lr: float = 1e-5  # Minimum learning rate

    # Replay buffer
    buffer_capacity: int = 100000
    batch_size: int = 1024 
    warmup_steps: int = 5000  # INCREASED from 1000 - agent needs MORE exploration before learning

    # Adaptive batch sizing
    use_adaptive_batch: bool = True
    min_batch_size: int = 128  # Minimum batch size for very early training
    adaptive_batch_threshold: int = 5000  # Buffer size to reach full batch_size

    # Regularization
    weight_decay: float = 1e-3
    actor_dropout: float = 0.05  # REDUCED from 0.1 to 0.05
    critic_dropout: float = 0.1  # REDUCED from 0.2 to 0.1

    # Entropy tuning
    auto_entropy_tuning: bool = True
    target_entropy: float = 0.3  # -dim(action)
    use_adaptive_entropy: bool = False  # Gradually reduce target entropy over training
    entropy_decay_steps: int = 100000  # Steps over which to decay entropy target

    # Training
    update_ratio: float = 1.0  # 1 gradient update per env step
    max_grad_norm: float = 2.0  # INCREASED from 1.0 to 2.0

    # Progressive training
    use_curriculum: bool = True
    curriculum_threshold: int = 50  # Episodes before increasing difficulty
    
    # Adaptive Normalization (AN-SAC)
    use_adaptive_norm: bool = False
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
    Optimized replay buffer with recency-weighted sampling and stratification.

    Performance improvements:
    - Pre-allocated numpy arrays for O(1) sampling
    - Incremental stratification index maintenance
    - Vectorized operations
    """

    def __init__(
        self,
        capacity: int = 100000,
        recency_weight: float = 0.00005,
        stratify_ratio: Dict[str, float] = None,
        state_dim: int = 30,
        action_dim: int = 1
    ):
        self.capacity = capacity
        self.recency_weight = recency_weight
        self.stratify_ratio = stratify_ratio or {
            'winning': 0.2,
            'losing': 0.2,
            'neutral': 0.6
        }

        # Pre-allocated numpy arrays for better memory locality
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.insert_steps = np.zeros(capacity, dtype=np.int32)

        # Stratification indices (maintain incrementally)
        self.winning_indices = []
        self.losing_indices = []
        self.neutral_indices = []

        self.position = 0
        self.size = 0
        self.insert_count = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer with incremental stratification update."""
        idx = self.position

        # Remove old index from stratification lists if overwriting
        if self.size == self.capacity:
            old_reward = self.rewards[idx]
            if old_reward > 0.01:
                if idx in self.winning_indices:
                    self.winning_indices.remove(idx)
            elif old_reward < -0.01:
                if idx in self.losing_indices:
                    self.losing_indices.remove(idx)
            else:
                if idx in self.neutral_indices:
                    self.neutral_indices.remove(idx)

        # Store transition
        self.states[idx] = state
        self.actions[idx] = action if hasattr(action, '__len__') else [action]
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.insert_steps[idx] = self.insert_count

        # Add to appropriate stratification list
        if reward > 0.01:
            self.winning_indices.append(idx)
        elif reward < -0.01:
            self.losing_indices.append(idx)
        else:
            self.neutral_indices.append(idx)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.insert_count += 1

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample batch with recency weighting and stratification (optimized).

        Args:
            batch_size: Batch size

        Returns:
            Dictionary of batched transitions
        """
        if self.size < batch_size:
            return None

        # Calculate recency weights (vectorized)
        current_step = self.insert_count
        ages = current_step - self.insert_steps[:self.size]
        weights = np.exp(-self.recency_weight * ages)
        weights = weights / weights.sum()

        # Sample from each category
        n_winning = int(batch_size * self.stratify_ratio['winning'])
        n_losing = int(batch_size * self.stratify_ratio['losing'])
        n_neutral = batch_size - n_winning - n_losing

        sampled_indices = []

        # Sample winning
        if len(self.winning_indices) > 0:
            winning_weights = weights[self.winning_indices]
            winning_weights = winning_weights / winning_weights.sum()
            n_sample = min(n_winning, len(self.winning_indices))
            sampled_winning = np.random.choice(
                self.winning_indices,
                size=n_sample,
                replace=False,
                p=winning_weights
            )
            sampled_indices.extend(sampled_winning)

        # Sample losing
        if len(self.losing_indices) > 0:
            losing_weights = weights[self.losing_indices]
            losing_weights = losing_weights / losing_weights.sum()
            n_sample = min(n_losing, len(self.losing_indices))
            sampled_losing = np.random.choice(
                self.losing_indices,
                size=n_sample,
                replace=False,
                p=losing_weights
            )
            sampled_indices.extend(sampled_losing)

        # Sample neutral
        if len(self.neutral_indices) > 0:
            neutral_weights = weights[self.neutral_indices]
            neutral_weights = neutral_weights / neutral_weights.sum()
            n_sample = min(n_neutral, len(self.neutral_indices))
            sampled_neutral = np.random.choice(
                self.neutral_indices,
                size=n_sample,
                replace=False,
                p=neutral_weights
            )
            sampled_indices.extend(sampled_neutral)

        # If not enough samples, fill with random weighted sampling
        if len(sampled_indices) < batch_size:
            remaining = batch_size - len(sampled_indices)
            all_indices = np.arange(self.size)
            available_mask = np.ones(self.size, dtype=bool)
            available_mask[sampled_indices] = False
            available_indices = all_indices[available_mask]

            if len(available_indices) > 0:
                available_weights = weights[available_indices]
                available_weights = available_weights / available_weights.sum()
                n_sample = min(remaining, len(available_indices))
                additional = np.random.choice(
                    available_indices,
                    size=n_sample,
                    replace=False,
                    p=available_weights
                )
                sampled_indices.extend(additional)

        # Convert to numpy array for vectorized indexing
        sampled_indices = np.array(sampled_indices, dtype=np.int32)

        # Vectorized data extraction
        return {
            'states': self.states[sampled_indices],
            'actions': self.actions[sampled_indices],
            'rewards': self.rewards[sampled_indices],
            'next_states': self.next_states[sampled_indices],
            'dones': self.dones[sampled_indices]
        }
    
    def trim_old(self, keep_percentage: float = 0.7):
        """
        Trim oldest transitions (optimized for array-based buffer).

        Args:
            keep_percentage: Percentage of buffer to keep
        """
        if self.size < self.capacity * 0.9:
            return

        # Sort indices by insert step (keep most recent)
        keep_count = int(self.size * keep_percentage)
        sorted_indices = np.argsort(self.insert_steps[:self.size])[-keep_count:]

        # Reorganize buffer to keep only recent transitions
        self.states[:keep_count] = self.states[sorted_indices]
        self.actions[:keep_count] = self.actions[sorted_indices]
        self.rewards[:keep_count] = self.rewards[sorted_indices]
        self.next_states[:keep_count] = self.next_states[sorted_indices]
        self.dones[:keep_count] = self.dones[sorted_indices]
        self.insert_steps[:keep_count] = self.insert_steps[sorted_indices]

        # Rebuild stratification indices
        self.winning_indices.clear()
        self.losing_indices.clear()
        self.neutral_indices.clear()

        for i in range(keep_count):
            reward = self.rewards[i]
            if reward > 0.01:
                self.winning_indices.append(i)
            elif reward < -0.01:
                self.losing_indices.append(i)
            else:
                self.neutral_indices.append(i)

        old_size = self.size
        self.size = keep_count
        self.position = keep_count % self.capacity

        logger.info(f"Trimmed buffer from {old_size} to {self.size}")

    def __len__(self) -> int:
        return self.size


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

        # Learning rate schedulers
        if self.config.use_lr_scheduler:
            self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
                self.actor_optimizer,
                gamma=self.config.lr_decay_factor
            )
            self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
                self.critic_optimizer,
                gamma=self.config.lr_decay_factor
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

        # Automatic entropy tuning
        if self.config.auto_entropy_tuning:
            self.target_entropy = self.config.target_entropy
            self.initial_target_entropy = self.config.target_entropy  # Store initial value
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(0.2, device=device)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            recency_weight=0.00005,
            state_dim=state_dim,
            action_dim=self.config.action_dim
        )
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.buffer_ready_logged = False  # Flag pour logger une seule fois quand buffer est prêt

        # Adaptive Normalization state
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_m2 = 0.0
        self.reward_count = 0

        # Mixed precision training
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Pre-allocated pinned memory tensors for faster CPU->GPU transfer
        if torch.cuda.is_available():
            self.state_buffer = torch.zeros((1, state_dim), dtype=torch.float32).pin_memory()
            self.batch_states_buffer = torch.zeros((self.config.batch_size, state_dim), dtype=torch.float32).pin_memory()
            self.batch_actions_buffer = torch.zeros((self.config.batch_size, self.config.action_dim), dtype=torch.float32).pin_memory()
            self.batch_rewards_buffer = torch.zeros((self.config.batch_size, 1), dtype=torch.float32).pin_memory()
            self.batch_next_states_buffer = torch.zeros((self.config.batch_size, state_dim), dtype=torch.float32).pin_memory()
            self.batch_dones_buffer = torch.zeros((self.config.batch_size, 1), dtype=torch.float32).pin_memory()
        else:
            self.state_buffer = None
            self.batch_states_buffer = None
            self.batch_actions_buffer = None
            self.batch_rewards_buffer = None
            self.batch_next_states_buffer = None
            self.batch_dones_buffer = None

        logger.info(f"SAC Agent {agent_id} initialized on {device}")
        logger.info(f"  State dim: {state_dim}")
        logger.info(f"  Action dim: {self.config.action_dim}")
        logger.info(f"  Hidden dims: {self.config.hidden_dims}")
        logger.info(f"  Gamma: {self.config.gamma}")
        logger.info(f"  Use regime Q-functions: {self.config.use_regime_qfuncs}")
        logger.info(f"  PERFORMANCE OPTIMIZATIONS ENABLED:")
        logger.info(f"    - Optimized ReplayBuffer: Pre-allocated arrays with O(1) sampling")
        logger.info(f"    - Pinned memory transfers: Faster CPU->GPU data movement")
        logger.info(f"    - Mixed precision training: {self.use_amp}")
        logger.info(f"    - Vectorized reward normalization: Eliminates per-reward loops")
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        regime: Optional[str] = None
    ) -> np.ndarray:
        """
        Select action from policy (optimized with pinned memory).

        Args:
            state: Current state
            deterministic: If True, return mean action
            regime: Current regime (for Agent 3)

        Returns:
            Action array
        """
        with torch.no_grad():
            if self.state_buffer is not None:
                # Use pre-allocated pinned buffer for faster transfer
                self.state_buffer[0] = torch.from_numpy(state)
                state_tensor = self.state_buffer.to(device, non_blocking=True)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            action, _ = self.actor.sample(state_tensor, deterministic=deterministic)

        return action.cpu().numpy()[0]
    
    def update(
        self,
        regime: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Update networks with one batch (optimized with pinned memory and mixed precision).

        Args:
            regime: Current regime (for Agent 3)

        Returns:
            Dictionary of losses
        """
        # Adaptive batch sizing based on buffer fill level
        if self.config.use_adaptive_batch:
            buffer_size = len(self.replay_buffer)
            if buffer_size < self.config.adaptive_batch_threshold:
                # Scale batch size linearly with buffer fill
                progress = buffer_size / self.config.adaptive_batch_threshold
                current_batch_size = int(
                    self.config.min_batch_size +
                    (self.config.batch_size - self.config.min_batch_size) * progress
                )
                # Ensure we don't sample more than 25% of buffer
                current_batch_size = min(current_batch_size, buffer_size // 4)
            else:
                current_batch_size = self.config.batch_size
        else:
            current_batch_size = self.config.batch_size

        # Sample batch
        batch = self.replay_buffer.sample(current_batch_size)
        if batch is None:
            return {}

        # Log une fois quand le replay buffer est plein et que les updates commencent
        if not self.buffer_ready_logged:
            self.buffer_ready_logged = True
            logger.info("="*80)
            logger.info(f"🚀 REPLAY BUFFER READY - Starting model updates!")
            logger.info(f"   Buffer size: {len(self.replay_buffer)}/{self.config.buffer_capacity}")
            logger.info(f"   Batch size: {self.config.batch_size}")
            logger.info(f"   Updates will now occur at EVERY step (1 update/step)")
            logger.info("="*80)

        # Convert to tensors using pinned memory buffers
        if self.batch_states_buffer is not None:
            actual_batch_size = len(batch['states'])

            # Copy data to pinned buffers
            self.batch_states_buffer[:actual_batch_size].copy_(torch.from_numpy(batch['states']))
            self.batch_actions_buffer[:actual_batch_size].copy_(torch.from_numpy(batch['actions']))
            self.batch_rewards_buffer[:actual_batch_size, 0].copy_(torch.from_numpy(batch['rewards']))
            self.batch_next_states_buffer[:actual_batch_size].copy_(torch.from_numpy(batch['next_states']))
            self.batch_dones_buffer[:actual_batch_size, 0].copy_(torch.from_numpy(batch['dones']))

            # Async transfer to GPU
            states = self.batch_states_buffer[:actual_batch_size].to(device, non_blocking=True)
            actions = self.batch_actions_buffer[:actual_batch_size].to(device, non_blocking=True)
            rewards = self.batch_rewards_buffer[:actual_batch_size].to(device, non_blocking=True)
            next_states = self.batch_next_states_buffer[:actual_batch_size].to(device, non_blocking=True)
            dones = self.batch_dones_buffer[:actual_batch_size].to(device, non_blocking=True)
        else:
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

        # Update scaler once at the end (for mixed precision)
        if self.use_amp:
            self.scaler.update()

        # Adaptive entropy scheduling: gradually reduce target entropy to encourage exploitation
        if self.config.use_adaptive_entropy and self.config.auto_entropy_tuning:
            progress = min(1.0, self.total_steps / self.config.entropy_decay_steps)
            # Start at -1.0, decay to -0.5 (less exploration over time)
            self.target_entropy = self.initial_target_entropy * (1.0 - 0.5 * progress)

        # Step learning rate schedulers every 1000 steps
        if self.config.use_lr_scheduler and self.total_steps % 1000 == 0 and self.total_steps > 0:
            # Get current LRs before stepping
            current_actor_lr = self.actor_optimizer.param_groups[0]['lr']
            current_critic_lr = self.critic_optimizer.param_groups[0]['lr']

            # Only decay if above minimum LR
            if current_actor_lr > self.config.min_lr:
                self.actor_scheduler.step()
            if current_critic_lr > self.config.min_lr:
                self.critic_scheduler.step()

            # Log LR changes every 5000 steps
            if self.total_steps % 5000 == 0:
                logger.info(
                    f"Learning rates at step {self.total_steps}: "
                    f"Actor={self.actor_optimizer.param_groups[0]['lr']:.6f}, "
                    f"Critic={self.critic_optimizer.param_groups[0]['lr']:.6f}"
                )

        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss if alpha_loss is not None else 0.0,
            'alpha': self.alpha.item()
        }
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics (vectorized)."""
        # Vectorized update of running statistics (Welford's algorithm)
        rewards_np = rewards.detach().cpu().numpy().flatten()
        batch_size = len(rewards_np)

        # Update statistics vectorized
        old_mean = self.reward_mean
        old_count = self.reward_count

        self.reward_count += batch_size
        delta = rewards_np - old_mean
        self.reward_mean += np.sum(delta) / self.reward_count

        # Update M2 for variance calculation
        delta2 = rewards_np - self.reward_mean
        self.reward_m2 += np.sum(delta * delta2)

        if self.reward_count > 1:
            self.reward_std = np.sqrt(self.reward_m2 / self.reward_count)

        # Normalize on GPU
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
        """Update critic networks with mixed precision."""
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Sample next actions
                    next_actions, next_log_probs = self.actor.sample(next_states)

                    # Compute target Q-values
                    if self.config.use_regime_qfuncs:
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
                    target_q = rewards + (1 - dones) * self.config.gamma * target_q
            else:
                next_actions, next_log_probs = self.actor.sample(next_states)

                if self.config.use_regime_qfuncs:
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
                target_q = rewards + (1 - dones) * self.config.gamma * target_q

        # Forward pass with mixed precision
        self.critic_optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
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

            # Backward with gradient scaling
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.critic1.parameters() if not self.config.use_regime_qfuncs else
                list(self.critic1_low.parameters()) + list(self.critic1_high.parameters()),
                self.config.max_grad_norm
            )
            self.scaler.step(self.critic_optimizer)
        else:
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
        """Update actor network and alpha with mixed precision."""
        self.actor_optimizer.zero_grad()

        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast():
                actions, log_probs = self.actor.sample(states)

                # Detach Q-values to prevent backprop through critic
                if self.config.use_regime_qfuncs:
                    if regime == 'low_vol':
                        q1 = self.critic1_low(states, actions).detach()
                        q2 = self.critic2_low(states, actions).detach()
                    else:
                        q1 = self.critic1_high(states, actions).detach()
                        q2 = self.critic2_high(states, actions).detach()
                else:
                    q1 = self.critic1(states, actions).detach()
                    q2 = self.critic2(states, actions).detach()

                q = torch.min(q1, q2)
                actor_loss = (self.alpha * log_probs - q).mean()

            # Backward with gradient scaling
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.actor_optimizer)
        else:
            actions, log_probs = self.actor.sample(states)

            # Detach Q-values to prevent backprop through critic
            if self.config.use_regime_qfuncs:
                if regime == 'low_vol':
                    q1 = self.critic1_low(states, actions).detach()
                    q2 = self.critic2_low(states, actions).detach()
                else:
                    q1 = self.critic1_high(states, actions).detach()
                    q2 = self.critic2_high(states, actions).detach()
            else:
                q1 = self.critic1(states, actions).detach()
                q2 = self.critic2(states, actions).detach()

            q = torch.min(q1, q2)
            actor_loss = (self.alpha * log_probs - q).mean()

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

        # Update alpha
        alpha_loss = None
        if self.config.auto_entropy_tuning:
            self.alpha_optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                self.scaler.scale(alpha_loss).backward()
                self.scaler.step(self.alpha_optimizer)
            else:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
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

        # PyTorch 2.6+ nécessite weights_only=False pour charger des classes personnalisées
        checkpoint = torch.load(input_path, map_location=device, weights_only=False)
        
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
