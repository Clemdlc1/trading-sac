"""
SAC EUR/USD Trading System - Ensemble Meta-Controller
======================================================

This module implements the meta-controller for aggregating actions from
the 3 SAC agents using a neural network trained with hindsight learning.

Features:
- Neural meta-controller with [128, 64] architecture
- Agent confidence calculation based on Q-values
- Weighted action aggregation
- Hindsight learning from episode trajectories
- Support for HMM regime detection
- Checkpoint save/load
- Production-ready ensemble inference

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
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
class MetaControllerConfig:
    """Configuration for meta-controller."""
    
    # Architecture
    state_dim: int = 30
    n_agents: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    
    # Training
    learning_rate: float = 1e-4
    hindsight_episodes: int = 50
    batch_size: int = 256
    max_training_steps: int = 10000
    
    # Agent configurations
    agent_configs: Dict[int, Dict] = field(default_factory=lambda: {
        1: {
            'name': 'Short-term',
            'gamma': 0.93,
            'hidden_dims': [256, 256],
            'max_position': 2.0,
            'focus': 'Scalping, short-term patterns'
        },
        2: {
            'name': 'Medium-term Balanced',
            'gamma': 0.95,
            'hidden_dims': [256, 128],
            'max_position': 1.5,
            'focus': 'Medium-term trading'
        },
        3: {
            'name': 'Swing Adaptive',
            'gamma': 0.97,
            'hidden_dims': [128, 128],
            'max_position': 2.0,
            'focus': 'Swing trading, long-term patterns',
            'use_regime_qfuncs': True
        }
    })
    
    # Model paths
    models_dir: Path = Path("models/ensemble")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)


class MetaControllerNetwork(nn.Module):
    """
    Neural network meta-controller.
    
    Takes state + agent confidences, outputs action weights.
    """
    
    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        
        # Input: state + agent confidences
        input_dim = state_dim + n_agents
        
        # Build network
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output layer: weights for each agent
        self.output_layer = nn.Linear(current_dim, n_agents)
    
    def forward(
        self,
        state: torch.Tensor,
        confidences: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
            confidences: Agent confidences [batch_size, n_agents]
            
        Returns:
            Agent weights [batch_size, n_agents] (sum to 1)
        """
        # Concatenate state and confidences
        x = torch.cat([state, confidences], dim=-1)
        
        # Forward through network
        features = self.network(x)
        logits = self.output_layer(features)
        
        # Apply softmax to get weights
        weights = F.softmax(logits, dim=-1)
        
        return weights


class EpisodeBuffer:
    """Buffer for storing episode trajectories for hindsight learning."""
    
    def __init__(self, max_episodes: int = 100):
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
    
    def add_episode(self, episode_data: Dict):
        """
        Add episode trajectory.
        
        Args:
            episode_data: Dictionary containing:
                - states: [T, state_dim]
                - agent_actions: [T, n_agents, action_dim]
                - agent_returns: [T, n_agents]
                - agent_q_values: [T, n_agents]
                - rewards: [T]
                - final_return: float
        """
        self.episodes.append(episode_data)
    
    def get_episodes(self, n: Optional[int] = None) -> List[Dict]:
        """
        Get recent episodes.
        
        Args:
            n: Number of episodes to return (None = all)
            
        Returns:
            List of episode dictionaries
        """
        if n is None:
            return list(self.episodes)
        else:
            return list(self.episodes)[-n:]
    
    def __len__(self) -> int:
        return len(self.episodes)


class MetaController:
    """
    Meta-controller for agent ensemble.
    
    Manages 3 SAC agents and aggregates their actions using learned weights.
    """
    
    def __init__(
        self,
        agents: List,
        config: Optional[MetaControllerConfig] = None
    ):
        """
        Initialize meta-controller.
        
        Args:
            agents: List of 3 SAC agents
            config: Meta-controller configuration
        """
        self.config = config or MetaControllerConfig()
        
        if len(agents) != self.config.n_agents:
            raise ValueError(f"Expected {self.config.n_agents} agents, got {len(agents)}")
        
        self.agents = agents
        
        # Initialize meta-controller network
        self.network = MetaControllerNetwork(
            state_dim=self.config.state_dim,
            n_agents=self.config.n_agents,
            hidden_dims=self.config.hidden_dims
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Episode buffer for hindsight learning
        self.episode_buffer = EpisodeBuffer(max_episodes=100)
        
        # Training state
        self.training_step = 0
        self.is_trained = False
        
        logger.info("Meta-Controller initialized")
        logger.info(f"  Number of agents: {self.config.n_agents}")
        logger.info(f"  State dim: {self.config.state_dim}")
        logger.info(f"  Hidden dims: {self.config.hidden_dims}")
        
        # Log agent configurations
        for i, agent_config in self.config.agent_configs.items():
            logger.info(f"\n  Agent {i} - {agent_config['name']}:")
            logger.info(f"    Gamma: {agent_config['gamma']}")
            logger.info(f"    Architecture: {agent_config['hidden_dims']}")
            logger.info(f"    Max position: {agent_config['max_position']}")
            logger.info(f"    Focus: {agent_config['focus']}")
    
    def calculate_agent_confidences(
        self,
        state: np.ndarray,
        regime: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate confidence for each agent based on Q-values.
        
        Formula: confidence_i = -abs(Q_value_i - mean(Q_values))
        (Closer to mean = more confident)
        
        Args:
            state: Current state
            regime: Current regime (for Agent 3)
            
        Returns:
            Array of confidences [n_agents]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        q_values = []
        
        with torch.no_grad():
            for i, agent in enumerate(self.agents):
                # Sample action from agent
                action, _ = agent.actor.sample(state_tensor, deterministic=True)
                
                # Get Q-value
                if i == 2 and hasattr(agent, 'critic1_low'):  # Agent 3 with regime Q-funcs
                    if regime == 'low_vol':
                        q1 = agent.critic1_low(state_tensor, action)
                        q2 = agent.critic2_low(state_tensor, action)
                    else:
                        q1 = agent.critic1_high(state_tensor, action)
                        q2 = agent.critic2_high(state_tensor, action)
                else:
                    q1 = agent.critic1(state_tensor, action)
                    q2 = agent.critic2(state_tensor, action)
                
                # Use minimum Q-value (conservative)
                q_value = torch.min(q1, q2).item()
                q_values.append(q_value)
        
        q_values = np.array(q_values)
        mean_q = np.mean(q_values)
        
        # Calculate confidences
        confidences = -np.abs(q_values - mean_q)
        
        return confidences
    
    def get_agent_actions(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        regime: Optional[str] = None
    ) -> np.ndarray:
        """
        Get actions from all agents.
        
        Args:
            state: Current state
            deterministic: If True, use deterministic actions
            regime: Current regime (for Agent 3)
            
        Returns:
            Array of actions [n_agents, action_dim]
        """
        actions = []
        
        for agent in self.agents:
            action = agent.select_action(state, deterministic=deterministic, regime=regime)
            actions.append(action)
        
        return np.array(actions)
    
    def get_ensemble_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        regime: Optional[str] = None,
        return_details: bool = False
    ) -> np.ndarray:
        """
        Get aggregated action from ensemble.
        
        Args:
            state: Current state
            deterministic: If True, use deterministic actions
            regime: Current regime (for Agent 3)
            return_details: If True, return dict with weights and individual actions
            
        Returns:
            Aggregated action or dictionary with details
        """
        # Get actions from all agents
        agent_actions = self.get_agent_actions(state, deterministic, regime)
        
        # Calculate confidences
        confidences = self.calculate_agent_confidences(state, regime)
        
        # Get weights from meta-controller
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        confidences_tensor = torch.FloatTensor(confidences).unsqueeze(0).to(device)
        
        with torch.no_grad():
            weights = self.network(state_tensor, confidences_tensor)
            weights = weights.cpu().numpy()[0]
        
        # Aggregate actions
        ensemble_action = np.sum(weights.reshape(-1, 1) * agent_actions, axis=0)
        
        if return_details:
            return {
                'action': ensemble_action,
                'weights': weights,
                'agent_actions': agent_actions,
                'confidences': confidences
            }
        else:
            return ensemble_action
    
    def collect_episode_trajectory(
        self,
        env,
        regime_detector=None,
        deterministic: bool = False
    ) -> Dict:
        """
        Collect full episode trajectory for hindsight learning.
        
        Args:
            env: Trading environment
            regime_detector: HMM regime detector (optional)
            deterministic: If True, use deterministic actions
            
        Returns:
            Episode trajectory dictionary
        """
        state = env.reset()
        done = False
        
        # Storage
        states = []
        agent_actions = []
        agent_returns = []
        agent_q_values = []
        rewards = []
        ensemble_actions = []
        
        # Track per-agent equity curves for calculating returns
        agent_equities = [[] for _ in range(self.config.n_agents)]
        initial_equity = env.equity
        
        while not done:
            # Get current regime if detector available
            regime = None
            if regime_detector is not None:
                regime_info = regime_detector.get_current_regime(
                    env.data.iloc[:env.episode_start + env.current_step + 1]
                )
                regime = regime_info['regime']
            
            # Store state
            states.append(state.copy())
            
            # Get actions from each agent
            actions = []
            q_vals = []
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            for i, agent in enumerate(self.agents):
                # Get action
                action = agent.select_action(state, deterministic=deterministic, regime=regime)
                actions.append(action)
                
                # Get Q-value
                with torch.no_grad():
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
                    
                    if i == 2 and hasattr(agent, 'critic1_low'):  # Agent 3
                        if regime == 'low_vol':
                            q1 = agent.critic1_low(state_tensor, action_tensor)
                            q2 = agent.critic2_low(state_tensor, action_tensor)
                        else:
                            q1 = agent.critic1_high(state_tensor, action_tensor)
                            q2 = agent.critic2_high(state_tensor, action_tensor)
                    else:
                        q1 = agent.critic1(state_tensor, action_tensor)
                        q2 = agent.critic2(state_tensor, action_tensor)
                    
                    q_value = torch.min(q1, q2).item()
                    q_vals.append(q_value)
            
            agent_actions.append(np.array(actions))
            agent_q_values.append(np.array(q_vals))
            
            # Get ensemble action
            ensemble_action = self.get_ensemble_action(
                state, deterministic=deterministic, regime=regime
            )
            ensemble_actions.append(ensemble_action)
            
            # Execute ensemble action
            next_state, reward, done, info = env.step(ensemble_action)
            rewards.append(reward)
            
            # Track equity for each agent (hypothetical)
            # Note: This is a simplification - ideally we'd run separate environments
            for i in range(self.config.n_agents):
                agent_equities[i].append(info['equity'])
            
            state = next_state
        
        # Calculate per-agent returns (simplified)
        # In practice, this would require running each agent independently
        for i in range(self.config.n_agents):
            agent_return = []
            for t in range(len(rewards)):
                # Use ensemble reward weighted by contribution
                agent_return.append(rewards[t])
            agent_returns.append(agent_return)
        
        agent_returns = np.array(agent_returns).T  # Shape: [T, n_agents]
        
        # Get final metrics
        final_metrics = env.get_episode_metrics()
        
        episode_data = {
            'states': np.array(states),
            'agent_actions': np.array(agent_actions),
            'agent_returns': agent_returns,
            'agent_q_values': np.array(agent_q_values),
            'rewards': np.array(rewards),
            'ensemble_actions': np.array(ensemble_actions),
            'final_return': final_metrics.get('total_return', 0.0),
            'final_sharpe': final_metrics.get('sharpe_ratio', 0.0),
            'final_equity': final_metrics.get('final_equity', initial_equity)
        }
        
        return episode_data
    
    def calculate_hindsight_optimal_weights(
        self,
        episode_data: Dict
    ) -> np.ndarray:
        """
        Calculate optimal weights in hindsight.
        
        Formula: optimal_weights[t] = softmax(-abs(agent_returns[t] - best_return[t]))
        
        Args:
            episode_data: Episode trajectory
            
        Returns:
            Optimal weights [T, n_agents]
        """
        agent_returns = episode_data['agent_returns']  # Shape: [T, n_agents]
        
        # Find best return at each timestep
        best_returns = np.max(agent_returns, axis=1, keepdims=True)  # Shape: [T, 1]
        
        # Calculate distances to best return
        distances = -np.abs(agent_returns - best_returns)  # Shape: [T, n_agents]
        
        # Apply softmax to get optimal weights
        # Temperature parameter for smoothness
        temperature = 1.0
        exp_distances = np.exp(distances / temperature)
        optimal_weights = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
        
        return optimal_weights
    
    def train_from_episodes(
        self,
        n_episodes: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Train meta-controller using hindsight learning on collected episodes.
        
        Args:
            n_episodes: Number of episodes to use (None = all available)
            verbose: If True, show training progress
        """
        # Get episodes
        episodes = self.episode_buffer.get_episodes(n_episodes or self.config.hindsight_episodes)
        
        if len(episodes) < 10:
            logger.warning(f"Only {len(episodes)} episodes available, need at least 10")
            return
        
        logger.info("="*80)
        logger.info("Training Meta-Controller with Hindsight Learning")
        logger.info("="*80)
        logger.info(f"Using {len(episodes)} episodes")
        
        # Prepare training data
        all_states = []
        all_confidences = []
        all_optimal_weights = []
        
        for episode in episodes:
            states = episode['states']
            agent_q_values = episode['agent_q_values']
            
            # Calculate optimal weights for this episode
            optimal_weights = self.calculate_hindsight_optimal_weights(episode)
            
            # Calculate confidences (based on Q-values)
            mean_q = np.mean(agent_q_values, axis=1, keepdims=True)
            confidences = -np.abs(agent_q_values - mean_q)
            
            all_states.append(states)
            all_confidences.append(confidences)
            all_optimal_weights.append(optimal_weights)
        
        # Concatenate all episodes
        all_states = np.concatenate(all_states, axis=0)
        all_confidences = np.concatenate(all_confidences, axis=0)
        all_optimal_weights = np.concatenate(all_optimal_weights, axis=0)
        
        logger.info(f"Total training samples: {len(all_states)}")
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(all_states).to(device)
        confidences_tensor = torch.FloatTensor(all_confidences).to(device)
        targets_tensor = torch.FloatTensor(all_optimal_weights).to(device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            states_tensor, confidences_tensor, targets_tensor
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Training loop
        self.network.train()
        
        best_loss = float('inf')
        patience = 0
        max_patience = 10
        
        progress_bar = tqdm(
            range(self.config.max_training_steps),
            desc="Training Meta-Controller",
            disable=not verbose
        )
        
        for step in progress_bar:
            epoch_losses = []
            
            for batch_states, batch_confidences, batch_targets in dataloader:
                # Forward pass
                predicted_weights = self.network(batch_states, batch_confidences)
                
                # Calculate loss (MSE)
                loss = F.mse_loss(predicted_weights, batch_targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            
            if verbose and step % 100 == 0:
                progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info(f"Early stopping at step {step}")
                    break
            
            self.training_step += 1
        
        self.is_trained = True
        
        logger.info(f"\nTraining complete! Final loss: {best_loss:.6f}")
        logger.info("="*80)
    
    def collect_and_train(
        self,
        env,
        n_episodes: int,
        regime_detector=None,
        verbose: bool = True
    ):
        """
        Collect episodes and train meta-controller.
        
        Args:
            env: Trading environment
            n_episodes: Number of episodes to collect
            regime_detector: HMM regime detector (optional)
            verbose: If True, show progress
        """
        logger.info("="*80)
        logger.info(f"Collecting {n_episodes} Episodes for Meta-Controller Training")
        logger.info("="*80)
        
        # Collect episodes
        for i in tqdm(range(n_episodes), desc="Collecting episodes", disable=not verbose):
            episode_data = self.collect_episode_trajectory(
                env, regime_detector, deterministic=False
            )
            self.episode_buffer.add_episode(episode_data)
            
            if verbose and (i + 1) % 10 == 0:
                logger.info(
                    f"Episode {i+1}/{n_episodes}: "
                    f"Return={episode_data['final_return']:.4f}, "
                    f"Sharpe={episode_data['final_sharpe']:.2f}"
                )
        
        # Train meta-controller
        logger.info("\nStarting meta-controller training...")
        self.train_from_episodes(n_episodes=n_episodes, verbose=verbose)
    
    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        regime_detector=None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            env: Trading environment
            n_episodes: Number of episodes
            regime_detector: HMM regime detector (optional)
            verbose: If True, show progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("="*80)
        logger.info(f"Evaluating Ensemble on {n_episodes} Episodes")
        logger.info("="*80)
        
        all_returns = []
        all_sharpes = []
        all_max_dds = []
        all_win_rates = []
        
        for i in tqdm(range(n_episodes), desc="Evaluating", disable=not verbose):
            episode_data = self.collect_episode_trajectory(
                env, regime_detector, deterministic=True
            )
            
            # Calculate metrics
            final_metrics = {
                'return': episode_data['final_return'],
                'sharpe': episode_data['final_sharpe']
            }
            
            all_returns.append(final_metrics['return'])
            all_sharpes.append(final_metrics['sharpe'])
            
            if verbose and (i + 1) % 5 == 0:
                logger.info(
                    f"Episode {i+1}: Return={final_metrics['return']:.4f}, "
                    f"Sharpe={final_metrics['sharpe']:.2f}"
                )
        
        # Aggregate statistics
        results = {
            'mean_return': np.mean(all_returns),
            'std_return': np.std(all_returns),
            'mean_sharpe': np.mean(all_sharpes),
            'std_sharpe': np.std(all_sharpes),
            'min_return': np.min(all_returns),
            'max_return': np.max(all_returns),
            'all_returns': all_returns,
            'all_sharpes': all_sharpes
        }
        
        logger.info("\n" + "="*80)
        logger.info("Evaluation Results")
        logger.info("="*80)
        logger.info(f"Mean Return: {results['mean_return']:.4f} ± {results['std_return']:.4f}")
        logger.info(f"Mean Sharpe: {results['mean_sharpe']:.2f} ± {results['std_sharpe']:.2f}")
        logger.info(f"Return Range: [{results['min_return']:.4f}, {results['max_return']:.4f}]")
        logger.info("="*80)
        
        return results
    

class EnsembleMetaController(MetaController):
    """
    Alias conservant la compatibilité avec les anciennes importations.
    """
    pass
    
    def save(self, filename: str = "meta_controller.pt") -> Path:
        """
        Save meta-controller.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.config.models_dir / filename
        
        checkpoint = {
            'config': self.config,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'is_trained': self.is_trained
        }
        
        torch.save(checkpoint, output_path)
        logger.info(f"Meta-controller saved to {output_path}")
        
        return output_path
    
    def load(self, filename: str = "meta_controller.pt"):
        """
        Load meta-controller.
        
        Args:
            filename: Input filename
        """
        input_path = self.config.models_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Meta-controller file not found: {input_path}")

        # PyTorch 2.6+ nécessite weights_only=False pour charger des classes personnalisées
        checkpoint = torch.load(input_path, map_location=device, weights_only=False)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.is_trained = checkpoint['is_trained']
        
        logger.info(f"Meta-controller loaded from {input_path}")
        logger.info(f"  Training steps: {self.training_step}")
        logger.info(f"  Is trained: {self.is_trained}")


def main():
    """Example usage of ensemble meta-controller."""
    from backend.data_pipeline import DataPipeline
    from backend.feature_engineering import FeaturePipeline
    from backend.trading_env import TradingEnvironment, TradingEnvConfig
    from backend.sac_agent import SACAgent, SACConfig
    from backend.hmm_detector import HMMRegimeDetector
    
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
    env = TradingEnvironment(
        data=train_data['EURUSD'],
        features=train_features,
        config=TradingEnvConfig(),
        eval_mode=False
    )
    
    # Create 3 SAC agents
    logger.info("\nCreating 3 SAC agents...")
    
    # Agent 1: Short-term
    config1 = SACConfig(
        state_dim=32,  # 30 + 2 regime features
        action_dim=1,
        gamma=0.93,
        hidden_dims=[256, 256]
    )
    agent1 = SACAgent(config=config1, agent_id=1)
    logger.info("Agent 1 (Short-term) created")
    
    # Agent 2: Medium-term
    config2 = SACConfig(
        state_dim=32,
        action_dim=1,
        gamma=0.95,
        hidden_dims=[256, 128]
    )
    agent2 = SACAgent(config=config2, agent_id=2)
    logger.info("Agent 2 (Medium-term) created")
    
    # Agent 3: Swing (with regime Q-functions)
    config3 = SACConfig(
        state_dim=30,  # No regime augmentation for Agent 3
        action_dim=1,
        gamma=0.97,
        hidden_dims=[128, 128],
        use_regime_qfuncs=True
    )
    agent3 = SACAgent(config=config3, agent_id=3)
    logger.info("Agent 3 (Swing Adaptive) created")
    
    # Load HMM detector (if available)
    try:
        hmm_detector = HMMRegimeDetector()
        hmm_detector.load("hmm_model.pkl")
        logger.info("HMM detector loaded")
    except:
        logger.warning("HMM detector not available, training without regime detection")
        hmm_detector = None
    
    # Create meta-controller
    logger.info("\nCreating meta-controller...")
    meta_controller = MetaController(
        agents=[agent1, agent2, agent3],
        config=MetaControllerConfig()
    )
    
    # Test ensemble action
    logger.info("\nTesting ensemble action...")
    state = env.reset()
    
    details = meta_controller.get_ensemble_action(
        state,
        deterministic=True,
        regime='low_vol' if hmm_detector else None,
        return_details=True
    )
    
    print("\n" + "="*80)
    print("Ensemble Action Details")
    print("="*80)
    print(f"Ensemble Action: {details['action']}")
    print(f"Agent Weights: {details['weights']}")
    print(f"Agent Actions: {details['agent_actions']}")
    print(f"Agent Confidences: {details['confidences']}")
    
    # Collect episodes and train
    logger.info("\n\nCollecting episodes for meta-controller training...")
    meta_controller.collect_and_train(
        env=env,
        n_episodes=10,  # Small number for demo
        regime_detector=hmm_detector,
        verbose=True
    )
    
    # Evaluate
    logger.info("\nEvaluating ensemble...")
    results = meta_controller.evaluate(
        env=env,
        n_episodes=5,
        regime_detector=hmm_detector,
        verbose=True
    )
    
    # Save meta-controller
    logger.info("\nSaving meta-controller...")
    meta_controller.save("meta_controller_demo.pt")
    
    logger.info("\n" + "="*80)
    logger.info("Ensemble Meta-Controller Demo Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
