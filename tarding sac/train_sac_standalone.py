"""
Script d'entraînement SAC Standalone pour Kaggle
=================================================

Ce script est 100% autonome et ne nécessite aucun fichier backend externe.
Il contient toute l'implémentation SAC, l'environnement de trading, et le code d'entraînement.

Usage sur Kaggle:
    1. Uploadez votre fichier .h5 contenant les données et features
    2. Uploadez ce script
    3. Exécutez: python train_sac_standalone.py --h5-path /kaggle/input/your-data/data.h5

Le fichier h5 doit contenir:
    - /train/EURUSD/: timestamp, open, high, low, close
    - /train/features/: 30 colonnes de features normalisées
    - /val/EURUSD/ et /val/features/ (idem pour validation)
    - /test/EURUSD/ et /test/features/ (optionnel)

Auteur: Trading SAC System
Date: 2025-11-23
"""

import argparse
import json
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SACConfig:
    """Configuration SAC Agent"""
    state_dim: int = 30
    action_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-5
    gamma: float = 0.95
    tau: float = 0.005

    buffer_capacity: int = 100000
    batch_size: int = 1024
    warmup_steps: int = 5000

    actor_dropout: float = 0.05
    critic_dropout: float = 0.1

    auto_entropy_tuning: bool = True
    target_entropy: float = -1.0
    min_alpha: float = 0.01

    max_grad_norm: float = 2.0


@dataclass
class TradingEnvConfig:
    """Configuration de l'environnement de trading"""
    initial_capital: float = 500000.0
    risk_per_trade: float = 0.0005
    max_leverage: float = 2.0
    min_position_size: float = 0.01

    sl_atr_multiplier: float = 3.0
    tp_atr_multiplier: float = 6.0

    base_spread: float = 0.1
    slippage_baseline: float = 0.1
    market_impact_base: float = 0.1

    episode_length: int = 3000
    no_trading_warmup_steps: int = 5000

    dense_weight: float = 0.70
    terminal_weight: float = 0.30
    use_simple_reward: bool = True

    n_features: int = 30
    obs_min: float = -10.0
    obs_max: float = 10.0


# =============================================================================
# RÉSEAUX DE NEURONES (SAC)
# =============================================================================

class SpectralNormLinear(nn.Module):
    """Linear layer avec Spectral Normalization"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Actor(nn.Module):
    """Réseau Actor avec Gaussian policy"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], dropout: float = 0.05):
        super().__init__()
        self.action_dim = action_dim

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(SpectralNormLinear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)

            # Log probability avec correction tanh
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """Réseau Critic (Q-function)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()

        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(SpectralNormLinear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Replay buffer avec stratification"""
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.choice(self.size, batch_size, replace=False)

        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }


# =============================================================================
# ENVIRONNEMENT DE TRADING
# =============================================================================

class TradingEnvironment:
    """Environnement de trading EUR/USD"""
    def __init__(self, data: pd.DataFrame, features: pd.DataFrame, config: TradingEnvConfig, eval_mode: bool = False):
        self.data = data
        self.features = features
        self.config = config
        self.eval_mode = eval_mode

        self.n_steps = len(data)
        self.current_step = 0

        # État du trading
        self.equity = config.initial_capital
        self.initial_capital = config.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0

        # Historique
        self.equity_history = []
        self.returns_buffer = deque(maxlen=100)

        # Statistiques
        self.total_steps = 0
        self.episode_steps = 0
        self.n_trades = 0

    def reset(self) -> np.ndarray:
        """Reset l'environnement"""
        if self.eval_mode:
            # Mode séquentiel pour évaluation
            self.current_step = 0
        else:
            # Mode aléatoire pour exploration
            max_start = self.n_steps - self.config.episode_length - 1
            self.current_step = np.random.randint(0, max(1, max_start))

        self.equity = self.config.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0

        self.equity_history = [self.equity]
        self.returns_buffer.clear()
        self.episode_steps = 0
        self.n_trades = 0

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Récupère l'observation (features)"""
        obs = self.features.iloc[self.current_step].values
        obs = np.clip(obs, self.config.obs_min, self.config.obs_max)
        return obs.astype(np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Exécute une action dans l'environnement"""
        action = np.clip(action, -1.0, 1.0)

        # Pendant le warmup, forcer action à 0
        if self.total_steps < self.config.no_trading_warmup_steps:
            action = 0.0

        # Récupérer les données de marché
        current_price = self.data.iloc[self.current_step]['close']
        atr = self._calculate_atr()

        # Calculer la nouvelle position
        position_size, sl_price, tp_price = self._calculate_position_size(
            self.equity, current_price, atr, action
        )

        # Exécuter le trade
        old_equity = self.equity
        self._execute_trade(position_size, current_price, sl_price, tp_price, atr)

        # Avancer le temps
        self.current_step += 1
        self.episode_steps += 1
        self.total_steps += 1

        # Calculer la récompense
        log_return = np.log(self.equity / old_equity) if old_equity > 0 else 0.0
        self.returns_buffer.append(log_return)

        if self.config.use_simple_reward:
            reward = log_return * 100.0
        else:
            reward = log_return * 100.0

        # Vérifier si l'épisode est terminé
        done = (
            self.current_step >= self.n_steps - 1 or
            self.episode_steps >= self.config.episode_length or
            self.equity <= self.config.initial_capital * 0.05  # Perte de 95%
        )

        # Récompense terminale
        if done:
            terminal_reward = self._calculate_terminal_reward()
            reward += terminal_reward * self.config.terminal_weight

        # Informations
        info = {
            'equity': self.equity,
            'position': self.position,
            'total_return': (self.equity - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': self._calculate_sharpe(),
            'max_drawdown': self._calculate_max_drawdown(),
            'n_trades': self.n_trades
        }

        obs = self._get_observation() if not done else np.zeros(self.config.n_features, dtype=np.float32)

        return obs, reward, done, info

    def _calculate_atr(self) -> float:
        """Calcule l'ATR (Average True Range)"""
        window = 14
        start_idx = max(0, self.current_step - window)
        end_idx = self.current_step + 1

        data_window = self.data.iloc[start_idx:end_idx]

        if len(data_window) < 2:
            return 0.001  # Valeur par défaut

        high_low = data_window['high'] - data_window['low']
        high_close = np.abs(data_window['high'] - data_window['close'].shift())
        low_close = np.abs(data_window['low'] - data_window['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.mean()

        return max(atr, 0.0001)

    def _calculate_position_size(self, equity: float, price: float, atr: float, action: float) -> Tuple[float, float, float]:
        """Calcule la taille de position basée sur le risque"""
        sl_distance = self.config.sl_atr_multiplier * atr
        risk_dollars = self.config.risk_per_trade * equity

        sl_distance_pips = sl_distance / 0.0001
        pip_value = 10.0

        position_size = risk_dollars / (sl_distance_pips * pip_value)
        position_final = action * position_size

        # Contraintes
        if abs(position_final) < self.config.min_position_size:
            position_final = 0.0 if abs(action) <= 0.01 else np.sign(action) * self.config.min_position_size

        position_value = abs(position_final) * 100000
        leverage = position_value / equity
        if leverage > self.config.max_leverage:
            position_final = np.sign(position_final) * (equity * self.config.max_leverage / 100000)

        # Calculer SL et TP
        if position_final > 0:
            sl_price = price - sl_distance
            tp_price = price + (self.config.tp_atr_multiplier * atr)
        elif position_final < 0:
            sl_price = price + sl_distance
            tp_price = price - (self.config.tp_atr_multiplier * atr)
        else:
            sl_price = 0.0
            tp_price = 0.0

        return position_final, sl_price, tp_price

    def _execute_trade(self, new_position: float, current_price: float, sl_price: float, tp_price: float, atr: float):
        """Exécute un trade"""
        # Vérifier SL/TP de la position actuelle
        if self.position != 0:
            if self.position > 0 and (current_price <= self.sl_price or current_price >= self.tp_price):
                # Clôturer long
                pnl = (current_price - self.entry_price) * self.position * 100000
                self.equity += pnl
                self.position = 0.0
                self.n_trades += 1
            elif self.position < 0 and (current_price >= self.sl_price or current_price <= self.tp_price):
                # Clôturer short
                pnl = (self.entry_price - current_price) * abs(self.position) * 100000
                self.equity += pnl
                self.position = 0.0
                self.n_trades += 1

        # Nouvelle position
        if abs(new_position - self.position) > 0.001:
            # Clôturer position existante si changement de direction
            if self.position != 0 and np.sign(new_position) != np.sign(self.position):
                if self.position > 0:
                    pnl = (current_price - self.entry_price) * self.position * 100000
                else:
                    pnl = (self.entry_price - current_price) * abs(self.position) * 100000
                self.equity += pnl
                self.position = 0.0
                self.n_trades += 1

            # Ouvrir nouvelle position
            if abs(new_position) > 0.001:
                self.position = new_position
                self.entry_price = current_price
                self.sl_price = sl_price
                self.tp_price = tp_price

        self.equity_history.append(self.equity)

    def _calculate_sharpe(self) -> float:
        """Calcule le Sharpe ratio"""
        if len(self.returns_buffer) < 2:
            return 0.0
        returns = np.array(self.returns_buffer)
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return < 1e-6:
            return 0.0
        return (mean_return / std_return) * np.sqrt(252 * 288)  # Annualisé

    def _calculate_max_drawdown(self) -> float:
        """Calcule le drawdown maximum"""
        if len(self.equity_history) < 2:
            return 0.0
        equity_curve = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_terminal_reward(self) -> float:
        """Calcule la récompense terminale"""
        sharpe = self._calculate_sharpe()
        total_return = (self.equity - self.initial_capital) / self.initial_capital
        max_dd = self._calculate_max_drawdown()

        reward = (
            0.5 * np.clip(sharpe / 2.0, -1, 1) +
            0.3 * np.clip(total_return * 10, -1, 1) +
            0.2 * (1.0 - np.clip(max_dd * 5, 0, 1))
        )

        return reward


# =============================================================================
# AGENT SAC
# =============================================================================

class SACAgent:
    """Agent SAC"""
    def __init__(self, config: SACConfig, device: str = 'auto'):
        self.config = config

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Réseaux
        self.actor = Actor(config.state_dim, config.action_dim, config.hidden_dims, config.actor_dropout).to(self.device)
        self.critic1 = Critic(config.state_dim, config.action_dim, config.hidden_dims, config.critic_dropout).to(self.device)
        self.critic2 = Critic(config.state_dim, config.action_dim, config.hidden_dims, config.critic_dropout).to(self.device)

        self.critic1_target = Critic(config.state_dim, config.action_dim, config.hidden_dims, config.critic_dropout).to(self.device)
        self.critic2_target = Critic(config.state_dim, config.action_dim, config.hidden_dims, config.critic_dropout).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.critic_lr)

        # Entropie
        if config.auto_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.target_entropy = config.target_entropy
        else:
            self.log_alpha = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_capacity, config.state_dim, config.action_dim)

        # Compteurs
        self.total_steps = 0
        self.episode_count = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> float:
        """Sélectionne une action"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic=evaluate)

        return action.cpu().numpy()[0, 0]

    def update(self) -> Optional[Dict]:
        """Met à jour l'agent"""
        if self.replay_buffer.size < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)

        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)

        # Alpha
        if self.config.auto_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = 0.2

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_target = rewards + self.config.gamma * (1 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.max_grad_norm)
        self.critic2_optimizer.step()

        # Actor update
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # Alpha update
        if self.config.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha = max(self.log_alpha.exp().item(), self.config.min_alpha)

        # Soft update targets
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': (critic1_loss + critic2_loss).item() / 2,
            'alpha': alpha if isinstance(alpha, float) else alpha.item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update des target networks"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def save(self, path: str):
        """Sauvegarde l'agent"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha.item() if self.log_alpha is not None else None,
            'config': self.config,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
        }, path)
        print(f"✅ Modèle sauvegardé: {path}")

    def load(self, path: str):
        """Charge l'agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        if checkpoint['log_alpha'] is not None:
            self.log_alpha.data = torch.tensor([checkpoint['log_alpha']], device=self.device)
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        print(f"✅ Modèle chargé: {path}")


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_data_from_h5(h5_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les données depuis le fichier h5

    Structure attendue:
        /train/EURUSD/ : timestamp, open, high, low, close
        /train/features/ : 30 colonnes de features
        /val/EURUSD/ : idem
        /val/features/ : idem
    """
    print(f"\n📊 Chargement des données depuis {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Train data
        train_timestamps = f['/train/EURUSD/timestamp'][:]
        train_ohlc = pd.DataFrame({
            'timestamp': train_timestamps,
            'open': f['/train/EURUSD/open'][:],
            'high': f['/train/EURUSD/high'][:],
            'low': f['/train/EURUSD/low'][:],
            'close': f['/train/EURUSD/close'][:]
        })

        # Train features
        feature_names = [key for key in f['/train/features/'].keys()]
        train_features = pd.DataFrame({
            name: f[f'/train/features/{name}'][:] for name in feature_names
        })

        # Val data
        val_timestamps = f['/val/EURUSD/timestamp'][:]
        val_ohlc = pd.DataFrame({
            'timestamp': val_timestamps,
            'open': f['/val/EURUSD/open'][:],
            'high': f['/val/EURUSD/high'][:],
            'low': f['/val/EURUSD/low'][:],
            'close': f['/val/EURUSD/close'][:]
        })

        # Val features
        val_features = pd.DataFrame({
            name: f[f'/val/features/{name}'][:] for name in feature_names
        })

        # Test data (optionnel)
        if '/test/EURUSD/timestamp' in f:
            test_timestamps = f['/test/EURUSD/timestamp'][:]
            test_ohlc = pd.DataFrame({
                'timestamp': test_timestamps,
                'open': f['/test/EURUSD/open'][:],
                'high': f['/test/EURUSD/high'][:],
                'low': f['/test/EURUSD/low'][:],
                'close': f['/test/EURUSD/close'][:]
            })
            test_features = pd.DataFrame({
                name: f[f'/test/features/{name}'][:] for name in feature_names
            })
        else:
            test_ohlc = val_ohlc.copy()
            test_features = val_features.copy()

    print(f"✅ Données chargées:")
    print(f"   Train: {len(train_ohlc)} candles, {train_features.shape[1]} features")
    print(f"   Val:   {len(val_ohlc)} candles")
    print(f"   Test:  {len(test_ohlc)} candles")

    return train_ohlc, train_features, val_ohlc, val_features, test_ohlc, test_features


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

class Trainer:
    """Trainer pour l'entraînement SAC"""
    def __init__(
        self,
        train_data: pd.DataFrame,
        train_features: pd.DataFrame,
        val_data: pd.DataFrame,
        val_features: pd.DataFrame,
        output_dir: str = "./output",
        num_episodes: int = 100,
        eval_frequency: int = 10,
        checkpoint_frequency: int = 5,
        device: str = 'auto'
    ):
        self.train_data = train_data
        self.train_features = train_features
        self.val_data = val_data
        self.val_features = val_features
        self.output_dir = Path(output_dir)
        self.num_episodes = num_episodes
        self.eval_frequency = eval_frequency
        self.checkpoint_frequency = checkpoint_frequency

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Configuration
        self.sac_config = SACConfig()
        self.env_config = TradingEnvConfig()

        # Environnements
        self.train_env = TradingEnvironment(train_data, train_features, self.env_config, eval_mode=False)
        self.val_env = TradingEnvironment(val_data, val_features, self.env_config, eval_mode=True)

        # Agent
        self.agent = SACAgent(self.sac_config, device=device)

        # Stats
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_values': []
        }

        print(f"\n🤖 Agent SAC créé:")
        print(f"   Device: {self.agent.device}")
        print(f"   State dim: {self.sac_config.state_dim}")
        print(f"   Warmup steps: {self.sac_config.warmup_steps}")

    def train_episode(self, episode: int) -> Dict:
        """Entraîne un épisode"""
        state = self.train_env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        losses = {'actor': [], 'critic': [], 'alpha': []}

        while not done:
            # Sélection action
            if self.agent.total_steps < self.sac_config.warmup_steps:
                action = np.random.uniform(-1, 1)
            else:
                action = self.agent.select_action(state, evaluate=False)

            # Step
            next_state, reward, done, info = self.train_env.step(action)

            # Stocker
            self.agent.replay_buffer.push(state, [action], reward, next_state, done)

            # Update
            if self.agent.total_steps >= self.sac_config.warmup_steps:
                loss = self.agent.update()
                if loss:
                    losses['actor'].append(loss['actor_loss'])
                    losses['critic'].append(loss['critic_loss'])
                    losses['alpha'].append(loss['alpha'])

            episode_reward += reward
            episode_length += 1
            state = next_state
            self.agent.total_steps += 1

        self.agent.episode_count += 1

        return {
            'reward': episode_reward,
            'length': episode_length,
            'actor_loss': np.mean(losses['actor']) if losses['actor'] else 0,
            'critic_loss': np.mean(losses['critic']) if losses['critic'] else 0,
            'alpha': np.mean(losses['alpha']) if losses['alpha'] else 0,
            'total_return': info['total_return'],
            'sharpe': info['sharpe_ratio'],
            'max_drawdown': info['max_drawdown']
        }

    def evaluate(self, num_episodes: int = 3) -> Dict:
        """Évalue l'agent"""
        eval_rewards = []
        eval_returns = []
        eval_sharpes = []

        for _ in range(num_episodes):
            state = self.val_env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, info = self.val_env.step(action)
                episode_reward += reward
                state = next_state

            eval_rewards.append(episode_reward)
            eval_returns.append(info['total_return'])
            eval_sharpes.append(info['sharpe_ratio'])

        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_return': np.mean(eval_returns),
            'mean_sharpe': np.mean(eval_sharpes)
        }

    def run(self):
        """Lance l'entraînement"""
        print("\n" + "="*70)
        print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT SAC")
        print("="*70)

        best_eval_reward = -float('inf')

        for episode in range(1, self.num_episodes + 1):
            start_time = time.time()

            # Entraîner
            metrics = self.train_episode(episode)

            # Logger
            self.stats['episode_rewards'].append(metrics['reward'])
            self.stats['episode_lengths'].append(metrics['length'])
            self.stats['actor_losses'].append(metrics['actor_loss'])
            self.stats['critic_losses'].append(metrics['critic_loss'])
            self.stats['alpha_values'].append(metrics['alpha'])

            elapsed = time.time() - start_time

            # Afficher
            print(f"\n📈 Episode {episode}/{self.num_episodes}")
            print(f"   Reward: {metrics['reward']:.2f}")
            print(f"   Return: {metrics['total_return']:.2%}")
            print(f"   Sharpe: {metrics['sharpe']:.2f}")
            print(f"   MaxDD: {metrics['max_drawdown']:.2%}")
            print(f"   Actor Loss: {metrics['actor_loss']:.4f}")
            print(f"   Alpha: {metrics['alpha']:.4f}")
            print(f"   Steps: {self.agent.total_steps} | Time: {elapsed:.1f}s")

            # Évaluer
            if episode % self.eval_frequency == 0:
                print(f"\n🔍 Évaluation...")
                eval_metrics = self.evaluate()
                self.stats['eval_rewards'].append(eval_metrics['mean_reward'])

                print(f"   Eval Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"   Eval Return: {eval_metrics['mean_return']:.2%}")
                print(f"   Eval Sharpe: {eval_metrics['mean_sharpe']:.2f}")

                if eval_metrics['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['mean_reward']
                    self.agent.save(str(self.output_dir / "agent_best.pt"))
                    print(f"   ⭐ Nouveau meilleur modèle!")

            # Checkpoint
            if episode % self.checkpoint_frequency == 0:
                self.agent.save(str(self.output_dir / f"checkpoints/agent_ep{episode}.pt"))

        # Sauvegarder final
        print("\n" + "="*70)
        print("💾 SAUVEGARDE DU MODÈLE FINAL")
        print("="*70)

        self.agent.save(str(self.output_dir / "agent_final.pt"))

        # Sauvegarder stats
        with open(self.output_dir / "logs/training_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)

        np.savez(
            self.output_dir / "logs/training_stats.npz",
            episode_rewards=np.array(self.stats['episode_rewards']),
            episode_lengths=np.array(self.stats['episode_lengths']),
            eval_rewards=np.array(self.stats['eval_rewards']),
            actor_losses=np.array(self.stats['actor_losses']),
            critic_losses=np.array(self.stats['critic_losses']),
            alpha_values=np.array(self.stats['alpha_values'])
        )

        print(f"\n✅ Statistiques sauvegardées: {self.output_dir / 'logs'}")

        print("\n" + "="*70)
        print("🎉 ENTRAÎNEMENT TERMINÉ")
        print("="*70)
        print(f"   Meilleure récompense eval: {best_eval_reward:.2f}")
        print(f"   Récompense moyenne (10 derniers): {np.mean(self.stats['episode_rewards'][-10:]):.2f}")
        print(f"   Fichiers dans: {self.output_dir}")
        print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entraînement SAC standalone pour Kaggle")
    parser.add_argument('--h5-path', type=str, required=True, help="Chemin vers le fichier h5")
    parser.add_argument('--output-dir', type=str, default="/kaggle/working", help="Répertoire de sortie")
    parser.add_argument('--num-episodes', type=int, default=100, help="Nombre d'épisodes")
    parser.add_argument('--eval-frequency', type=int, default=10, help="Fréquence d'évaluation")
    parser.add_argument('--checkpoint-frequency', type=int, default=5, help="Fréquence de checkpoint")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help="Device")

    args = parser.parse_args()

    # Charger les données
    train_data, train_features, val_data, val_features, test_data, test_features = load_data_from_h5(args.h5_path)

    # Créer le trainer
    trainer = Trainer(
        train_data=train_data,
        train_features=train_features,
        val_data=val_data,
        val_features=val_features,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        eval_frequency=args.eval_frequency,
        checkpoint_frequency=args.checkpoint_frequency,
        device=args.device
    )

    # Lancer l'entraînement
    trainer.run()

    print("\n✅ Script terminé avec succès!")


if __name__ == "__main__":
    main()
