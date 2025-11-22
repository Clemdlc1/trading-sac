"""
SAC EUR/USD Trading System - Trading Environment
=================================================

This module implements a Gym-compatible trading environment for EUR/USD.

Features:
- Continuous action space [-1, 1] for position sizing
- Risk-based position sizing (2% per trade)
- Dynamic Stop-Loss (2×ATR) and Take-Profit (4×ATR)
- Realistic transaction cost model (2.5-4.5 bps)
- Margin call detection (equity < 20% initial)
- Variable episode lengths [1000, 3000, 6000]
- Dense + terminal reward function
- Complete metrics tracking

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""
    
    # Initial capital - MASSIVE account for resilience
    initial_capital: float = 500000.0  # $500k (5× more resilient!)

    # Position sizing - ULTRA CONSERVATIVE: 2× leverage, 0.05% risk
    risk_per_trade: float = 0.0005  # 0.05% per trade (4× safer than before!)
    max_leverage: float = 2.0  # 2× maximum - extremely safe
    min_position_size: float = 0.01  # Minimum lot size
    
    # Stop-Loss and Take-Profit
    sl_atr_multiplier: float = 2.0  # SL = 2×ATR
    tp_atr_multiplier: float = 4.0  # TP = 4×ATR (2:1 risk/reward)
    
    # Transaction costs (basis points)
    base_spread: float = 0.5  # 0.5 bps base spread
    slippage_baseline: float = 1.5  # 1.5 bps baseline slippage
    market_impact_base: float = 1.0  # 1.0 bps market impact
    
    # Risk management
    margin_call_threshold: float = 0.20  # 20% of initial capital
    max_drawdown_threshold: float = 0.20  # 20% max drawdown
    
    # Episode parameters - FIXED LENGTH for consistent rewards
    episode_lengths: List[int] = field(default_factory=lambda: [3000])  # FIXED to 3000 only
    episode_probs: List[float] = field(default_factory=lambda: [1.0])  # 100% probability

    # Learning phases
    # REDUCED from 20000 to 5000 to match SAC warmup_steps and reduce bias from forced actions
    # During warmup: actions forced to 0, buffer fills, but networks don't update
    no_trading_warmup_steps: int = 5000  # First 5k steps: ZERO trading, buffer filling phase
    
    # Reward function parameters
    dense_weight: float = 0.70  # INCREASED from 0.40 - more immediate feedback
    terminal_weight: float = 0.30  # DECREASED from 0.60 - less delayed signal

    # Reward scaling to improve learning signal
    reward_scale: float = 1.0  # Scale up rewards for better gradients

    # DSR parameters (Differential Sharpe Ratio)
    dsr_eta: float = 0.01  # INCREASED from 0.001 for faster adaptation
    
    # Observation space
    n_features: int = 30
    obs_min: float = -10.0
    obs_max: float = 10.0
    
    # Action space
    action_min: float = -1.0
    action_max: float = 1.0


class TransactionCostModel:
    """Realistic transaction cost model."""
    
    def __init__(self, config: TradingEnvConfig):
        self.config = config
    
    def calculate_cost(
        self, 
        position_change: float,
        hour: int,
        volatility: float,
        position_size: float
    ) -> float:
        """
        Calculate total transaction cost.
        
        Components:
        1. Spread: 0.5-1.0 bps (varies by hour)
        2. Slippage: 1-2 bps baseline
        3. Market impact: 0.5-1.5 bps (depends on size)
        
        Total: 2.5-4.5 bps per direction
        
        Args:
            position_change: Absolute position change
            hour: Hour of day (0-23)
            volatility: Current volatility (ATR)
            position_size: Size of position
            
        Returns:
            Total cost in basis points
        """
        if position_change < 1e-6:
            return 0.0
        
        # 1. Spread (varies by session)
        # Lower during London/NY overlap (13:00-16:00 UTC)
        if 13 <= hour < 16:
            spread = self.config.base_spread
        # Higher during Asian session
        elif hour >= 23 or hour < 7:
            spread = self.config.base_spread * 1.5
        else:
            spread = self.config.base_spread * 1.2
        
        # 2. Slippage (increases with volatility)
        slippage = self.config.slippage_baseline * (1.0 + volatility)
        
        # 3. Market impact (increases with position size)
        # Normalize position size to [0, 1] range
        size_factor = min(position_size / 100000.0, 1.0)  # Relative to $100k
        market_impact = self.config.market_impact_base * (0.5 + size_factor)
        
        # Total cost
        total_cost = spread + slippage + market_impact
        
        return total_cost
    
    def cost_in_dollars(
        self,
        position_change: float,
        entry_price: float,
        hour: int,
        volatility: float
    ) -> float:
        """
        Convert cost from basis points to dollars.
        
        Args:
            position_change: Position change in lots
            entry_price: Entry price
            hour: Hour of day
            volatility: Current volatility
            
        Returns:
            Cost in dollars
        """
        # Calculate cost in bps
        cost_bps = self.calculate_cost(
            position_change,
            hour,
            volatility,
            position_change * 100000  # Convert lots to notional
        )
        
        # Convert to dollars
        # 1 lot = 100,000 EUR
        # Cost per pip = position_size * pip_value
        # Pip value ≈ 10 USD per lot for EUR/USD
        cost_dollars = (cost_bps / 10000.0) * position_change * 100000 * entry_price
        
        return cost_dollars


class PositionSizer:
    """Risk-based position sizing calculator."""
    
    def __init__(self, config: TradingEnvConfig):
        self.config = config
    
    def calculate_position_size(
        self,
        equity: float,
        current_price: float,
        atr: float,
        action: float
    ) -> Tuple[float, float, float]:
        """
        Calculate position size based on risk management.
        
        Formula:
        1. SL_distance = 2 × ATR
        2. Risk_dollars = 2% × equity
        3. Position_size = Risk_dollars / (SL_distance_pips × pip_value)
        4. Position_final = action × Position_size
        
        Args:
            equity: Current equity
            current_price: Current EUR/USD price
            atr: Current ATR value
            action: Agent action [-1, 1]
            
        Returns:
            Tuple of (position_lots, sl_price, tp_price)
        """
        # Calculate stop-loss distance
        sl_distance = self.config.sl_atr_multiplier * atr
        
        # Calculate risk in dollars
        risk_dollars = self.config.risk_per_trade * equity
        
        # Calculate position size
        # For EUR/USD, 1 pip = 0.0001
        # Pip value ≈ 10 USD per standard lot
        sl_distance_pips = sl_distance / 0.0001
        pip_value = 10.0  # USD per pip per lot
        
        position_size = risk_dollars / (sl_distance_pips * pip_value)
        
        # Apply action multiplier
        position_final = action * position_size
        
        # Apply constraints
        # 1. Minimum position size
        if abs(position_final) < self.config.min_position_size:
            if abs(action) > 0.01:  # Small threshold to avoid noise
                position_final = np.sign(action) * self.config.min_position_size
            else:
                position_final = 0.0
        
        # 2. Leverage constraint
        position_value = abs(position_final) * 100000  # Notional value
        leverage = position_value / equity
        if leverage > self.config.max_leverage:
            position_final = np.sign(position_final) * (equity * self.config.max_leverage / 100000)
        
        # Calculate SL and TP prices
        if position_final > 0:  # Long position
            sl_price = current_price - sl_distance
            tp_price = current_price + (self.config.tp_atr_multiplier * atr)
        elif position_final < 0:  # Short position
            sl_price = current_price + sl_distance
            tp_price = current_price - (self.config.tp_atr_multiplier * atr)
        else:  # Flat
            sl_price = 0.0
            tp_price = 0.0
        
        return position_final, sl_price, tp_price


class RewardCalculator:
    """Calculate dense and terminal rewards."""
    
    def __init__(self, config: TradingEnvConfig):
        self.config = config
        
        # DSR state variables
        self.A = 0.0  # Average return
        self.B = 0.0  # Average squared return
    
    def reset(self):
        """Reset DSR state variables."""
        self.A = 0.0
        self.B = 0.0
    
    def calculate_dense_reward(
        self,
        equity_t: float,
        equity_t_minus_1: float,
        initial_capital: float,
        returns_buffer: List[float],
        position_change: float,
        transaction_cost: float,
        n_trades_today: int
    ) -> float:
        """
        Calculate dense reward (per step).

        Formula:
        dense_reward = 0.15 × R_return + 0.15 × R_dsr + 0.08 × R_downside
                     + 0.07 × R_cost + 0.05 × R_position + 0.50 × R_survival
        
        Args:
            equity_t: Current equity
            equity_t_minus_1: Previous equity
            initial_capital: Initial capital
            returns_buffer: Recent returns
            position_change: Position change magnitude
            transaction_cost: Transaction cost in dollars
            n_trades_today: Number of trades in last 288 steps
            
        Returns:
            Dense reward value
        """
        # Component 1: Returns (30%)
        r_t = (equity_t - equity_t_minus_1) / initial_capital
        R_return = r_t
        
        # Component 2: Simplified Sharpe-like component (30%)
        # Simplified from DSR to avoid numerical instabilities
        eta = self.config.dsr_eta
        self.A = (1 - eta) * self.A + eta * r_t
        self.B = (1 - eta) * self.B + eta * (r_t ** 2)

        # Simplified Sharpe: excess return / volatility
        mean_return = self.A
        variance = max(self.B - self.A ** 2, 1e-8)  # Prevent negative variance
        std = np.sqrt(variance)

        # Sharpe-like reward: reward excess returns relative to volatility
        if std > 1e-6:
            R_dsr = mean_return / (std + 1e-6)
        else:
            R_dsr = 0.0

        # Gentler clipping to preserve signal strength
        R_dsr = np.clip(R_dsr, -5.0, 5.0)
        
        # Component 3: Downside Penalty (20%)
        recent_returns = returns_buffer[-50:] if len(returns_buffer) >= 50 else returns_buffer
        negative_returns = [r for r in recent_returns if r < 0]

        if len(negative_returns) > 0:
            R_downside = -np.std(negative_returns) * 3.0  # REDUCED from 10.0 to 3.0
        else:
            R_downside = 0.0
        
        # Component 4: Transaction Costs (15%)
        # Always penalize costs proportionally (removed threshold to simplify)
        R_cost = -transaction_cost / initial_capital
        
        # Component 5: Position Management (5%)
        # STRONG penalty for large positions to prevent account blowup
        # Encourage conservative trading during learning
        if n_trades_today > 20:  # INCREASED threshold from 10 to 20
            R_position = -(n_trades_today - 20) ** 2 * 0.0001  # Overtrading penalty
        else:
            # Penalty proportional to position size to encourage small positions
            R_position = -position_change * 0.01  # INCREASED from 0.0005 - discourage large trades!
        
        # Component 6: ULTRA-HIGH Survival bonus to DOMINATE trading losses
        # GOAL: Make surviving longer ALWAYS better, even with terrible trading
        #
        # User requirement:
        # - 200 steps with -100 total
        # - 450 steps should be -80 total (BETTER!)
        #
        # NO SURVIVAL BONUS - Agent must learn profitable trading, not just survival
        # Focus entirely on trading performance metrics
        dense_reward = (
            0.40 * R_return +      # 40% - basic returns (MAIN OBJECTIVE)
            0.40 * R_dsr +         # 40% - risk-adjusted returns (QUALITY)
            0.10 * R_downside +    # 10% - downside protection
            0.05 * R_cost +        # 5% - cost control
            0.05 * R_position      # 5% - position management
        )

        return dense_reward
    
    def calculate_terminal_reward(
        self,
        returns_episode: List[float],
        equity_curve: List[float],
        trades: List[Dict]
    ) -> float:
        """
        Calculate terminal reward (end of episode).
        
        Formula:
        terminal_reward = 0.35 × R_sortino + 0.25 × R_calmar 
                        + 0.25 × R_drawdown + 0.15 × R_expectancy
        
        Args:
            returns_episode: All returns during episode
            equity_curve: Equity at each step
            trades: List of completed trades
            
        Returns:
            Terminal reward value
        """
        if len(returns_episode) < 10:
            return -1.0  # Penalty for too short episode
        
        returns_array = np.array(returns_episode)
        
        # Component 1: Sortino Ratio (35%)
        mean_return = np.mean(returns_array)
        downside_returns = returns_array[returns_array < 0]

        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = mean_return / (downside_std + 1e-8)
            # ADJUSTED: Center at 0.5 (more realistic), normalize to [-2, +2]
            R_sortino = np.clip((sortino - 0.5) / 0.75, -2.0, 2.0)
        else:
            R_sortino = 2.0  # Perfect score if no downside
        
        # Component 2: Calmar Ratio (25%)
        annual_return = mean_return * 252 * 288  # Annualize
        max_dd = self._calculate_max_drawdown(equity_curve)

        if max_dd > 1e-6:
            calmar = annual_return / max_dd
            # ADJUSTED: Center at 1.0 (realistic target), normalize to [-2, +2]
            R_calmar = np.clip((calmar - 1.0) / 1.0, -2.0, 2.0)
        else:
            # No drawdown but positive return = excellent
            R_calmar = 2.0 if annual_return > 0 else 0.0
        
        # Component 3: Drawdown Penalty (25%)
        # ADJUSTED: Gentler penalties, normalize to [-2, 0]
        if max_dd < 0.05:
            R_drawdown = 0.0
        elif max_dd < 0.15:
            R_drawdown = -max_dd * 10.0  # Linear: -0.5 at 5%, -1.5 at 15%
        else:
            # Quadratic penalty for severe drawdowns
            R_drawdown = np.clip(-0.15 * (max_dd ** 2) * 50, -2.0, 0.0)
        
        # Component 4: Expectancy Ratio (15%)
        if len(trades) > 5:  # REDUCED from 10 to 5 to encourage initial exploration
            trade_pnls = [t['pnl'] for t in trades]
            winning = [p for p in trade_pnls if p > 0]
            losing = [p for p in trade_pnls if p < 0]

            if len(winning) > 0 and len(losing) > 0:
                win_rate = len(winning) / len(trades)
                avg_win = np.mean(winning)
                avg_loss = np.mean(np.abs(losing))

                # FIXED: Use profit factor (ratio) instead of expectancy (dollars)
                profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss + 1e-8)
                # Center at 1.3 (breakeven after costs), normalize to [-2, +2]
                R_expectancy = np.clip((profit_factor - 1.3) / 0.7, -2.0, 2.0)
            else:
                # Only winners or only losers
                if len(winning) > 0:
                    R_expectancy = 1.0  # All winning trades
                else:
                    R_expectancy = -1.5  # All losing trades
        else:
            R_expectancy = -0.5  # Gentle penalty for few trades
        
        # Total terminal reward
        terminal_reward = (
            0.35 * R_sortino +
            0.25 * R_calmar +
            0.25 * R_drawdown +
            0.15 * R_expectancy
        )
        
        return terminal_reward
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_dd = np.max(drawdown)
        
        return max_dd


class TradingEnvironment(gym.Env):
    """Gym environment for EUR/USD trading."""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        config: Optional[TradingEnvConfig] = None,
        eval_mode: bool = False
    ):
        """
        Initialize trading environment.
        
        Args:
            data: OHLC data (EUR/USD)
            features: Normalized features (30 features)
            config: Environment configuration
            eval_mode: If True, use sequential episodes (no random start)
        """
        super().__init__()
        
        self.config = config or TradingEnvConfig()
        self.eval_mode = eval_mode
        
        # Data
        self.data = data.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.total_steps = len(self.data)
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=self.config.obs_min,
            high=self.config.obs_max,
            shape=(self.config.n_features,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=self.config.action_min,
            high=self.config.action_max,
            shape=(1,),
            dtype=np.float32
        )
        
        # Components
        self.cost_model = TransactionCostModel(self.config)
        self.position_sizer = PositionSizer(self.config)
        self.reward_calculator = RewardCalculator(self.config)
        
        # Episode state
        self.current_step = 0
        self.episode_length = 0
        self.episode_start = 0
        
        # Trading state
        self.equity = self.config.initial_capital
        self.position = 0.0  # Current position in lots
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        
        # Tracking
        self.equity_curve = []
        self.returns_buffer = []
        self.trades = []
        self.position_history = []
        self.daily_pnl = []
        
        # Metrics
        self.peak_equity = self.config.initial_capital
        self.total_trades = 0
        self.winning_trades = 0

        # Curriculum learning parameters
        self.curriculum_stage = 0  # 0=short, 1=medium, 2=long episodes
        self.episodes_completed = 0
        self.global_step_count = 0  # Total steps across ALL episodes (for warmup tracking)

        logger.info(f"Trading Environment initialized with {self.total_steps} steps")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.

        Returns:
            Initial observation
        """
        # Reset reward calculator
        self.reward_calculator.reset()

        # FIXED episode length (no curriculum, always 3000 steps)
        self.episode_length = self.config.episode_lengths[0]  # Always 3000
        
        # Set episode start (random or sequential)
        if self.eval_mode:
            # Sequential for evaluation
            if not hasattr(self, '_eval_position'):
                self._eval_position = 0
            self.episode_start = self._eval_position
            self._eval_position = min(
                self._eval_position + self.episode_length,
                self.total_steps - self.episode_length - 1
            )
        else:
            # Random start for training
            max_start = self.total_steps - self.episode_length - 100  # Buffer
            self.episode_start = np.random.randint(100, max_start)
        
        self.current_step = 0
        
        # Reset trading state
        self.equity = self.config.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        
        # Reset tracking
        self.equity_curve = [self.equity]
        self.returns_buffer = []
        self.trades = []
        self.position_history = []
        self.daily_pnl = []
        
        # Reset metrics
        self.peak_equity = self.equity
        self.total_trades = 0
        self.winning_trades = 0
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Agent action (position target)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Convert action to scalar
        action = float(action[0])
        action = np.clip(action, self.config.action_min, self.config.action_max)

        # PURE SURVIVAL WARMUP: Force no trading for first N steps
        if self.global_step_count < self.config.no_trading_warmup_steps:
            action = 0.0  # Force neutral position (no trading)
        elif self.global_step_count == self.config.no_trading_warmup_steps:
            logger.info(f"🎯 WARMUP COMPLETE! Trading enabled after {self.global_step_count} steps of pure survival training")
        
        # Get current state
        idx = self.episode_start + self.current_step
        current_price = self.data.iloc[idx]['close']
        current_high = self.data.iloc[idx]['high']
        current_low = self.data.iloc[idx]['low']
        timestamp = self.data.iloc[idx]['timestamp']
        hour = timestamp.hour
        
        # Calculate ATR for position sizing
        atr = self._calculate_atr(idx)
        volatility = atr / current_price  # Normalized volatility
        
        # Check Stop-Loss and Take-Profit
        sl_hit, tp_hit = self._check_sl_tp(current_high, current_low)
        
        if sl_hit or tp_hit:
            # Close position automatically
            exit_price = self.sl_price if sl_hit else self.tp_price
            pnl = self._close_position(exit_price, hour, volatility)
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'position': self.position,
                'pnl': pnl,
                'type': 'SL' if sl_hit else 'TP',
                'step': self.current_step
            })
            self.position = 0.0
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.tp_price = 0.0
        
        # Calculate target position
        target_position, new_sl, new_tp = self.position_sizer.calculate_position_size(
            self.equity,
            current_price,
            atr,
            action
        )
        
        # Execute position change
        position_change = abs(target_position - self.position)

        # CRITICAL FIX: Require minimum meaningful position change to count as trade
        # This prevents micro-adjustments from stochastic policy counting as trades
        # Threshold: 10% of typical position size (≈0.02 lots for 100k account)
        MIN_POSITION_CHANGE = 0.1  # Minimum lots to trigger a trade

        if position_change > MIN_POSITION_CHANGE:
            # Calculate transaction cost
            cost = self.cost_model.cost_in_dollars(
                position_change,
                current_price,
                hour,
                volatility
            )
            
            # Close old position if exists
            if abs(self.position) > 1e-6:
                pnl = self._close_position(current_price, hour, volatility)
                self.trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position,
                    'pnl': pnl,
                    'type': 'Normal',
                    'step': self.current_step
                })
            
            # Open new position
            if abs(target_position) > 1e-6:
                self.position = target_position
                self.entry_price = current_price
                self.sl_price = new_sl
                self.tp_price = new_tp
                self.equity -= cost
                self.total_trades += 1
        else:
            cost = 0.0
        
        # Update equity based on unrealized P&L
        if abs(self.position) > 1e-6:
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            current_equity = self.equity + unrealized_pnl
        else:
            current_equity = self.equity
        
        # Track metrics
        self.equity_curve.append(current_equity)
        self.position_history.append(self.position)
        
        # Calculate return
        if len(self.equity_curve) > 1:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / self.config.initial_capital
            self.returns_buffer.append(ret)
        
        # Update peak equity
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Count trades in last day (288 steps)
        recent_steps = self.current_step
        n_trades_today = sum(1 for t in self.trades if t['step'] > recent_steps - 288)
        
        # Calculate reward
        dense_reward = self.reward_calculator.calculate_dense_reward(
            current_equity,
            self.equity_curve[-2] if len(self.equity_curve) > 1 else self.config.initial_capital,
            self.config.initial_capital,
            self.returns_buffer,
            position_change,
            cost,
            n_trades_today
        )
        
        # Calculate current drawdown (used for termination at 80%)
        current_dd = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        # Check termination conditions
        done = False
        terminal_reward = 0.0

        # 1. Episode length reached
        if self.current_step >= self.episode_length - 1:
            done = True

        # 2. Drawdown >= 95% (VERY LENIENT - allow more learning time)
        if current_dd >= 0.95:
            done = True
            # NO PENALTY - Let natural trading rewards/losses teach the agent
            logger.warning(
                f"Drawdown critique at step {self.current_step} "
                f"(planned: {self.episode_length}): DD={current_dd:.2%}, "
                f"equity={current_equity:.2f}"
            )

        # 3. Balance = 0 (ruiné)
        if current_equity <= 0:
            done = True
            # NO PENALTY - Natural losses already reflected in returns
            logger.warning(f"Balance épuisée at step {self.current_step}: equity={current_equity:.2f}")

        # Calculate terminal reward if done
        if done and terminal_reward == 0.0:
            # Episode completed normally (reached episode_length)
            terminal_reward = self.reward_calculator.calculate_terminal_reward(
                self.returns_buffer,
                self.equity_curve,
                self.trades
            )

            # BONUS: Reward for completing full episode without catastrophic failure
            # Scaled by actual completion ratio
            completion_ratio = self.current_step / self.episode_length
            if completion_ratio >= 0.95:  # Completed at least 95% of episode
                survival_bonus = 2.0  # Significant bonus for full completion
                terminal_reward += survival_bonus
                logger.info(
                    f"Episode completed successfully! Steps: {self.current_step}/"
                    f"{self.episode_length}, bonus: +{survival_bonus:.2f}"
                )

        # Combine rewards
        total_reward = (
            self.config.dense_weight * dense_reward +
            self.config.terminal_weight * terminal_reward
        )

        # Scale reward to improve learning signal (gradients)
        total_reward *= self.config.reward_scale

        # Next step
        self.current_step += 1
        self.global_step_count += 1  # Track total steps across all episodes

        # Get next observation
        obs = self._get_observation()

        # Info dictionary
        info = {
            'equity': current_equity,
            'position': self.position,
            'total_trades': self.total_trades,
            'dense_reward': dense_reward,
            'terminal_reward': terminal_reward,
            'transaction_cost': cost,
            'current_step': self.current_step,
            'drawdown': current_dd
        }
        
        return obs, total_reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (features)."""
        idx = self.episode_start + self.current_step
        
        if idx >= len(self.features):
            idx = len(self.features) - 1
        
        obs = self.features.iloc[idx].values.astype(np.float32)
        
        # Clip to observation space bounds
        obs = np.clip(obs, self.config.obs_min, self.config.obs_max)
        
        return obs
    
    def _calculate_atr(self, idx: int, period: int = 14) -> float:
        """Calculate ATR at given index."""
        start_idx = max(0, idx - period)
        
        highs = self.data.iloc[start_idx:idx+1]['high']
        lows = self.data.iloc[start_idx:idx+1]['low']
        closes = self.data.iloc[start_idx:idx+1]['close']
        
        if len(highs) < 2:
            return 0.0001  # Default small ATR
        
        high_low = highs - lows
        high_close = np.abs(highs - closes.shift(1))
        low_close = np.abs(lows - closes.shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.mean()
        
        return atr
    
    def _check_sl_tp(self, current_high: float, current_low: float) -> Tuple[bool, bool]:
        """Check if Stop-Loss or Take-Profit was hit."""
        if abs(self.position) < 1e-6:
            return False, False
        
        sl_hit = False
        tp_hit = False
        
        if self.position > 0:  # Long position
            if current_low <= self.sl_price:
                sl_hit = True
            elif current_high >= self.tp_price:
                tp_hit = True
        else:  # Short position
            if current_high >= self.sl_price:
                sl_hit = True
            elif current_low <= self.tp_price:
                tp_hit = True
        
        return sl_hit, tp_hit
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if abs(self.position) < 1e-6:
            return 0.0
        
        price_change = current_price - self.entry_price
        pnl = self.position * 100000 * price_change  # 1 lot = 100,000 EUR
        
        return pnl
    
    def _close_position(self, exit_price: float, hour: int, volatility: float) -> float:
        """Close current position and return realized P&L."""
        if abs(self.position) < 1e-6:
            return 0.0
        
        # Calculate P&L
        pnl = self._calculate_unrealized_pnl(exit_price)
        
        # Subtract exit transaction cost
        position_size = abs(self.position)
        cost = self.cost_model.cost_in_dollars(
            position_size,
            exit_price,
            hour,
            volatility
        )
        
        net_pnl = pnl - cost
        
        # Update equity
        self.equity += net_pnl
        
        # Track winning trades
        if net_pnl > 0:
            self.winning_trades += 1
        
        return net_pnl
    
    def get_episode_metrics(self) -> Dict:
        """Calculate and return episode metrics."""
        if len(self.equity_curve) < 2:
            return {}
        
        equity_array = np.array(self.equity_curve)
        returns_array = np.array(self.returns_buffer)
        
        # Sharpe Ratio
        if len(returns_array) > 0 and np.std(returns_array) > 0:
            sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252 * 288)
        else:
            sharpe = 0.0
        
        # Sortino Ratio
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns_array) / np.std(downside_returns) * np.sqrt(252 * 288)
        else:
            sortino = 0.0
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_dd = np.max(drawdown)
        
        # Win Rate
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
        else:
            win_rate = 0.0
        
        # Profit Factor
        winning_pnls = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        losing_pnls = [abs(t['pnl']) for t in self.trades if t['pnl'] < 0]
        
        if len(losing_pnls) > 0:
            profit_factor = sum(winning_pnls) / sum(losing_pnls)
        else:
            profit_factor = float('inf') if len(winning_pnls) > 0 else 0.0
        
        # Calmar Ratio
        if max_dd > 0:
            annual_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
            calmar = annual_return / max_dd
        else:
            calmar = 0.0
        
        metrics = {
            'final_equity': equity_array[-1],
            'total_return': (equity_array[-1] - equity_array[0]) / equity_array[0],
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_position': np.mean(np.abs(self.position_history)) if self.position_history else 0.0
        }
        
        return metrics
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        if mode == 'human':
            metrics = self.get_episode_metrics()
            print(f"\nStep: {self.current_step}/{self.episode_length}")
            print(f"Equity: ${self.equity:.2f}")
            print(f"Position: {self.position:.2f} lots")
            print(f"Total Trades: {self.total_trades}")
            if metrics:
                print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
                print(f"Max DD: {metrics['max_drawdown']:.2%}")


def main():
    """Example usage of trading environment."""
    from backend.data_pipeline import DataPipeline
    from backend.feature_engineering import FeaturePipeline
    
    # Load data and features
    logger.info("Loading data and features...")
    data_pipeline = DataPipeline()
    train_data, val_data, test_data = data_pipeline.get_processed_data()
    
    feature_pipeline = FeaturePipeline()
    train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
        train_data, val_data, test_data
    )
    
    # Create environment
    logger.info("Creating trading environment...")
    env = TradingEnvironment(
        data=train_data['EURUSD'],
        features=train_features,
        eval_mode=False
    )
    
    # Test random episodes
    logger.info("\nTesting random episodes...")
    for episode in range(3):
        obs = env.reset()
        done = False
        total_reward = 0.0
        
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}")
        print(f"{'='*80}")
        
        while not done:
            # Random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if env.current_step % 500 == 0:
                print(f"Step {env.current_step}: Equity=${info['equity']:.2f}, "
                      f"Position={info['position']:.2f}, Reward={reward:.4f}")
        
        # Episode metrics
        metrics = env.get_episode_metrics()
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1} Complete")
        print(f"{'='*80}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Equity: ${metrics['final_equity']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    logger.info("\nTrading environment test complete!")


if __name__ == "__main__":
    main()
