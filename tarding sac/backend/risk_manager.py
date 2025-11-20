"""
SAC EUR/USD Trading System - Risk Manager
==========================================

This module implements comprehensive risk management for production trading.

Features:
- Risk-based position sizing (2% per trade)
- Daily loss monitoring (5% limit)
- Drawdown tracking (15% max)
- Peak equity tracking
- Kill switch with 3 levels (Soft, Hard, Emergency)
- Trade validation before execution
- Distribution shift detection (KS test, Wasserstein distance, CUSUM)
- Real-time risk metrics
- Automatic safety triggers

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    # Position sizing
    risk_per_trade: float = 0.02  # 2% per trade
    max_position_size: float = 100.0  # Maximum lots
    min_position_size: float = 0.01  # Minimum lots
    max_leverage: float = 30.0  # EU regulation limit
    
    # Daily limits
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    daily_profit_target: Optional[float] = None  # Optional daily profit target
    
    # Drawdown limits
    max_drawdown_limit: float = 0.15  # 15% maximum drawdown
    warning_drawdown: float = 0.10  # Warning at 10%
    
    # Margin requirements
    margin_call_threshold: float = 0.20  # 20% of initial capital
    margin_warning_threshold: float = 0.30  # Warning at 30%
    
    # Kill switch thresholds
    soft_stop_drawdown: float = 0.12  # 12% DD triggers soft stop
    hard_stop_drawdown: float = 0.15  # 15% DD triggers hard stop
    emergency_stop_equity: float = 0.20  # 20% of initial triggers emergency
    
    # Distribution shift detection
    shift_detection_window: int = 30  # Days for shift detection
    ks_pvalue_threshold: float = 0.05  # KS test p-value threshold
    wasserstein_multiplier: float = 2.0  # Multiplier for Wasserstein distance
    cusum_threshold: float = 3.0  # CUSUM threshold in std devs
    
    # Trading restrictions
    max_trades_per_day: int = 50
    max_trades_per_hour: int = 10
    min_time_between_trades: int = 60  # seconds
    
    # Monitoring
    log_dir: Path = Path("logs/risk")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)


class KillSwitchLevel:
    """Kill switch levels enumeration."""
    NONE = 0
    SOFT_STOP = 1  # Stop new positions, keep existing
    HARD_STOP = 2  # Close all positions and stop
    EMERGENCY = 3  # Immediate close all and disconnect


class EquityTracker:
    """Track equity, drawdowns, and peak values."""
    
    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        
        self.equity_history = deque(maxlen=10000)
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': initial_equity
        })
        
        self.daily_start_equity = initial_equity
        self.daily_pnl = 0.0
        
        logger.info(f"Equity Tracker initialized with {initial_equity:.2f}")
    
    def update(self, equity: float):
        """
        Update equity and calculate metrics.
        
        Args:
            equity: Current equity value
        """
        self.current_equity = equity
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Update history
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': equity
        })
        
        # Update daily P&L
        self.daily_pnl = equity - self.daily_start_equity
    
    def reset_daily(self):
        """Reset daily tracking (call at start of trading day)."""
        self.daily_start_equity = self.current_equity
        self.daily_pnl = 0.0
        logger.info(f"Daily reset: Starting equity = {self.current_equity:.2f}")
    
    def get_current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak.
        
        Returns:
            Drawdown as positive fraction (0.15 = 15% DD)
        """
        if self.peak_equity == 0:
            return 0.0
        
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        return max(0.0, drawdown)
    
    def get_daily_return(self) -> float:
        """
        Calculate daily return.
        
        Returns:
            Daily return as fraction
        """
        if self.daily_start_equity == 0:
            return 0.0
        
        return self.daily_pnl / self.daily_start_equity
    
    def get_total_return(self) -> float:
        """
        Calculate total return from initial.
        
        Returns:
            Total return as fraction
        """
        return (self.current_equity - self.initial_equity) / self.initial_equity
    
    def get_equity_curve(self, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.
        
        Args:
            periods: Number of recent periods (None = all)
            
        Returns:
            DataFrame with timestamp and equity
        """
        if periods is None:
            data = list(self.equity_history)
        else:
            data = list(self.equity_history)[-periods:]
        
        df = pd.DataFrame(data)
        return df


class TradeTracker:
    """Track trades for rate limiting and statistics."""
    
    def __init__(self):
        self.trades = deque(maxlen=1000)
        self.last_trade_time = None
        
        logger.info("Trade Tracker initialized")
    
    def add_trade(
        self,
        timestamp: datetime,
        action: str,
        volume: float,
        price: float,
        success: bool
    ):
        """Add a trade to history."""
        self.trades.append({
            'timestamp': timestamp,
            'action': action,
            'volume': volume,
            'price': price,
            'success': success
        })
        
        if success:
            self.last_trade_time = timestamp
    
    def get_trades_since(self, since: datetime) -> int:
        """Get number of trades since timestamp."""
        count = 0
        for trade in self.trades:
            if trade['timestamp'] >= since:
                count += 1
        return count
    
    def get_trades_last_day(self) -> int:
        """Get number of trades in last 24 hours."""
        cutoff = datetime.now() - timedelta(days=1)
        return self.get_trades_since(cutoff)
    
    def get_trades_last_hour(self) -> int:
        """Get number of trades in last hour."""
        cutoff = datetime.now() - timedelta(hours=1)
        return self.get_trades_since(cutoff)
    
    def seconds_since_last_trade(self) -> Optional[float]:
        """Get seconds since last successful trade."""
        if self.last_trade_time is None:
            return None
        
        delta = datetime.now() - self.last_trade_time
        return delta.total_seconds()


class DistributionShiftDetector:
    """Detect distribution shifts in returns."""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        
        # Reference distribution (from training/validation)
        self.reference_returns = None
        self.reference_mean = 0.0
        self.reference_std = 1.0
        
        # Current rolling window
        self.current_returns = deque(maxlen=config.shift_detection_window * 288)
        
        # CUSUM tracking
        self.cusum_positive = 0.0
        self.cusum_negative = 0.0
        
        # Historical baselines
        self.historical_wasserstein = []
        
        logger.info("Distribution Shift Detector initialized")
    
    def set_reference_distribution(self, returns: np.ndarray):
        """
        Set reference distribution from training data.
        
        Args:
            returns: Array of training returns
        """
        self.reference_returns = returns
        self.reference_mean = np.mean(returns)
        self.reference_std = np.std(returns)
        
        logger.info(f"Reference distribution set: "
                   f"mean={self.reference_mean:.6f}, std={self.reference_std:.6f}")
    
    def add_return(self, ret: float):
        """Add a new return to current window."""
        self.current_returns.append(ret)
        
        # Update CUSUM
        deviation = (ret - self.reference_mean) / self.reference_std
        self.cusum_positive = max(0, self.cusum_positive + deviation - 0.5)
        self.cusum_negative = max(0, self.cusum_negative - deviation - 0.5)
    
    def detect_shift(self) -> Dict:
        """
        Detect distribution shift using multiple methods.
        
        Returns:
            Dictionary with detection results
        """
        if len(self.current_returns) < 100:
            return {
                'shift_detected': False,
                'reason': 'Insufficient data',
                'tests': {}
            }
        
        current_array = np.array(self.current_returns)
        
        results = {
            'shift_detected': False,
            'reason': None,
            'tests': {}
        }
        
        # Test 1: Kolmogorov-Smirnov test
        if self.reference_returns is not None:
            ks_statistic, ks_pvalue = stats.ks_2samp(
                self.reference_returns,
                current_array
            )
            
            results['tests']['ks'] = {
                'statistic': ks_statistic,
                'pvalue': ks_pvalue,
                'shifted': ks_pvalue < self.config.ks_pvalue_threshold
            }
            
            if ks_pvalue < self.config.ks_pvalue_threshold:
                results['shift_detected'] = True
                results['reason'] = f'KS test p-value {ks_pvalue:.4f} < {self.config.ks_pvalue_threshold}'
        
        # Test 2: Wasserstein distance
        if self.reference_returns is not None:
            wasserstein_dist = stats.wasserstein_distance(
                self.reference_returns,
                current_array
            )
            
            # Calculate baseline if we have historical data
            if len(self.historical_wasserstein) > 10:
                baseline = np.mean(self.historical_wasserstein)
                threshold = baseline * self.config.wasserstein_multiplier
            else:
                threshold = self.reference_std * self.config.wasserstein_multiplier
            
            results['tests']['wasserstein'] = {
                'distance': wasserstein_dist,
                'threshold': threshold,
                'shifted': wasserstein_dist > threshold
            }
            
            self.historical_wasserstein.append(wasserstein_dist)
            if len(self.historical_wasserstein) > 100:
                self.historical_wasserstein.pop(0)
            
            if wasserstein_dist > threshold:
                results['shift_detected'] = True
                results['reason'] = f'Wasserstein distance {wasserstein_dist:.4f} > {threshold:.4f}'
        
        # Test 3: CUSUM
        cusum_max = max(self.cusum_positive, self.cusum_negative)
        
        results['tests']['cusum'] = {
            'positive': self.cusum_positive,
            'negative': self.cusum_negative,
            'max': cusum_max,
            'threshold': self.config.cusum_threshold,
            'shifted': cusum_max > self.config.cusum_threshold
        }
        
        if cusum_max > self.config.cusum_threshold:
            results['shift_detected'] = True
            results['reason'] = f'CUSUM {cusum_max:.2f} > {self.config.cusum_threshold}'
        
        # Test 4: Mean and volatility change
        current_mean = np.mean(current_array)
        current_std = np.std(current_array)
        
        mean_change = abs(current_mean - self.reference_mean) / self.reference_std
        vol_change = abs(current_std - self.reference_std) / self.reference_std
        
        results['tests']['moments'] = {
            'current_mean': current_mean,
            'current_std': current_std,
            'mean_change_std': mean_change,
            'vol_change_pct': vol_change
        }
        
        return results


class RiskManager:
    """
    Comprehensive risk management system.
    
    Monitors equity, drawdowns, daily losses, and implements kill switch.
    """
    
    def __init__(
        self,
        initial_equity: float,
        config: Optional[RiskConfig] = None
    ):
        self.config = config or RiskConfig()
        
        # Trackers
        self.equity_tracker = EquityTracker(initial_equity)
        self.trade_tracker = TradeTracker()
        self.shift_detector = DistributionShiftDetector(self.config)
        
        # Kill switch state
        self.kill_switch_level = KillSwitchLevel.NONE
        self.kill_switch_reason = None
        self.kill_switch_timestamp = None
        
        # Warnings issued
        self.warnings_issued = set()
        
        # Risk metrics
        self.risk_metrics = {}
        
        logger.info("="*80)
        logger.info("Risk Manager Initialized")
        logger.info("="*80)
        logger.info(f"Initial Equity: {initial_equity:.2f}")
        logger.info(f"Risk per Trade: {self.config.risk_per_trade*100:.1f}%")
        logger.info(f"Daily Loss Limit: {self.config.daily_loss_limit*100:.1f}%")
        logger.info(f"Max Drawdown Limit: {self.config.max_drawdown_limit*100:.1f}%")
        logger.info("="*80)
    
    def update_equity(self, equity: float):
        """
        Update current equity and check risk limits.
        
        Args:
            equity: Current equity value
        """
        self.equity_tracker.update(equity)
        
        # Calculate metrics
        self._update_risk_metrics()
        
        # Check risk limits
        self._check_risk_limits()
    
    def _update_risk_metrics(self):
        """Update all risk metrics."""
        self.risk_metrics = {
            'current_equity': self.equity_tracker.current_equity,
            'peak_equity': self.equity_tracker.peak_equity,
            'current_drawdown': self.equity_tracker.get_current_drawdown(),
            'daily_return': self.equity_tracker.get_daily_return(),
            'daily_pnl': self.equity_tracker.daily_pnl,
            'total_return': self.equity_tracker.get_total_return(),
            'trades_today': self.trade_tracker.get_trades_last_day(),
            'trades_last_hour': self.trade_tracker.get_trades_last_hour(),
            'kill_switch_level': self.kill_switch_level,
            'kill_switch_reason': self.kill_switch_reason
        }
    
    def _check_risk_limits(self):
        """Check all risk limits and trigger kill switch if needed."""
        equity = self.equity_tracker.current_equity
        initial = self.equity_tracker.initial_equity
        drawdown = self.equity_tracker.get_current_drawdown()
        daily_return = self.equity_tracker.get_daily_return()
        
        # Check Emergency Stop (immediate action required)
        if equity < initial * self.config.emergency_stop_equity:
            self._trigger_kill_switch(
                KillSwitchLevel.EMERGENCY,
                f"Equity {equity:.2f} < {self.config.emergency_stop_equity*100:.0f}% of initial"
            )
            return
        
        # Check Hard Stop (close all positions)
        if drawdown >= self.config.hard_stop_drawdown:
            self._trigger_kill_switch(
                KillSwitchLevel.HARD_STOP,
                f"Drawdown {drawdown:.2%} >= {self.config.hard_stop_drawdown:.0%}"
            )
            return
        
        if daily_return <= -self.config.daily_loss_limit:
            self._trigger_kill_switch(
                KillSwitchLevel.HARD_STOP,
                f"Daily loss {daily_return:.2%} >= {self.config.daily_loss_limit:.0%}"
            )
            return
        
        # Check Soft Stop (stop new positions)
        if drawdown >= self.config.soft_stop_drawdown:
            self._trigger_kill_switch(
                KillSwitchLevel.SOFT_STOP,
                f"Drawdown {drawdown:.2%} >= {self.config.soft_stop_drawdown:.0%}"
            )
            return
        
        # Issue warnings
        if drawdown >= self.config.warning_drawdown:
            self._issue_warning(
                'drawdown',
                f"Drawdown at {drawdown:.2%} (warning threshold: {self.config.warning_drawdown:.0%})"
            )
        
        if equity < initial * self.config.margin_warning_threshold:
            self._issue_warning(
                'margin',
                f"Equity {equity:.2f} below margin warning threshold"
            )
    
    def _trigger_kill_switch(self, level: int, reason: str):
        """
        Trigger kill switch at specified level.
        
        Args:
            level: Kill switch level
            reason: Reason for triggering
        """
        if level <= self.kill_switch_level:
            return  # Already at this level or higher
        
        self.kill_switch_level = level
        self.kill_switch_reason = reason
        self.kill_switch_timestamp = datetime.now()
        
        level_names = {
            KillSwitchLevel.SOFT_STOP: "SOFT STOP",
            KillSwitchLevel.HARD_STOP: "HARD STOP",
            KillSwitchLevel.EMERGENCY: "EMERGENCY STOP"
        }
        
        logger.critical("="*80)
        logger.critical(f"KILL SWITCH TRIGGERED: {level_names.get(level, 'UNKNOWN')}")
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Timestamp: {self.kill_switch_timestamp}")
        logger.critical("="*80)
    
    def _issue_warning(self, warning_type: str, message: str):
        """Issue a warning (only once per type)."""
        if warning_type not in self.warnings_issued:
            logger.warning(f"RISK WARNING [{warning_type}]: {message}")
            self.warnings_issued.add(warning_type)
    
    def reset_kill_switch(self, manual_override: bool = False):
        """
        Reset kill switch (requires manual override).
        
        Args:
            manual_override: Must be True to reset
        """
        if not manual_override:
            logger.error("Kill switch reset requires manual_override=True")
            return False
        
        logger.warning("Kill switch manually reset")
        self.kill_switch_level = KillSwitchLevel.NONE
        self.kill_switch_reason = None
        self.kill_switch_timestamp = None
        
        return True
    
    def validate_trade(
        self,
        position_size: float,
        current_equity: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if a trade can be executed.
        
        Args:
            position_size: Proposed position size in lots
            current_equity: Current account equity
            
        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        # Check kill switch
        if self.kill_switch_level >= KillSwitchLevel.HARD_STOP:
            return False, f"Kill switch active: {self.kill_switch_reason}"
        
        if self.kill_switch_level == KillSwitchLevel.SOFT_STOP:
            return False, "Soft stop active: no new positions allowed"
        
        # Check position size limits
        if position_size < self.config.min_position_size:
            return False, f"Position size {position_size:.2f} < minimum {self.config.min_position_size:.2f}"
        
        if position_size > self.config.max_position_size:
            return False, f"Position size {position_size:.2f} > maximum {self.config.max_position_size:.2f}"
        
        # Check leverage
        notional_value = position_size * 100000  # 1 lot = 100,000 EUR
        leverage = notional_value / current_equity
        
        if leverage > self.config.max_leverage:
            return False, f"Leverage {leverage:.1f} > maximum {self.config.max_leverage:.1f}"
        
        # Check trade frequency
        trades_today = self.trade_tracker.get_trades_last_day()
        if trades_today >= self.config.max_trades_per_day:
            return False, f"Daily trade limit reached: {trades_today}/{self.config.max_trades_per_day}"
        
        trades_hour = self.trade_tracker.get_trades_last_hour()
        if trades_hour >= self.config.max_trades_per_hour:
            return False, f"Hourly trade limit reached: {trades_hour}/{self.config.max_trades_per_hour}"
        
        # Check minimum time between trades
        seconds_since = self.trade_tracker.seconds_since_last_trade()
        if seconds_since is not None and seconds_since < self.config.min_time_between_trades:
            return False, f"Too soon after last trade: {seconds_since:.0f}s < {self.config.min_time_between_trades}s"
        
        return True, None
    
    def calculate_position_size(
        self,
        equity: float,
        atr: float,
        sl_multiplier: float = 2.0
    ) -> float:
        """
        Calculate risk-based position size.
        
        Args:
            equity: Current equity
            atr: Current ATR value
            sl_multiplier: Stop-loss ATR multiplier
            
        Returns:
            Position size in lots
        """
        # Risk amount
        risk_amount = equity * self.config.risk_per_trade
        
        # Stop-loss distance in pips
        sl_distance = sl_multiplier * atr
        sl_pips = sl_distance / 0.0001  # For 5-digit quotes
        
        # Position size
        # 1 lot = 100,000 EUR, 1 pip = $10 for 1 lot EUR/USD
        pip_value = 10.0
        position_size = risk_amount / (sl_pips * pip_value)
        
        # Apply constraints
        position_size = max(self.config.min_position_size, position_size)
        position_size = min(self.config.max_position_size, position_size)
        
        # Round to 0.01
        position_size = round(position_size, 2)
        
        return position_size
    
    def log_trade(
        self,
        action: str,
        volume: float,
        price: float,
        success: bool,
        ret: Optional[float] = None
    ):
        """
        Log a trade.
        
        Args:
            action: 'buy' or 'sell'
            volume: Position size
            price: Execution price
            success: Whether trade was successful
            ret: Return from trade (if applicable)
        """
        self.trade_tracker.add_trade(
            timestamp=datetime.now(),
            action=action,
            volume=volume,
            price=price,
            success=success
        )
        
        if ret is not None:
            self.shift_detector.add_return(ret)
    
    def check_distribution_shift(self) -> Dict:
        """
        Check for distribution shift in returns.
        
        Returns:
            Dictionary with shift detection results
        """
        return self.shift_detector.detect_shift()
    
    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report.
        
        Returns:
            Dictionary with all risk metrics and status
        """
        shift_status = self.check_distribution_shift()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'equity': {
                'current': self.equity_tracker.current_equity,
                'initial': self.equity_tracker.initial_equity,
                'peak': self.equity_tracker.peak_equity,
                'daily_start': self.equity_tracker.daily_start_equity
            },
            'returns': {
                'total': self.equity_tracker.get_total_return(),
                'daily': self.equity_tracker.get_daily_return(),
                'daily_pnl': self.equity_tracker.daily_pnl
            },
            'drawdown': {
                'current': self.equity_tracker.get_current_drawdown(),
                'limit': self.config.max_drawdown_limit,
                'warning': self.config.warning_drawdown,
                'percentage_of_limit': self.equity_tracker.get_current_drawdown() / self.config.max_drawdown_limit * 100
            },
            'daily_limits': {
                'loss_limit': self.config.daily_loss_limit,
                'current_return': self.equity_tracker.get_daily_return(),
                'percentage_of_limit': abs(self.equity_tracker.get_daily_return()) / self.config.daily_loss_limit * 100
            },
            'trades': {
                'today': self.trade_tracker.get_trades_last_day(),
                'last_hour': self.trade_tracker.get_trades_last_hour(),
                'max_per_day': self.config.max_trades_per_day,
                'max_per_hour': self.config.max_trades_per_hour
            },
            'kill_switch': {
                'level': self.kill_switch_level,
                'level_name': self._get_kill_switch_name(),
                'reason': self.kill_switch_reason,
                'triggered_at': self.kill_switch_timestamp.isoformat() if self.kill_switch_timestamp else None
            },
            'distribution_shift': shift_status,
            'warnings': list(self.warnings_issued)
        }
        
        return report
    
    def _get_kill_switch_name(self) -> str:
        """Get kill switch level name."""
        names = {
            KillSwitchLevel.NONE: "None",
            KillSwitchLevel.SOFT_STOP: "Soft Stop",
            KillSwitchLevel.HARD_STOP: "Hard Stop",
            KillSwitchLevel.EMERGENCY: "Emergency"
        }
        return names.get(self.kill_switch_level, "Unknown")
    
    def reset_daily_tracking(self):
        """Reset daily tracking (call at start of each trading day)."""
        self.equity_tracker.reset_daily()
        
        # Clear daily warnings
        self.warnings_issued.discard('daily_loss')
        
        logger.info("Daily tracking reset")
    
    def set_reference_returns(self, returns: np.ndarray):
        """
        Set reference return distribution for shift detection.
        
        Args:
            returns: Array of reference returns (from backtest/validation)
        """
        self.shift_detector.set_reference_distribution(returns)
        logger.info("Reference returns set for distribution shift detection")


def main():
    """Example usage of risk manager."""
    
    # Initialize risk manager
    initial_equity = 100000.0
    config = RiskConfig(
        risk_per_trade=0.02,
        daily_loss_limit=0.05,
        max_drawdown_limit=0.15
    )
    
    risk_manager = RiskManager(initial_equity, config)
    
    # Set reference returns (from backtest)
    reference_returns = np.random.normal(0.0001, 0.01, 10000)
    risk_manager.set_reference_returns(reference_returns)
    
    # Simulate trading day
    logger.info("\n" + "="*80)
    logger.info("Risk Manager Demo - Simulating Trading Day")
    logger.info("="*80)
    
    current_equity = initial_equity
    
    # Simulate some trades
    for i in range(5):
        logger.info(f"\nTrade {i+1}:")
        
        # Calculate position size
        atr = 0.0010
        position_size = risk_manager.calculate_position_size(
            equity=current_equity,
            atr=atr,
            sl_multiplier=2.0
        )
        
        logger.info(f"  Calculated position size: {position_size:.2f} lots")
        
        # Validate trade
        can_trade, reason = risk_manager.validate_trade(position_size, current_equity)
        
        if can_trade:
            logger.info(f"  Trade validation: PASSED")
            
            # Simulate trade execution
            success = True
            ret = np.random.normal(0.0001, 0.01)
            
            risk_manager.log_trade(
                action='buy',
                volume=position_size,
                price=1.10000,
                success=success,
                ret=ret
            )
            
            # Update equity
            current_equity *= (1 + ret)
            risk_manager.update_equity(current_equity)
            
            logger.info(f"  Trade executed: Return = {ret:.4%}")
            logger.info(f"  New equity: {current_equity:.2f}")
        else:
            logger.info(f"  Trade validation: FAILED - {reason}")
    
    # Generate risk report
    logger.info("\n" + "="*80)
    logger.info("Risk Report")
    logger.info("="*80)
    
    report = risk_manager.get_risk_report()
    
    logger.info(f"\nEquity:")
    logger.info(f"  Current: {report['equity']['current']:.2f}")
    logger.info(f"  Peak: {report['equity']['peak']:.2f}")
    
    logger.info(f"\nReturns:")
    logger.info(f"  Total: {report['returns']['total']:.2%}")
    logger.info(f"  Daily: {report['returns']['daily']:.2%}")
    
    logger.info(f"\nDrawdown:")
    logger.info(f"  Current: {report['drawdown']['current']:.2%}")
    logger.info(f"  % of Limit: {report['drawdown']['percentage_of_limit']:.1f}%")
    
    logger.info(f"\nTrades:")
    logger.info(f"  Today: {report['trades']['today']}")
    logger.info(f"  Last Hour: {report['trades']['last_hour']}")
    
    logger.info(f"\nKill Switch:")
    logger.info(f"  Status: {report['kill_switch']['level_name']}")
    
    logger.info(f"\nDistribution Shift:")
    logger.info(f"  Detected: {report['distribution_shift']['shift_detected']}")
    
    logger.info("\n" + "="*80)
    logger.info("Risk Manager Demo Complete")
    logger.info("="*80)


if __name__ == "__main__":
    main()
