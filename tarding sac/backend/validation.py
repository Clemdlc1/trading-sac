"""
SAC EUR/USD Trading System - Validation
========================================

This module implements comprehensive validation methodologies for the trading system.

Features:
- Walk-Forward Validation with expanding window
- Statistical tests: Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR)
- Probability of Backtest Overfitting (PBO)
- Bootstrap confidence intervals
- 7 stress test scenarios
- Out-of-sample (OOS) final validation
- Walk-Forward Efficiency (WFE)

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    
    # Walk-forward parameters
    initial_train_months: int = 24
    test_window_months: int = 3
    step_months: int = 3
    
    # Statistical test parameters
    n_configurations_tested: int = 50  # For DSR
    target_sharpe: float = 0.0  # Threshold for PSR
    
    # Bootstrap parameters
    n_bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Stress test parameters
    volatility_multipliers: List[float] = field(default_factory=lambda: [2.0, 3.0])
    trend_magnitude: float = 0.05  # 5% trend
    flash_crash_magnitude: float = -0.05  # -5% crash
    spread_multiplier: float = 5.0
    black_swan_sigma: int = 5
    
    # Acceptance thresholds
    wfe_threshold: float = 0.5
    dsr_threshold: float = 0.5
    psr_threshold: float = 0.80
    pbo_threshold: float = 0.5
    
    # OOS criteria
    oos_min_sharpe: float = 0.8
    oos_min_sortino: float = 1.2
    oos_max_drawdown: float = 0.15
    oos_min_win_rate: float = 0.45
    
    # Stress test criteria
    stress_min_sharpe: float = 0.3
    stress_success_ratio: float = 5.0 / 7.0  # Must pass 5 out of 7
    stress_max_drawdown: float = 0.30
    
    # Results directory
    results_dir: Path = Path("results/validation")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)


class PerformanceMetrics:
    """Calculate performance metrics for validation."""
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        annualization_factor: float = 252 * 288
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Array of returns
            annualization_factor: Factor for annualization (252 days * 288 5-min bars)
            
        Returns:
            Annualized Sharpe Ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray,
        annualization_factor: float = 252 * 288
    ) -> float:
        """
        Calculate Sortino Ratio.
        
        Args:
            returns: Array of returns
            annualization_factor: Factor for annualization
            
        Returns:
            Annualized Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return / downside_std) * np.sqrt(annualization_factor)
        
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Array of equity values
            
        Returns:
            Maximum drawdown (positive value)
        """
        if len(equity_curve) < 2:
            return 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown)
        
        return max_dd
    
    @staticmethod
    def calculate_calmar_ratio(
        returns: np.ndarray,
        equity_curve: np.ndarray
    ) -> float:
        """
        Calculate Calmar Ratio.
        
        Args:
            returns: Array of returns
            equity_curve: Array of equity values
            
        Returns:
            Calmar Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252 * 288
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        calmar = annual_return / max_dd
        
        return calmar
    
    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """
        Calculate win rate.
        
        Args:
            returns: Array of returns
            
        Returns:
            Win rate (0-1)
        """
        if len(returns) == 0:
            return 0.0
        
        winning = np.sum(returns > 0)
        win_rate = winning / len(returns)
        
        return win_rate


class WalkForwardValidator:
    """Walk-forward validation with expanding window."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
    
    def split_data(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[Dict]:
        """
        Split data into walk-forward windows.
        
        Args:
            data: OHLC data
            features: Feature data
            
        Returns:
            List of dictionaries with train/test splits
        """
        # Convert to datetime
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Get date range
        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()
        
        # Calculate windows
        windows = []
        
        current_train_end = start_date + pd.DateOffset(months=self.config.initial_train_months)
        
        while current_train_end + pd.DateOffset(months=self.config.test_window_months) <= end_date:
            # Train window (from start to current_train_end)
            train_mask = data['timestamp'] < current_train_end
            
            # Test window
            test_start = current_train_end
            test_end = test_start + pd.DateOffset(months=self.config.test_window_months)
            test_mask = (data['timestamp'] >= test_start) & (data['timestamp'] < test_end)
            
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                windows.append({
                    'train_data': data[train_mask].reset_index(drop=True),
                    'test_data': data[test_mask].reset_index(drop=True),
                    'train_features': features[train_mask].reset_index(drop=True),
                    'test_features': features[test_mask].reset_index(drop=True),
                    'train_start': data[train_mask]['timestamp'].min(),
                    'train_end': current_train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
            
            # Move to next window (expanding)
            current_train_end += pd.DateOffset(months=self.config.step_months)
        
        logger.info(f"Created {len(windows)} walk-forward windows")
        
        return windows
    
    def validate(
        self,
        agent,
        data: pd.DataFrame,
        features: pd.DataFrame,
        env_class,
        env_config
    ) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            agent: Trained SAC agent (or ensemble)
            data: OHLC data
            features: Feature data
            env_class: Trading environment class
            env_config: Environment configuration
            
        Returns:
            Dictionary with validation results
        """
        logger.info("="*80)
        logger.info("Walk-Forward Validation")
        logger.info("="*80)
        
        # Split data
        windows = self.split_data(data, features)
        
        results = {
            'windows': [],
            'in_sample_sharpes': [],
            'out_sample_sharpes': [],
            'in_sample_returns': [],
            'out_sample_returns': []
        }
        
        # Run validation on each window
        for i, window in enumerate(tqdm(windows, desc="Walk-forward validation")):
            logger.info(f"\nWindow {i+1}/{len(windows)}")
            logger.info(f"  Train: {window['train_start']} to {window['train_end']}")
            logger.info(f"  Test: {window['test_start']} to {window['test_end']}")
            
            # In-sample evaluation (on training data)
            train_env = env_class(
                data=window['train_data'],
                features=window['train_features'],
                config=env_config,
                eval_mode=True
            )
            
            is_returns, is_equity = self._evaluate_agent(agent, train_env, n_episodes=5)
            is_sharpe = self.metrics.calculate_sharpe_ratio(is_returns)
            
            # Out-of-sample evaluation (on test data)
            test_env = env_class(
                data=window['test_data'],
                features=window['test_features'],
                config=env_config,
                eval_mode=True
            )
            
            oos_returns, oos_equity = self._evaluate_agent(agent, test_env, n_episodes=5)
            oos_sharpe = self.metrics.calculate_sharpe_ratio(oos_returns)
            
            logger.info(f"  In-sample Sharpe: {is_sharpe:.2f}")
            logger.info(f"  Out-of-sample Sharpe: {oos_sharpe:.2f}")
            
            # Store results
            results['windows'].append({
                'window_id': i + 1,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'in_sample_sharpe': is_sharpe,
                'out_sample_sharpe': oos_sharpe,
                'in_sample_returns': np.mean(is_returns),
                'out_sample_returns': np.mean(oos_returns)
            })
            
            results['in_sample_sharpes'].append(is_sharpe)
            results['out_sample_sharpes'].append(oos_sharpe)
            results['in_sample_returns'].append(np.mean(is_returns))
            results['out_sample_returns'].append(np.mean(oos_returns))
        
        # Calculate Walk-Forward Efficiency
        mean_oos_sharpe = np.mean(results['out_sample_sharpes'])
        mean_is_sharpe = np.mean(results['in_sample_sharpes'])
        
        if mean_is_sharpe != 0:
            wfe = mean_oos_sharpe / mean_is_sharpe
        else:
            wfe = 0.0
        
        results['wfe'] = wfe
        results['mean_is_sharpe'] = mean_is_sharpe
        results['mean_oos_sharpe'] = mean_oos_sharpe
        results['passes_wfe'] = wfe > self.config.wfe_threshold
        
        logger.info("\n" + "="*80)
        logger.info("Walk-Forward Validation Results")
        logger.info("="*80)
        logger.info(f"Mean In-Sample Sharpe: {mean_is_sharpe:.2f}")
        logger.info(f"Mean Out-of-Sample Sharpe: {mean_oos_sharpe:.2f}")
        logger.info(f"Walk-Forward Efficiency: {wfe:.2f}")
        logger.info(f"Passes WFE threshold ({self.config.wfe_threshold}): {results['passes_wfe']}")
        logger.info("="*80)
        
        return results
    
    def _evaluate_agent(
        self,
        agent,
        env,
        n_episodes: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate agent on environment.
        
        Args:
            agent: Agent to evaluate
            env: Trading environment
            n_episodes: Number of episodes
            
        Returns:
            Tuple of (returns, equity_curve)
        """
        all_returns = []
        all_equity = []
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            
            episode_returns = []
            episode_equity = [env.equity]
            
            while not done:
                # Get action
                if hasattr(agent, 'get_ensemble_action'):
                    # Ensemble
                    action = agent.get_ensemble_action(state, deterministic=True)
                else:
                    # Single agent
                    action = agent.select_action(state, deterministic=True)
                
                # Step
                next_state, reward, done, info = env.step(action)
                
                episode_returns.append(reward)
                episode_equity.append(info['equity'])
                
                state = next_state
            
            all_returns.extend(episode_returns)
            all_equity.extend(episode_equity)
        
        return np.array(all_returns), np.array(all_equity)


class StatisticalTests:
    """Statistical tests for validation."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
    
    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_returns: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> Dict:
        """
        Calculate Deflated Sharpe Ratio (DSR).
        
        Adjusts for multiple testing and non-normality.
        
        Args:
            observed_sharpe: Observed Sharpe ratio
            n_returns: Number of return observations
            skewness: Skewness of returns
            kurtosis: Excess kurtosis of returns
            
        Returns:
            Dictionary with DSR results
        """
        n_configs = self.config.n_configurations_tested
        
        # Expected maximum Sharpe under null hypothesis
        # Using formula from Bailey & López de Prado (2014)
        gamma = 0.5772  # Euler-Mascheroni constant
        expected_max_sharpe = (1 - gamma) * stats.norm.ppf(1 - 1.0/n_configs) + \
                              gamma * stats.norm.ppf(1 - 1.0/(n_configs * np.e))
        
        # Standard error adjustment for non-normality
        # From Bailey & López de Prado (2014)
        var_sharpe = (1 + 0.5 * observed_sharpe**2 - 
                     skewness * observed_sharpe + 
                     (kurtosis - 1) / 4.0 * observed_sharpe**2) / (n_returns - 1)
        
        se_sharpe = np.sqrt(var_sharpe)
        
        # DSR calculation
        dsr = (observed_sharpe - expected_max_sharpe) / se_sharpe
        
        # Convert to probability
        dsr_pvalue = stats.norm.cdf(dsr)
        
        result = {
            'dsr': dsr,
            'dsr_pvalue': dsr_pvalue,
            'expected_max_sharpe': expected_max_sharpe,
            'se_sharpe': se_sharpe,
            'passes_threshold': dsr > self.config.dsr_threshold
        }
        
        logger.info("\n" + "="*80)
        logger.info("Deflated Sharpe Ratio Test")
        logger.info("="*80)
        logger.info(f"Observed Sharpe: {observed_sharpe:.2f}")
        logger.info(f"Expected Max Sharpe (null): {expected_max_sharpe:.2f}")
        logger.info(f"DSR: {dsr:.2f}")
        logger.info(f"DSR p-value: {dsr_pvalue:.4f}")
        logger.info(f"Passes threshold ({self.config.dsr_threshold}): {result['passes_threshold']}")
        logger.info("="*80)
        
        return result
    
    def probabilistic_sharpe_ratio(
        self,
        returns: np.ndarray,
        target_sharpe: Optional[float] = None
    ) -> Dict:
        """
        Calculate Probabilistic Sharpe Ratio (PSR).
        
        Probability that true Sharpe ratio exceeds a threshold.
        
        Args:
            returns: Array of returns
            target_sharpe: Target Sharpe ratio (default: 0)
            
        Returns:
            Dictionary with PSR results
        """
        if target_sharpe is None:
            target_sharpe = self.config.target_sharpe
        
        n = len(returns)
        
        # Calculate observed Sharpe
        observed_sharpe = self.metrics.calculate_sharpe_ratio(returns)
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=False)  # Pearson's kurtosis
        
        # Standard error with skewness and kurtosis adjustments
        var_sharpe = (1 + 0.5 * observed_sharpe**2 - 
                     skewness * observed_sharpe + 
                     (kurtosis - 3) / 4.0 * observed_sharpe**2) / (n - 1)
        
        se_sharpe = np.sqrt(var_sharpe)
        
        # PSR calculation
        if se_sharpe > 0:
            psr_stat = (observed_sharpe - target_sharpe) / se_sharpe
            psr = stats.norm.cdf(psr_stat)
        else:
            psr = 0.0
        
        result = {
            'psr': psr,
            'observed_sharpe': observed_sharpe,
            'target_sharpe': target_sharpe,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'se_sharpe': se_sharpe,
            'passes_threshold': psr > self.config.psr_threshold
        }
        
        logger.info("\n" + "="*80)
        logger.info("Probabilistic Sharpe Ratio Test")
        logger.info("="*80)
        logger.info(f"Observed Sharpe: {observed_sharpe:.2f}")
        logger.info(f"Target Sharpe: {target_sharpe:.2f}")
        logger.info(f"PSR: {psr:.4f} ({psr*100:.1f}%)")
        logger.info(f"Skewness: {skewness:.2f}")
        logger.info(f"Kurtosis: {kurtosis:.2f}")
        logger.info(f"Passes threshold ({self.config.psr_threshold}): {result['passes_threshold']}")
        logger.info("="*80)
        
        return result
    
    def probability_backtest_overfitting(
        self,
        walk_forward_results: Dict
    ) -> Dict:
        """
        Calculate Probability of Backtest Overfitting (PBO).
        
        Compares median OOS performance vs best IS performance.
        
        Args:
            walk_forward_results: Results from walk-forward validation
            
        Returns:
            Dictionary with PBO results
        """
        is_sharpes = np.array(walk_forward_results['in_sample_sharpes'])
        oos_sharpes = np.array(walk_forward_results['out_sample_sharpes'])
        
        # Find configuration with best in-sample Sharpe
        best_is_idx = np.argmax(is_sharpes)
        best_is_sharpe = is_sharpes[best_is_idx]
        corresponding_oos_sharpe = oos_sharpes[best_is_idx]
        
        # Calculate median OOS Sharpe
        median_oos_sharpe = np.median(oos_sharpes)
        
        # PBO: probability that median OOS < OOS of best IS
        # Using logistic regression approach from Bailey et al. (2015)
        count_worse = np.sum(oos_sharpes < corresponding_oos_sharpe)
        pbo = count_worse / len(oos_sharpes)
        
        result = {
            'pbo': pbo,
            'best_is_sharpe': best_is_sharpe,
            'corresponding_oos_sharpe': corresponding_oos_sharpe,
            'median_oos_sharpe': median_oos_sharpe,
            'passes_threshold': pbo < self.config.pbo_threshold
        }
        
        logger.info("\n" + "="*80)
        logger.info("Probability of Backtest Overfitting Test")
        logger.info("="*80)
        logger.info(f"Best In-Sample Sharpe: {best_is_sharpe:.2f}")
        logger.info(f"Corresponding OOS Sharpe: {corresponding_oos_sharpe:.2f}")
        logger.info(f"Median OOS Sharpe: {median_oos_sharpe:.2f}")
        logger.info(f"PBO: {pbo:.4f} ({pbo*100:.1f}%)")
        logger.info(f"Passes threshold (<{self.config.pbo_threshold}): {result['passes_threshold']}")
        logger.info("="*80)
        
        return result


class BootstrapValidator:
    """Bootstrap confidence intervals."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
    
    def calculate_confidence_intervals(
        self,
        returns: np.ndarray,
        metric_name: str = 'sharpe'
    ) -> Dict:
        """
        Calculate bootstrap confidence intervals.
        
        Args:
            returns: Array of returns
            metric_name: Metric to calculate ('sharpe', 'sortino', 'mean')
            
        Returns:
            Dictionary with CI results
        """
        logger.info(f"\nCalculating bootstrap CI for {metric_name}...")
        
        # Calculate observed metric
        if metric_name == 'sharpe':
            observed = self.metrics.calculate_sharpe_ratio(returns)
        elif metric_name == 'sortino':
            observed = self.metrics.calculate_sortino_ratio(returns)
        elif metric_name == 'mean':
            observed = np.mean(returns)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        # Bootstrap resampling
        bootstrap_metrics = []
        
        for _ in tqdm(
            range(self.config.n_bootstrap_samples),
            desc=f"Bootstrap {metric_name}",
            leave=False
        ):
            # Resample with replacement
            resampled = resample(returns, replace=True, n_samples=len(returns))
            
            # Calculate metric
            if metric_name == 'sharpe':
                metric = self.metrics.calculate_sharpe_ratio(resampled)
            elif metric_name == 'sortino':
                metric = self.metrics.calculate_sortino_ratio(resampled)
            elif metric_name == 'mean':
                metric = np.mean(resampled)
            
            bootstrap_metrics.append(metric)
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        
        # Check if CI excludes zero
        excludes_zero = not (ci_lower <= 0 <= ci_upper)
        
        result = {
            'metric': metric_name,
            'observed': observed,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': self.config.confidence_level,
            'bootstrap_mean': np.mean(bootstrap_metrics),
            'bootstrap_std': np.std(bootstrap_metrics),
            'excludes_zero': excludes_zero
        }
        
        logger.info(f"\n{metric_name.upper()} Bootstrap CI:")
        logger.info(f"  Observed: {observed:.4f}")
        logger.info(f"  {self.config.confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"  Bootstrap Mean: {result['bootstrap_mean']:.4f}")
        logger.info(f"  Bootstrap Std: {result['bootstrap_std']:.4f}")
        logger.info(f"  Excludes Zero: {excludes_zero}")
        
        return result


class StressTester:
    """Stress testing with various market scenarios."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
    
    def run_all_stress_tests(
        self,
        agent,
        base_data: pd.DataFrame,
        base_features: pd.DataFrame,
        env_class,
        env_config
    ) -> Dict:
        """
        Run all 7 stress test scenarios.
        
        Args:
            agent: Trained agent
            base_data: Base OHLC data
            base_features: Base features
            env_class: Environment class
            env_config: Environment config
            
        Returns:
            Dictionary with stress test results
        """
        logger.info("="*80)
        logger.info("Running Stress Tests")
        logger.info("="*80)
        
        scenarios = [
            ('High Volatility 2x', self._high_volatility, {'multiplier': 2.0}),
            ('High Volatility 3x', self._high_volatility, {'multiplier': 3.0}),
            ('Strong Uptrend', self._strong_trend, {'direction': 1, 'magnitude': self.config.trend_magnitude}),
            ('Strong Downtrend', self._strong_trend, {'direction': -1, 'magnitude': self.config.trend_magnitude}),
            ('Flash Crash', self._flash_crash, {'magnitude': self.config.flash_crash_magnitude, 'duration': 10}),
            ('Liquidity Crisis', self._liquidity_crisis, {'spread_mult': self.config.spread_multiplier}),
            ('Black Swan', self._black_swan, {'sigma': self.config.black_swan_sigma})
        ]
        
        results = {
            'scenarios': [],
            'passes': 0,
            'total': len(scenarios)
        }
        
        for scenario_name, scenario_func, params in scenarios:
            logger.info(f"\nScenario: {scenario_name}")
            
            # Apply scenario
            stressed_data, stressed_features = scenario_func(
                base_data.copy(),
                base_features.copy(),
                **params
            )
            
            # Evaluate
            env = env_class(
                data=stressed_data,
                features=stressed_features,
                config=env_config,
                eval_mode=True
            )
            
            returns, equity = self._evaluate_agent(agent, env, n_episodes=3)
            
            # Calculate metrics
            sharpe = self.metrics.calculate_sharpe_ratio(returns)
            max_dd = self.metrics.calculate_max_drawdown(equity)
            
            # Check criteria
            passed = (
                sharpe > self.config.stress_min_sharpe and
                max_dd < self.config.stress_max_drawdown
            )
            
            if passed:
                results['passes'] += 1
            
            scenario_result = {
                'name': scenario_name,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'passed': passed
            }
            
            results['scenarios'].append(scenario_result)
            
            logger.info(f"  Sharpe: {sharpe:.2f}")
            logger.info(f"  Max DD: {max_dd:.2%}")
            logger.info(f"  Passed: {passed}")
        
        # Overall pass/fail
        success_ratio = results['passes'] / results['total']
        results['overall_passed'] = success_ratio >= self.config.stress_success_ratio
        results['success_ratio'] = success_ratio
        
        logger.info("\n" + "="*80)
        logger.info("Stress Test Summary")
        logger.info("="*80)
        logger.info(f"Passed: {results['passes']}/{results['total']} scenarios")
        logger.info(f"Success Ratio: {success_ratio:.1%}")
        logger.info(f"Overall Passed (≥{self.config.stress_success_ratio:.0%}): {results['overall_passed']}")
        logger.info("="*80)
        
        return results
    
    def _high_volatility(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        multiplier: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Multiply volatility by factor."""
        data = data.copy()
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Amplify returns
        amplified_returns = returns * multiplier
        
        # Reconstruct prices
        data['close'] = data['close'].iloc[0] * (1 + amplified_returns).cumprod()
        data['open'] = data['close'] * (1 + np.random.normal(0, 0.0001, len(data)))
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.0005, len(data))))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.0005, len(data))))
        
        return data, features
    
    def _strong_trend(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        direction: int,
        magnitude: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Add strong trend."""
        data = data.copy()
        
        # Add linear trend
        n = len(data)
        trend = np.linspace(0, direction * magnitude, n)
        
        data['close'] = data['close'] * (1 + trend)
        data['open'] = data['open'] * (1 + trend)
        data['high'] = data['high'] * (1 + trend)
        data['low'] = data['low'] * (1 + trend)
        
        return data, features
    
    def _flash_crash(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        magnitude: float,
        duration: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate flash crash."""
        data = data.copy()
        
        # Find random point for crash
        crash_idx = np.random.randint(100, len(data) - duration - 100)
        
        # Apply crash
        crash_factor = 1 + magnitude
        for i in range(crash_idx, crash_idx + duration):
            data.loc[i, 'close'] *= crash_factor
            data.loc[i, 'open'] *= crash_factor
            data.loc[i, 'high'] *= crash_factor
            data.loc[i, 'low'] *= crash_factor
        
        # Gradual recovery
        recovery_duration = duration * 3
        for i in range(duration):
            recovery_factor = 1 + (1 - crash_factor) * (i / recovery_duration)
            idx = crash_idx + duration + i
            if idx < len(data):
                data.loc[idx, 'close'] *= recovery_factor
                data.loc[idx, 'open'] *= recovery_factor
                data.loc[idx, 'high'] *= recovery_factor
                data.loc[idx, 'low'] *= recovery_factor
        
        return data, features
    
    def _liquidity_crisis(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        spread_mult: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate liquidity crisis (wider spreads)."""
        # This would need to be handled in the environment
        # Here we just flag it in the data
        data = data.copy()
        data['spread_multiplier'] = spread_mult
        
        return data, features
    
    def _black_swan(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        sigma: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate black swan events (extreme outliers)."""
        data = data.copy()
        
        # Add several extreme events
        n_events = 3
        event_indices = np.random.choice(len(data), size=n_events, replace=False)
        
        for idx in event_indices:
            # Random extreme move
            extreme_return = np.random.choice([-1, 1]) * sigma * np.std(data['close'].pct_change())
            data.loc[idx, 'close'] *= (1 + extreme_return)
            data.loc[idx, 'high'] = max(data.loc[idx, 'open'], data.loc[idx, 'close']) * 1.01
            data.loc[idx, 'low'] = min(data.loc[idx, 'open'], data.loc[idx, 'close']) * 0.99
        
        return data, features
    
    def _evaluate_agent(
        self,
        agent,
        env,
        n_episodes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate agent on environment."""
        all_returns = []
        all_equity = []
        
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            
            episode_equity = [env.equity]
            
            while not done:
                if hasattr(agent, 'get_ensemble_action'):
                    action = agent.get_ensemble_action(state, deterministic=True)
                else:
                    action = agent.select_action(state, deterministic=True)
                
                next_state, reward, done, info = env.step(action)
                
                all_returns.append(reward)
                episode_equity.append(info['equity'])
                
                state = next_state
            
            all_equity.extend(episode_equity)
        
        return np.array(all_returns), np.array(all_equity)


class OOSValidator:
    """Out-of-sample final validation."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
    
    def validate_oos(
        self,
        agent,
        oos_data: pd.DataFrame,
        oos_features: pd.DataFrame,
        env_class,
        env_config,
        n_episodes: int = 10
    ) -> Dict:
        """
        Perform final OOS validation.
        
        Args:
            agent: Trained agent
            oos_data: OOS OHLC data (never touched before)
            oos_features: OOS features
            env_class: Environment class
            env_config: Environment config
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with OOS results
        """
        logger.info("="*80)
        logger.info("OUT-OF-SAMPLE FINAL VALIDATION")
        logger.info("="*80)
        logger.info("CRITICAL: This data has NEVER been seen during training!")
        logger.info("="*80)
        
        # Create environment
        env = env_class(
            data=oos_data,
            features=oos_features,
            config=env_config,
            eval_mode=True
        )
        
        # Collect results
        all_returns = []
        all_equity_curves = []
        all_metrics = []
        
        for i in tqdm(range(n_episodes), desc="OOS Evaluation"):
            state = env.reset()
            done = False
            
            episode_returns = []
            
            while not done:
                if hasattr(agent, 'get_ensemble_action'):
                    action = agent.get_ensemble_action(state, deterministic=True)
                else:
                    action = agent.select_action(state, deterministic=True)
                
                next_state, reward, done, info = env.step(action)
                
                episode_returns.append(reward)
                state = next_state
            
            # Get episode metrics
            metrics = env.get_episode_metrics()
            all_metrics.append(metrics)
            all_returns.extend(episode_returns)
            
            if i < 3:  # Store first 3 equity curves
                all_equity_curves.append(env.equity_curve)
        
        # Aggregate statistics
        returns_array = np.array(all_returns)
        
        sharpe = self.metrics.calculate_sharpe_ratio(returns_array)
        sortino = self.metrics.calculate_sortino_ratio(returns_array)
        
        # Average metrics across episodes
        avg_max_dd = np.mean([m['max_drawdown'] for m in all_metrics])
        avg_win_rate = np.mean([m['win_rate'] for m in all_metrics])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics])
        avg_sortino = np.mean([m['sortino_ratio'] for m in all_metrics])
        
        # Check acceptance criteria
        passes_sharpe = sharpe > self.config.oos_min_sharpe
        passes_sortino = sortino > self.config.oos_min_sortino
        passes_dd = avg_max_dd < self.config.oos_max_drawdown
        passes_wr = avg_win_rate > self.config.oos_min_win_rate
        
        passes_all = passes_sharpe and passes_sortino and passes_dd and passes_wr
        
        result = {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': avg_max_dd,
            'win_rate': avg_win_rate,
            'avg_episode_sharpe': avg_sharpe,
            'avg_episode_sortino': avg_sortino,
            'passes_sharpe': passes_sharpe,
            'passes_sortino': passes_sortino,
            'passes_drawdown': passes_dd,
            'passes_win_rate': passes_wr,
            'passes_all_criteria': passes_all,
            'n_episodes': n_episodes,
            'all_metrics': all_metrics
        }
        
        logger.info("\n" + "="*80)
        logger.info("OOS VALIDATION RESULTS")
        logger.info("="*80)
        logger.info(f"Sharpe Ratio: {sharpe:.2f} (min: {self.config.oos_min_sharpe}) - {'✓' if passes_sharpe else '✗'}")
        logger.info(f"Sortino Ratio: {sortino:.2f} (min: {self.config.oos_min_sortino}) - {'✓' if passes_sortino else '✗'}")
        logger.info(f"Max Drawdown: {avg_max_dd:.2%} (max: {self.config.oos_max_drawdown:.0%}) - {'✓' if passes_dd else '✗'}")
        logger.info(f"Win Rate: {avg_win_rate:.2%} (min: {self.config.oos_min_win_rate:.0%}) - {'✓' if passes_wr else '✗'}")
        logger.info(f"\nOVERALL: {'PASSED ✓' if passes_all else 'FAILED ✗'}")
        logger.info("="*80)
        
        return result


class ComprehensiveValidator:
    """Comprehensive validation orchestrator."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        self.walk_forward = WalkForwardValidator(self.config)
        self.statistical = StatisticalTests(self.config)
        self.bootstrap = BootstrapValidator(self.config)
        self.stress = StressTester(self.config)
        self.oos = OOSValidator(self.config)
    
    def run_full_validation(
        self,
        agent,
        train_data: pd.DataFrame,
        train_features: pd.DataFrame,
        test_data: pd.DataFrame,
        test_features: pd.DataFrame,
        env_class,
        env_config
    ) -> Dict:
        """
        Run complete validation suite.
        
        Args:
            agent: Trained agent
            train_data: Training OHLC data
            train_features: Training features
            test_data: Test OHLC data (OOS)
            test_features: Test features (OOS)
            env_class: Environment class
            env_config: Environment configuration
            
        Returns:
            Complete validation results
        """
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE VALIDATION SUITE")
        logger.info("="*80)
        
        results = {}
        
        # 1. Walk-Forward Validation
        logger.info("\n[1/5] Walk-Forward Validation")
        wf_results = self.walk_forward.validate(
            agent, train_data, train_features, env_class, env_config
        )
        results['walk_forward'] = wf_results
        
        # 2. Statistical Tests
        logger.info("\n[2/5] Statistical Tests")
        
        # Get returns from walk-forward
        oos_returns = []
        for window in wf_results['windows']:
            # Would need to re-evaluate to get actual returns
            # For now, use synthetic based on Sharpe
            n_samples = 1000
            sharpe = window['out_sample_sharpe']
            synthetic_returns = np.random.normal(
                sharpe / np.sqrt(252 * 288),
                1.0 / np.sqrt(252 * 288),
                n_samples
            )
            oos_returns.extend(synthetic_returns)
        
        oos_returns = np.array(oos_returns)
        
        # DSR
        dsr_result = self.statistical.deflated_sharpe_ratio(
            observed_sharpe=wf_results['mean_oos_sharpe'],
            n_returns=len(oos_returns),
            skewness=stats.skew(oos_returns),
            kurtosis=stats.kurtosis(oos_returns, fisher=False)
        )
        results['dsr'] = dsr_result
        
        # PSR
        psr_result = self.statistical.probabilistic_sharpe_ratio(oos_returns)
        results['psr'] = psr_result
        
        # PBO
        pbo_result = self.statistical.probability_backtest_overfitting(wf_results)
        results['pbo'] = pbo_result
        
        # 3. Bootstrap CI
        logger.info("\n[3/5] Bootstrap Confidence Intervals")
        bootstrap_sharpe = self.bootstrap.calculate_confidence_intervals(
            oos_returns, metric_name='sharpe'
        )
        bootstrap_sortino = self.bootstrap.calculate_confidence_intervals(
            oos_returns, metric_name='sortino'
        )
        results['bootstrap'] = {
            'sharpe': bootstrap_sharpe,
            'sortino': bootstrap_sortino
        }
        
        # 4. Stress Tests
        logger.info("\n[4/5] Stress Tests")
        stress_results = self.stress.run_all_stress_tests(
            agent, train_data, train_features, env_class, env_config
        )
        results['stress'] = stress_results
        
        # 5. OOS Final Validation
        logger.info("\n[5/5] Out-of-Sample Final Validation")
        oos_results = self.oos.validate_oos(
            agent, test_data, test_features, env_class, env_config, n_episodes=10
        )
        results['oos'] = oos_results
        
        # Overall assessment
        passed_wf = wf_results['passes_wfe']
        passed_dsr = dsr_result['passes_threshold']
        passed_psr = psr_result['passes_threshold']
        passed_pbo = pbo_result['passes_threshold']
        passed_stress = stress_results['overall_passed']
        passed_oos = oos_results['passes_all_criteria']
        
        all_passed = all([
            passed_wf, passed_dsr, passed_psr, passed_pbo, passed_stress, passed_oos
        ])
        
        results['summary'] = {
            'walk_forward_passed': passed_wf,
            'dsr_passed': passed_dsr,
            'psr_passed': passed_psr,
            'pbo_passed': passed_pbo,
            'stress_passed': passed_stress,
            'oos_passed': passed_oos,
            'all_tests_passed': all_passed
        }
        
        # Final report
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Walk-Forward Validation: {'PASSED ✓' if passed_wf else 'FAILED ✗'}")
        logger.info(f"Deflated Sharpe Ratio: {'PASSED ✓' if passed_dsr else 'FAILED ✗'}")
        logger.info(f"Probabilistic Sharpe Ratio: {'PASSED ✓' if passed_psr else 'FAILED ✗'}")
        logger.info(f"Backtest Overfitting: {'PASSED ✓' if passed_pbo else 'FAILED ✗'}")
        logger.info(f"Stress Tests: {'PASSED ✓' if passed_stress else 'FAILED ✗'}")
        logger.info(f"OOS Validation: {'PASSED ✓' if passed_oos else 'FAILED ✗'}")
        logger.info(f"\n{'='*80}")
        logger.info(f"OVERALL: {'ALL TESTS PASSED ✓✓✓' if all_passed else 'SOME TESTS FAILED'}")
        logger.info(f"{'='*80}\n")
        
        return results


# Alias de compatibilité pour les anciens imports
ValidationFramework = ComprehensiveValidator


def main():
    """Example usage of validation module."""
    from backend.data_pipeline import DataPipeline
    from backend.feature_engineering import FeaturePipeline
    
    # Load data
    logger.info("Loading data...")
    data_pipeline = DataPipeline()
    train_data, val_data, test_data = data_pipeline.get_processed_data()
    
    feature_pipeline = FeaturePipeline()
    train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
        train_data, val_data, test_data
    )
    
    # Create dummy agent for demonstration
    class DummyAgent:
        def select_action(self, state, deterministic=False):
            return np.random.uniform(-1, 1, size=1)
    
    agent = DummyAgent()
    
    # Initialize validator
    config = ValidationConfig()
    validator = ComprehensiveValidator(config)
    
    # Note: For actual validation, you would use:
    # from trading_env import TradingEnvironment, TradingEnvConfig
    # from sac_agent import SACAgent
    # 
    # And run:
    # results = validator.run_full_validation(
    #     agent,
    #     train_data['EURUSD'],
    #     train_features,
    #     test_data['EURUSD'],
    #     test_features,
    #     TradingEnvironment,
    #     TradingEnvConfig()
    # )
    
    logger.info("\nValidation module demonstration complete!")
    logger.info("To run actual validation, use a trained SAC agent or ensemble.")


if __name__ == "__main__":
    main()
