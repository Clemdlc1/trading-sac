"""
SAC EUR/USD Trading System - HMM Regime Detection
==================================================

This module handles regime detection using Hidden Markov Models to identify
low and high volatility market conditions.

Features:
- Gaussian HMM with 2 states (low_volatility, high_volatility)
- Full covariance matrix
- Expectation-Maximization training
- Viterbi decoding for historical sequences
- Forward algorithm for real-time regime probabilities
- Model persistence (pickle)
- Automatic regime identification

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HMMConfig:
    """Configuration for HMM regime detection."""
    
    # HMM parameters
    n_states: int = 2
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42
    
    # Feature parameters
    volatility_window: int = 20  # Rolling window for realized volatility
    
    # Model paths
    models_dir: Path = Path("models/hmm")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)


class HMMFeatureExtractor:
    """Extract features for HMM regime detection."""
    
    @staticmethod
    def extract_hmm_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Extract 3 features for HMM:
        1. Returns EUR/USD
        2. Realized volatility (rolling std of returns)
        3. Range (high - low) / close
        
        Args:
            df: DataFrame with OHLC data
            window: Window for realized volatility calculation
            
        Returns:
            DataFrame with HMM features
        """
        features = pd.DataFrame(index=df.index)
        
        # Feature 1: Returns
        features['returns'] = df['close'].pct_change()
        
        # Feature 2: Realized volatility (rolling std of returns)
        features['realized_vol'] = features['returns'].rolling(
            window=window, min_periods=window
        ).std()
        
        # Feature 3: Range normalized by close
        features['range'] = (df['high'] - df['low']) / df['close']
        
        # Remove NaN values
        features = features.dropna()
        
        return features


class HMMRegimeDetector:
    """Hidden Markov Model for regime detection."""
    
    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.high_vol_state = None
        self.low_vol_state = None
        self.feature_extractor = HMMFeatureExtractor()
        
    def _identify_states(self) -> Tuple[int, int]:
        """
        Identify which state corresponds to high/low volatility.
        
        High volatility state = state with largest trace of covariance matrix
        
        Returns:
            Tuple of (high_vol_state_idx, low_vol_state_idx)
        """
        if self.model is None:
            raise ValueError("Model must be trained before identifying states")
        
        # Calculate trace of covariance matrices
        traces = []
        for i in range(self.config.n_states):
            cov_matrix = self.model.covars_[i]
            trace = np.trace(cov_matrix)
            traces.append(trace)
            logger.info(f"State {i} covariance trace: {trace:.6f}")
        
        # High volatility = state with largest trace
        high_vol_state = int(np.argmax(traces))
        low_vol_state = 1 - high_vol_state
        
        logger.info(f"Identified: High volatility = State {high_vol_state}, "
                   f"Low volatility = State {low_vol_state}")
        
        return high_vol_state, low_vol_state
    
    def train(self, data: pd.DataFrame, verbose: bool = True) -> 'HMMRegimeDetector':
        """
        Train HMM on historical data.
        
        Args:
            data: DataFrame with OHLC data (entire training set)
            verbose: Whether to print training progress
            
        Returns:
            Self for chaining
        """
        logger.info("="*80)
        logger.info("Training HMM Regime Detector")
        logger.info("="*80)
        
        # Extract features
        logger.info("Extracting HMM features...")
        features = self.feature_extractor.extract_hmm_features(
            data, window=self.config.volatility_window
        )
        
        logger.info(f"Total samples for training: {len(features)}")
        logger.info(f"Feature columns: {list(features.columns)}")
        
        # Scale features
        logger.info("Scaling features...")
        features_scaled = self.scaler.fit_transform(features.values)
        
        # Initialize and train HMM
        logger.info(f"Initializing Gaussian HMM with {self.config.n_states} states...")
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
            verbose=verbose
        )
        
        logger.info("Training HMM with Expectation-Maximization...")
        self.model.fit(features_scaled)
        
        # Identify states
        logger.info("Identifying regime states...")
        self.high_vol_state, self.low_vol_state = self._identify_states()
        
        # Calculate convergence statistics
        logger.info(f"Training converged after {self.model.monitor_.iter} iterations")
        logger.info(f"Final log-likelihood: {self.model.score(features_scaled):.2f}")
        
        # Log learned parameters
        logger.info("\nLearned Parameters:")
        logger.info(f"Start probabilities: {self.model.startprob_}")
        logger.info(f"Transition matrix:\n{self.model.transmat_}")
        
        logger.info("\n" + "="*80)
        logger.info("HMM Training Complete!")
        logger.info("="*80)
        
        return self
    
    def predict_sequence(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regime sequence using Viterbi algorithm.
        
        This is used for historical analysis where we want the most likely
        sequence of states.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Array of state predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Extract and scale features
        features = self.feature_extractor.extract_hmm_features(
            data, window=self.config.volatility_window
        )
        features_scaled = self.scaler.transform(features.values)
        
        # Predict using Viterbi algorithm
        states = self.model.predict(features_scaled)
        
        return states
    
    def predict_proba(self, data: pd.DataFrame, recent_only: int = 100) -> np.ndarray:
        """
        Predict regime probabilities using Forward algorithm.
        
        This is used for real-time trading where we want probabilities
        rather than hard state assignments.
        
        Args:
            data: DataFrame with OHLC data
            recent_only: Use only the last N observations for computation efficiency
            
        Returns:
            Array of shape (n_samples, n_states) with regime probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Extract and scale features
        features = self.feature_extractor.extract_hmm_features(
            data, window=self.config.volatility_window
        )
        
        # Use only recent data for efficiency
        if recent_only is not None and len(features) > recent_only:
            features = features.iloc[-recent_only:]
        
        features_scaled = self.scaler.transform(features.values)
        
        # Predict probabilities using Forward algorithm
        probs = self.model.predict_proba(features_scaled)
        
        return probs
    
    def get_current_regime(
        self, 
        data: pd.DataFrame, 
        recent_only: int = 100,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Get current regime information for trading decisions.
        
        Args:
            data: DataFrame with OHLC data
            recent_only: Use only the last N observations
            threshold: Probability threshold for regime classification
            
        Returns:
            Dictionary with regime information:
            - 'regime': 'high_vol' or 'low_vol'
            - 'probs': [P(low_vol), P(high_vol)]
            - 'state': 0 or 1
            - 'confidence': Max probability
            - 'is_low_vol': Binary flag
            - 'is_high_vol': Binary flag
        """
        # Get probabilities
        probs = self.predict_proba(data, recent_only=recent_only)
        
        # Get current (last) probabilities
        current_probs = probs[-1]
        
        # Determine regime
        prob_high_vol = current_probs[self.high_vol_state]
        prob_low_vol = current_probs[self.low_vol_state]
        
        # Classify based on threshold
        if prob_high_vol > threshold:
            regime = 'high_vol'
            state = self.high_vol_state
            is_high_vol = 1.0
            is_low_vol = 0.0
        else:
            regime = 'low_vol'
            state = self.low_vol_state
            is_high_vol = 0.0
            is_low_vol = 1.0
        
        return {
            'regime': regime,
            'probs': [prob_low_vol, prob_high_vol],
            'state': state,
            'confidence': max(prob_high_vol, prob_low_vol),
            'is_low_vol': is_low_vol,
            'is_high_vol': is_high_vol
        }
    
    def analyze_regimes(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze regime characteristics over historical data.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary with regime analysis
        """
        if self.model is None:
            raise ValueError("Model must be trained before analysis")
        
        # Get state sequence
        states = self.predict_sequence(data)
        
        # Extract features
        features = self.feature_extractor.extract_hmm_features(
            data, window=self.config.volatility_window
        )
        
        # Align features with states
        features = features.iloc[:len(states)]
        
        # Calculate statistics for each regime
        analysis = {
            'low_vol': {},
            'high_vol': {},
            'transitions': {},
            'duration': {}
        }
        
        # Low volatility regime statistics
        low_vol_mask = states == self.low_vol_state
        if low_vol_mask.sum() > 0:
            analysis['low_vol'] = {
                'count': int(low_vol_mask.sum()),
                'percentage': float(low_vol_mask.sum() / len(states) * 100),
                'mean_returns': float(features.loc[low_vol_mask, 'returns'].mean()),
                'mean_volatility': float(features.loc[low_vol_mask, 'realized_vol'].mean()),
                'mean_range': float(features.loc[low_vol_mask, 'range'].mean())
            }
        
        # High volatility regime statistics
        high_vol_mask = states == self.high_vol_state
        if high_vol_mask.sum() > 0:
            analysis['high_vol'] = {
                'count': int(high_vol_mask.sum()),
                'percentage': float(high_vol_mask.sum() / len(states) * 100),
                'mean_returns': float(features.loc[high_vol_mask, 'returns'].mean()),
                'mean_volatility': float(features.loc[high_vol_mask, 'realized_vol'].mean()),
                'mean_range': float(features.loc[high_vol_mask, 'range'].mean())
            }
        
        # Transition analysis
        transitions = np.sum(states[1:] != states[:-1])
        analysis['transitions'] = {
            'total': int(transitions),
            'rate': float(transitions / len(states) * 100)
        }
        
        # Average regime duration
        regime_changes = np.where(states[1:] != states[:-1])[0]
        if len(regime_changes) > 0:
            durations = np.diff(np.concatenate([[0], regime_changes, [len(states)]]))
            analysis['duration'] = {
                'mean': float(durations.mean()),
                'median': float(np.median(durations)),
                'min': int(durations.min()),
                'max': int(durations.max())
            }
        
        return analysis
    
    def save(self, filename: str = "hmm_model.pkl") -> Path:
        """
        Save trained model to disk.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        output_path = self.config.models_dir / filename
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'high_vol_state': self.high_vol_state,
            'low_vol_state': self.low_vol_state,
            'config': self.config
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {output_path}")
        return output_path
    
    def load(self, filename: str = "hmm_model.pkl") -> 'HMMRegimeDetector':
        """
        Load trained model from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Self for chaining
        """
        input_path = self.config.models_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Model file not found: {input_path}")
        
        with open(input_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.high_vol_state = model_data['high_vol_state']
        self.low_vol_state = model_data['low_vol_state']
        self.config = model_data['config']
        
        logger.info(f"Model loaded from {input_path}")
        logger.info(f"High volatility state: {self.high_vol_state}")
        logger.info(f"Low volatility state: {self.low_vol_state}")
        
        return self


class HMMIntegration:
    """Integration utilities for HMM with SAC agents."""
    
    def __init__(self, hmm_detector: HMMRegimeDetector):
        self.hmm_detector = hmm_detector
    
    def augment_state_for_agents_1_2(
        self, 
        state: np.ndarray, 
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Augment state with regime features for Agents 1 and 2.
        
        Adds 2 binary features:
        - is_low_vol
        - is_high_vol
        
        Args:
            state: Original state vector (30 features)
            data: Recent OHLC data for regime detection
            
        Returns:
            Augmented state vector (32 features)
        """
        # Get current regime
        regime_info = self.hmm_detector.get_current_regime(data)
        
        # Add regime features
        regime_features = np.array([
            regime_info['is_low_vol'],
            regime_info['is_high_vol']
        ])
        
        # Concatenate
        augmented_state = np.concatenate([state, regime_features])
        
        return augmented_state
    
    def get_regime_for_agent_3(self, data: pd.DataFrame) -> str:
        """
        Get current regime for Agent 3 Q-function selection.
        
        Agent 3 uses separate Q-functions for each regime.
        
        Args:
            data: Recent OHLC data for regime detection
            
        Returns:
            Regime string: 'low_vol' or 'high_vol'
        """
        regime_info = self.hmm_detector.get_current_regime(data)
        return regime_info['regime']
    
    def should_adjust_parameters(
        self, 
        data: pd.DataFrame,
        previous_regime: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Determine if trading parameters should be adjusted based on regime.
        
        According to spec:
        - High volatility detected: α → α × 1.5 (more exploration)
        - Regime transition: position_sizing → position_sizing × 0.7
        - Extreme volatility (top 5%): disable trading
        
        Args:
            data: Recent OHLC data
            previous_regime: Previous regime for transition detection
            
        Returns:
            Dictionary with adjustment recommendations
        """
        regime_info = self.hmm_detector.get_current_regime(data)
        current_regime = regime_info['regime']
        
        adjustments = {
            'alpha_multiplier': 1.0,
            'position_multiplier': 1.0,
            'disable_trading': False,
            'reason': None,
            'regime': current_regime,
            'confidence': regime_info['confidence']
        }
        
        # High volatility: increase exploration
        if current_regime == 'high_vol':
            adjustments['alpha_multiplier'] = 1.5
            adjustments['reason'] = 'High volatility regime detected'
        
        # Regime transition: reduce position sizing
        if previous_regime is not None and previous_regime != current_regime:
            adjustments['position_multiplier'] = 0.7
            adjustments['reason'] = f'Regime transition: {previous_regime} → {current_regime}'
        
        # Check for extreme volatility (top 5%)
        # This would require historical statistics - simplified here
        if regime_info['confidence'] > 0.95 and current_regime == 'high_vol':
            adjustments['disable_trading'] = True
            adjustments['reason'] = 'Extreme volatility detected (top 5%)'
        
        return adjustments


def main():
    """Example usage of HMM regime detector."""
    from backend.data_pipeline import DataPipeline
    
    # Load data
    logger.info("Loading processed data...")
    data_pipeline = DataPipeline()
    train_data, val_data, test_data = data_pipeline.get_processed_data()
    
    # Get EUR/USD data
    eurusd_train = train_data['EURUSD']
    eurusd_val = val_data['EURUSD']
    
    # Initialize and train HMM
    logger.info("\nInitializing HMM detector...")
    hmm_detector = HMMRegimeDetector()
    
    # Train on entire training set
    hmm_detector.train(eurusd_train, verbose=True)
    
    # Save model
    hmm_detector.save("hmm_model.pkl")
    
    # Analyze regimes on training data
    logger.info("\nAnalyzing regimes on training data...")
    train_analysis = hmm_detector.analyze_regimes(eurusd_train)
    
    print("\n" + "="*80)
    print("Regime Analysis - Training Data")
    print("="*80)
    print(f"\nLow Volatility Regime:")
    for key, value in train_analysis['low_vol'].items():
        print(f"  {key}: {value}")
    
    print(f"\nHigh Volatility Regime:")
    for key, value in train_analysis['high_vol'].items():
        print(f"  {key}: {value}")
    
    print(f"\nTransitions:")
    for key, value in train_analysis['transitions'].items():
        print(f"  {key}: {value}")
    
    print(f"\nRegime Duration (bars):")
    for key, value in train_analysis['duration'].items():
        print(f"  {key}: {value}")
    
    # Test on validation data
    logger.info("\nTesting on validation data...")
    val_analysis = hmm_detector.analyze_regimes(eurusd_val)
    
    print("\n" + "="*80)
    print("Regime Analysis - Validation Data")
    print("="*80)
    print(f"\nLow Volatility Regime:")
    for key, value in val_analysis['low_vol'].items():
        print(f"  {key}: {value}")
    
    print(f"\nHigh Volatility Regime:")
    for key, value in val_analysis['high_vol'].items():
        print(f"  {key}: {value}")
    
    # Test real-time regime detection
    logger.info("\nTesting real-time regime detection...")
    current_regime = hmm_detector.get_current_regime(eurusd_val, recent_only=100)
    
    print("\n" + "="*80)
    print("Current Regime (Validation Data - Last Point)")
    print("="*80)
    print(f"Regime: {current_regime['regime']}")
    print(f"State: {current_regime['state']}")
    print(f"Probabilities: {current_regime['probs']}")
    print(f"Confidence: {current_regime['confidence']:.2%}")
    print(f"Is Low Vol: {current_regime['is_low_vol']}")
    print(f"Is High Vol: {current_regime['is_high_vol']}")
    
    # Test integration utilities
    logger.info("\nTesting integration utilities...")
    integration = HMMIntegration(hmm_detector)
    
    # Simulate state augmentation for Agents 1 and 2
    dummy_state = np.random.randn(30)  # 30 original features
    augmented_state = integration.augment_state_for_agents_1_2(dummy_state, eurusd_val)
    
    print(f"\nState Augmentation:")
    print(f"  Original state shape: {dummy_state.shape}")
    print(f"  Augmented state shape: {augmented_state.shape}")
    print(f"  Regime features: {augmented_state[-2:]}")
    
    # Test parameter adjustments
    adjustments = integration.should_adjust_parameters(eurusd_val)
    
    print(f"\nParameter Adjustments:")
    for key, value in adjustments.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("HMM Regime Detection Demo Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
