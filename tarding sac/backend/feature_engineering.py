"""
SAC EUR/USD Trading System - Feature Engineering
================================================

This module handles feature calculation, normalization, and validation
for the algorithmic trading system.

Features:
- 30 technical features across 5 groups
- Zero look-ahead bias (all calculations use shift(1))
- Expanding window normalization
- DXY synthetic index calculation
- Temporal encoding (cyclical)
- HDF5 persistence with metadata

CRITICAL: All features respect causality - no future data leakage!

Author: SAC EUR/USD Project
Version: 2.0
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Feature groups
    eurusd_features: int = 10
    dxy_features: int = 5
    cross_features: int = 6
    risk_features: int = 2
    temporal_features: int = 7
    total_features: int = 30
    
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    correlation_period: int = 60  # days for correlation (60 days * 288 bars/day)
    
    # Normalization parameters
    clip_min: float = -5.0
    clip_max: float = 5.0
    epsilon: float = 1e-8
    
    # DXY synthetic index weights (from spec)
    dxy_constant: float = 50.14348112
    dxy_weights: Dict[str, float] = field(default_factory=lambda: {
        'EURUSD': -0.576,  # Inverted
        'USDJPY': 0.136,
        'GBPUSD': -0.119,  # Inverted
        'USDCAD': 0.091,
        'USDSEK': 0.042,
        'USDCHF': 0.036
    })
    
    # File paths
    normalized_data_dir: Path = Path("data/normalized")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.normalized_data_dir.mkdir(parents=True, exist_ok=True)


class TechnicalIndicators:
    """Calculate technical indicators with zero look-ahead bias."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate percentage returns.
        
        CRITICAL: Uses shift(1) to avoid look-ahead bias.
        
        Args:
            prices: Price series
            periods: Number of periods for return calculation
            
        Returns:
            Returns series
        """
        # Shift prices to avoid look-ahead
        prices_shifted = prices.shift(1)
        returns = (prices - prices_shifted) / (prices_shifted + 1e-8)
        return returns
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        CRITICAL: All calculations use historical data only.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values (normalized to [0, 1])
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate rolling averages using expanding window to avoid look-ahead
        # Then convert to rolling with min_periods
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Normalize to [0, 1]
        rsi_normalized = rsi / 100.0
        
        # Shift to avoid look-ahead bias
        return rsi_normalized.shift(1)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """
        Calculate MACD histogram.
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            MACD histogram
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        # Shift to avoid look-ahead bias
        return histogram.shift(1)
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR values
        """
        # Calculate true range
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR using rolling mean
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        # Shift to avoid look-ahead bias
        return atr.shift(1)
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period, min_periods=period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=period, min_periods=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Shift to avoid look-ahead bias
        return upper_band.shift(1), middle_band.shift(1), lower_band.shift(1)
    
    @staticmethod
    def calculate_parkinson_volatility(high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Calculate Parkinson volatility estimator.
        
        Formula: sqrt(log(high/low)² / (4*log(2)))
        
        Args:
            high: High prices
            low: Low prices
            
        Returns:
            Parkinson volatility
        """
        log_hl = np.log(high / (low + 1e-8))
        parkinson = np.sqrt(log_hl ** 2 / (4 * np.log(2)))
        
        # Shift to avoid look-ahead bias
        return parkinson.shift(1)


class DXYCalculator:
    """Calculate DXY (US Dollar Index) synthetic index."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def calculate_dxy(self, data_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate DXY synthetic index.
        
        Formula:
        DXY = 50.14348112 × (1/EURUSD)^0.576 × USDJPY^0.136 × (1/GBPUSD)^0.119 
                           × USDCAD^0.091 × USDSEK^0.042 × USDCHF^0.036
        
        Args:
            data_dict: Dictionary of DataFrames with close prices
            
        Returns:
            DXY index series
        """
        # Extract close prices for each pair
        eurusd = data_dict['EURUSD']['close']
        usdjpy = data_dict['USDJPY']['close']
        gbpusd = data_dict['GBPUSD']['close']
        usdcad = data_dict['USDCAD']['close']
        usdsek = data_dict['USDSEK']['close']
        usdchf = data_dict['USDCHF']['close']
        
        # Calculate DXY
        dxy = self.config.dxy_constant * \
              (1 / eurusd) ** 0.576 * \
              usdjpy ** 0.136 * \
              (1 / gbpusd) ** 0.119 * \
              usdcad ** 0.091 * \
              usdsek ** 0.042 * \
              usdchf ** 0.036
        
        return dxy


class FeatureEngineer:
    """Main feature engineering class."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.tech_indicators = TechnicalIndicators()
        self.dxy_calculator = DXYCalculator(self.config)
    
    def calculate_eurusd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EUR/USD direct features (Group 1: 10 features).
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=df.index)
        
        logger.info("Calculating EUR/USD features...")
        
        # Returns multi-timeframe (4 features)
        features['return_5min'] = self.tech_indicators.calculate_returns(df['close'], periods=1)
        features['return_1h'] = self.tech_indicators.calculate_returns(df['close'], periods=12)
        features['return_4h'] = self.tech_indicators.calculate_returns(df['close'], periods=48)
        features['return_1d'] = self.tech_indicators.calculate_returns(df['close'], periods=288)
        
        # Momentum (2 features)
        features['rsi_14'] = self.tech_indicators.calculate_rsi(df['close'], period=14)
        features['macd_histogram'] = self.tech_indicators.calculate_macd(
            df['close'], fast=12, slow=26, signal=9
        )
        
        # Volatility (3 features)
        atr = self.tech_indicators.calculate_atr(df['high'], df['low'], df['close'], period=14)
        features['atr_14'] = atr / (df['close'].shift(1) + 1e-8)  # Normalize by price
        
        features['parkinson_vol'] = self.tech_indicators.calculate_parkinson_volatility(
            df['high'], df['low']
        )
        
        # Bollinger Bands width
        bb_upper, bb_middle, bb_lower = self.tech_indicators.calculate_bollinger_bands(
            df['close'], period=20, std_dev=2.0
        )
        features['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
        
        # Microstructure (1 feature)
        features['hl_range'] = (df['high'] - df['low']) / (df['close'].shift(1) + 1e-8)
        features['hl_range'] = features['hl_range'].shift(1)  # Additional shift for causality
        
        return features
    
    def calculate_dxy_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate DXY synthetic index features (Group 2: 5 features).
        
        Args:
            data_dict: Dictionary of DataFrames for all pairs
            
        Returns:
            DataFrame with DXY features
        """
        logger.info("Calculating DXY features...")
        
        # Calculate DXY index
        dxy = self.dxy_calculator.calculate_dxy(data_dict)
        
        # Create DataFrame for DXY features
        features = pd.DataFrame(index=dxy.index)
        
        # DXY returns (2 features)
        features['dxy_return_1h'] = self.tech_indicators.calculate_returns(dxy, periods=12)
        features['dxy_return_4h'] = self.tech_indicators.calculate_returns(dxy, periods=48)
        
        # DXY RSI (1 feature)
        features['dxy_rsi_14'] = self.tech_indicators.calculate_rsi(dxy, period=14)
        
        # DXY ATR (1 feature)
        # For DXY, we'll approximate high/low using price variations
        dxy_high = dxy * 1.001  # Approximate
        dxy_low = dxy * 0.999
        atr = self.tech_indicators.calculate_atr(dxy_high, dxy_low, dxy, period=14)
        features['dxy_atr_14'] = atr / (dxy.shift(1) + 1e-8)
        
        # EUR/USD vs DXY correlation (1 feature)
        eurusd_close = data_dict['EURUSD']['close']
        
        # Calculate rolling correlation with proper shifting to avoid look-ahead
        eurusd_shifted = eurusd_close.shift(1)
        dxy_shifted = dxy.shift(1)
        
        # Rolling correlation over 60 days (60 * 288 bars)
        correlation_window = 60 * 288
        features['corr_eurusd_dxy_60d'] = eurusd_shifted.rolling(
            window=correlation_window, min_periods=correlation_window//2
        ).corr(dxy_shifted)
        
        return features
    
    def calculate_cross_pair_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate cross-pair features (Group 3: 6 features).
        
        Args:
            data_dict: Dictionary of DataFrames for all pairs
            
        Returns:
            DataFrame with cross-pair features
        """
        logger.info("Calculating cross-pair features...")
        
        # Get the index from EURUSD
        index = data_dict['EURUSD'].index
        features = pd.DataFrame(index=index)
        
        # USD/JPY (2 features)
        features['usdjpy_return_1h'] = self.tech_indicators.calculate_returns(
            data_dict['USDJPY']['close'], periods=12
        )
        features['usdjpy_rsi_14'] = self.tech_indicators.calculate_rsi(
            data_dict['USDJPY']['close'], period=14
        )
        
        # EUR/GBP (2 features)
        features['eurgbp_return_1h'] = self.tech_indicators.calculate_returns(
            data_dict['EURGBP']['close'], periods=12
        )
        features['eurgbp_rsi_14'] = self.tech_indicators.calculate_rsi(
            data_dict['EURGBP']['close'], period=14
        )
        
        # EUR/JPY (2 features)
        features['eurjpy_return_1h'] = self.tech_indicators.calculate_returns(
            data_dict['EURJPY']['close'], periods=12
        )
        features['eurjpy_rsi_14'] = self.tech_indicators.calculate_rsi(
            data_dict['EURJPY']['close'], period=14
        )
        
        return features
    
    def calculate_risk_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate risk indicator features (Group 4: 2 features).
        
        Note: SPX not available in dataset, will use synthetic or skip if not available.
        
        Args:
            data_dict: Dictionary of DataFrames for all pairs
            
        Returns:
            DataFrame with risk features
        """
        logger.info("Calculating risk indicator features...")
        
        index = data_dict['EURUSD'].index
        features = pd.DataFrame(index=index)
        
        # SPX ATR (proxy for VIX) - if SPX available
        if 'SPXUSD' in data_dict:
            spx_df = data_dict['SPXUSD']
            atr = self.tech_indicators.calculate_atr(
                spx_df['high'], spx_df['low'], spx_df['close'], period=14
            )
            features['spx_atr_14'] = atr / (spx_df['close'].shift(1) + 1e-8)
        else:
            logger.warning("SPX data not available, using EUR/USD volatility as proxy")
            eurusd_df = data_dict['EURUSD']
            atr = self.tech_indicators.calculate_atr(
                eurusd_df['high'], eurusd_df['low'], eurusd_df['close'], period=14
            )
            features['spx_atr_14'] = atr / (eurusd_df['close'].shift(1) + 1e-8)
        
        # Gold (XAU/USD) return (1 feature)
        if 'XAUUSD' in data_dict:
            features['xauusd_return_1h'] = self.tech_indicators.calculate_returns(
                data_dict['XAUUSD']['close'], periods=12
            )
        else:
            logger.warning("Gold data not available, filling with zeros")
            features['xauusd_return_1h'] = 0.0
        
        return features
    
    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temporal features (Group 5: 7 features).
        
        Uses cyclical encoding for continuous time representation.
        
        Args:
            df: DataFrame with timestamp index
            
        Returns:
            DataFrame with temporal features
        """
        logger.info("Calculating temporal features...")
        
        features = pd.DataFrame(index=df.index)
        
        # Extract time components
        hours = df['timestamp'].dt.hour
        days = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        
        # Cyclical encoding for hours (24-hour cycle)
        features['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        
        # Cyclical encoding for days (7-day cycle)
        features['day_sin'] = np.sin(2 * np.pi * days / 7)
        features['day_cos'] = np.cos(2 * np.pi * days / 7)
        
        # Trading session indicators (binary)
        # European session: 07:00-16:00 UTC
        features['session_european'] = ((hours >= 7) & (hours < 16)).astype(float)
        
        # US session: 13:00-22:00 UTC
        features['session_us'] = ((hours >= 13) & (hours < 22)).astype(float)
        
        # Asian session: 23:00-09:00 UTC (wraps around midnight)
        features['session_asian'] = ((hours >= 23) | (hours < 9)).astype(float)
        
        return features
    
    def calculate_all_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate all 30 features.
        
        Args:
            data_dict: Dictionary of DataFrames for all pairs
            
        Returns:
            DataFrame with all 30 features
        """
        logger.info("="*80)
        logger.info("Starting Feature Engineering")
        logger.info("="*80)
        
        # Group 1: EUR/USD features (10)
        eurusd_features = self.calculate_eurusd_features(data_dict['EURUSD'])
        
        # Group 2: DXY features (5)
        dxy_features = self.calculate_dxy_features(data_dict)
        
        # Group 3: Cross-pair features (6)
        cross_features = self.calculate_cross_pair_features(data_dict)
        
        # Group 4: Risk features (2)
        risk_features = self.calculate_risk_features(data_dict)
        
        # Group 5: Temporal features (7)
        temporal_features = self.calculate_temporal_features(data_dict['EURUSD'])
        
        # Combine all features
        all_features = pd.concat([
            eurusd_features,
            dxy_features,
            cross_features,
            risk_features,
            temporal_features
        ], axis=1)
        
        logger.info(f"Total features calculated: {len(all_features.columns)}")
        logger.info(f"Feature names: {list(all_features.columns)}")
        
        # Verify we have exactly 30 features
        if len(all_features.columns) != self.config.total_features:
            logger.warning(f"Expected {self.config.total_features} features, got {len(all_features.columns)}")
        
        return all_features
    
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using expanding window Z-score.
        
        Formula: normalized_t = clip((value_t - mean_{0:t-1}) / (std_{0:t-1} + ε), -5, 5)
        
        CRITICAL: Uses expanding window to avoid look-ahead bias.
        
        Args:
            features: DataFrame with raw features
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features with expanding window...")
        
        normalized = pd.DataFrame(index=features.index)
        
        # Features that should NOT be normalized
        non_normalized_features = [
            'rsi_14', 'dxy_rsi_14', 'usdjpy_rsi_14', 'eurgbp_rsi_14', 'eurjpy_rsi_14',
            'session_european', 'session_us', 'session_asian'
        ]
        
        for col in features.columns:
            if col in non_normalized_features:
                # These are already in [0, 1] range
                normalized[col] = features[col]
            else:
                # Calculate expanding mean and std
                expanding_mean = features[col].expanding(min_periods=100).mean()
                expanding_std = features[col].expanding(min_periods=100).std()
                
                # Shift to avoid using current value in normalization
                expanding_mean = expanding_mean.shift(1)
                expanding_std = expanding_std.shift(1)
                
                # Normalize
                normalized_values = (features[col] - expanding_mean) / (expanding_std + self.config.epsilon)
                
                # Clip to [-5, 5]
                normalized_values = normalized_values.clip(
                    lower=self.config.clip_min,
                    upper=self.config.clip_max
                )
                
                # Fill NaN with 0 for warmup period
                normalized[col] = normalized_values.fillna(0.0)
        
        logger.info("Normalization complete")
        return normalized
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, any]:
        """
        Validate feature quality and check for issues.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating features...")
        
        validation_results = {
            'total_features': len(features.columns),
            'total_samples': len(features),
            'nan_counts': {},
            'inf_counts': {},
            'feature_ranges': {},
            'issues': []
        }
        
        for col in features.columns:
            # Check for NaN
            nan_count = features[col].isna().sum()
            validation_results['nan_counts'][col] = nan_count
            if nan_count > len(features) * 0.1:  # More than 10% NaN
                validation_results['issues'].append(f"{col}: {nan_count} NaN values ({nan_count/len(features)*100:.1f}%)")
            
            # Check for inf
            inf_count = np.isinf(features[col]).sum()
            validation_results['inf_counts'][col] = inf_count
            if inf_count > 0:
                validation_results['issues'].append(f"{col}: {inf_count} infinite values")
            
            # Check ranges
            valid_values = features[col].dropna()
            if len(valid_values) > 0:
                validation_results['feature_ranges'][col] = {
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std())
                }
        
        # Log validation results
        if validation_results['issues']:
            logger.warning("Validation issues found:")
            for issue in validation_results['issues']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("All features passed validation!")
        
        return validation_results


class FeaturePersistence:
    """Save and load features using HDF5."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def save_features_to_hdf5(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_timestamps: pd.Series,
        val_timestamps: pd.Series,
        test_timestamps: pd.Series,
        validation_results: Dict,
        filename: str = "features_normalized.h5"
    ) -> Path:
        """
        Save normalized features to HDF5.
        
        Structure:
        /features/
          train/
          val/
          test/
        /metadata/
          timestamps/
          validation/
        /normalization_params/
          (future: for storing means/stds if needed)
        
        Args:
            train_features: Training features
            val_features: Validation features
            test_features: Test features
            train_timestamps: Training timestamps
            val_timestamps: Validation timestamps
            test_timestamps: Test timestamps
            validation_results: Validation results dictionary
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.config.normalized_data_dir / filename
        
        logger.info(f"Saving normalized features to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Create feature groups
            features_grp = f.create_group('features')
            
            # Save training features
            train_grp = features_grp.create_group('train')
            for col in train_features.columns:
                train_grp.create_dataset(col, data=train_features[col].values)
            
            # Save validation features
            val_grp = features_grp.create_group('val')
            for col in val_features.columns:
                val_grp.create_dataset(col, data=val_features[col].values)
            
            # Save test features
            test_grp = features_grp.create_group('test')
            for col in test_features.columns:
                test_grp.create_dataset(col, data=test_features[col].values)
            
            # Save metadata
            meta_grp = f.create_group('metadata')
            
            # Save timestamps
            timestamps_grp = meta_grp.create_group('timestamps')
            timestamps_grp.create_dataset('train', data=train_timestamps.astype('int64') // 10**9)
            timestamps_grp.create_dataset('val', data=val_timestamps.astype('int64') // 10**9)
            timestamps_grp.create_dataset('test', data=test_timestamps.astype('int64') // 10**9)
            
            # Save feature names
            feature_names = [name.encode('utf-8') for name in train_features.columns]
            meta_grp.create_dataset('feature_names', data=feature_names)
            
            # Save validation results as attributes
            validation_grp = meta_grp.create_group('validation')
            validation_grp.attrs['total_features'] = validation_results['total_features']
            validation_grp.attrs['total_samples'] = validation_results['total_samples']
            
            # Save feature ranges
            for feature, ranges in validation_results['feature_ranges'].items():
                feat_grp = validation_grp.create_group(feature)
                for key, value in ranges.items():
                    feat_grp.attrs[key] = value
            
            # Save general metadata
            meta_grp.attrs['created_at'] = pd.Timestamp.now().isoformat()
            meta_grp.attrs['total_features'] = len(train_features.columns)
        
        logger.info(f"Successfully saved features to {output_path}")
        return output_path
    
    def load_features_from_hdf5(self, filename: str = "features_normalized.h5") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load normalized features from HDF5.
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (train_features, val_features, test_features)
        """
        input_path = self.config.normalized_data_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Features file not found: {input_path}")
        
        logger.info(f"Loading normalized features from {input_path}")
        
        with h5py.File(input_path, 'r') as f:
            # Load feature names
            feature_names = [name.decode('utf-8') for name in f['metadata/feature_names'][:]]
            
            # Load training features
            train_features = pd.DataFrame({
                col: f[f'features/train/{col}'][:]
                for col in feature_names
            })
            
            # Load validation features
            val_features = pd.DataFrame({
                col: f[f'features/val/{col}'][:]
                for col in feature_names
            })
            
            # Load test features
            test_features = pd.DataFrame({
                col: f[f'features/test/{col}'][:]
                for col in feature_names
            })
            
            # Load timestamps
            train_features['timestamp'] = pd.to_datetime(
                f['metadata/timestamps/train'][:], unit='s'
            )
            val_features['timestamp'] = pd.to_datetime(
                f['metadata/timestamps/val'][:], unit='s'
            )
            test_features['timestamp'] = pd.to_datetime(
                f['metadata/timestamps/test'][:], unit='s'
            )
            
            # Log metadata
            logger.info(f"Data created at: {f['metadata'].attrs['created_at']}")
            logger.info(f"Total features: {f['metadata'].attrs['total_features']}")
        
        logger.info("Successfully loaded features")
        return train_features, val_features, test_features


class FeaturePipeline:
    """Main feature engineering pipeline."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.engineer = FeatureEngineer(self.config)
        self.persistence = FeaturePersistence(self.config)
    
    def run_full_pipeline(
        self,
        train_data: Dict[str, pd.DataFrame],
        val_data: Dict[str, pd.DataFrame],
        test_data: Dict[str, pd.DataFrame],
        force_recalculate: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete feature engineering pipeline.
        
        Steps:
        1. Calculate raw features for each split
        2. Normalize features
        3. Validate features
        4. Save to HDF5
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            test_data: Test data dictionary
            force_recalculate: If True, recalculate even if cache exists
            
        Returns:
            Tuple of (train_features, val_features, test_features)
        """
        logger.info("="*80)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("="*80)
        
        # Check if features already exist
        features_file = self.config.normalized_data_dir / "features_normalized.h5"
        if features_file.exists() and not force_recalculate:
            logger.info("Features file found, loading from cache...")
            return self.persistence.load_features_from_hdf5()
        
        # Calculate features for each split
        logger.info("\n[1/4] Calculating raw features...")
        
        logger.info("Training set:")
        train_features = self.engineer.calculate_all_features(train_data)
        
        logger.info("Validation set:")
        val_features = self.engineer.calculate_all_features(val_data)
        
        logger.info("Test set:")
        test_features = self.engineer.calculate_all_features(test_data)
        
        # Normalize features
        logger.info("\n[2/4] Normalizing features...")
        train_features_norm = self.engineer.normalize_features(train_features)
        val_features_norm = self.engineer.normalize_features(val_features)
        test_features_norm = self.engineer.normalize_features(test_features)
        
        # Validate features
        logger.info("\n[3/4] Validating features...")
        validation_results = self.engineer.validate_features(train_features_norm)
        
        # Save features
        logger.info("\n[4/4] Saving features to HDF5...")
        self.persistence.save_features_to_hdf5(
            train_features_norm,
            val_features_norm,
            test_features_norm,
            train_data['EURUSD']['timestamp'],
            val_data['EURUSD']['timestamp'],
            test_data['EURUSD']['timestamp'],
            validation_results
        )
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Feature Engineering Complete!")
        logger.info("="*80)
        logger.info(f"Total features: {len(train_features_norm.columns)}")
        logger.info(f"Training samples: {len(train_features_norm)}")
        logger.info(f"Validation samples: {len(val_features_norm)}")
        logger.info(f"Test samples: {len(test_features_norm)}")
        logger.info("="*80)
        
        return train_features_norm, val_features_norm, test_features_norm


def main():
    """Example usage of the feature engineering pipeline."""
    from backend.data_pipeline import DataPipeline
    
    # Load processed data
    logger.info("Loading processed data...")
    data_pipeline = DataPipeline()
    train_data, val_data, test_data = data_pipeline.get_processed_data()
    
    # Create feature pipeline
    feature_pipeline = FeaturePipeline()
    
    # Run feature engineering
    train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
        train_data, val_data, test_data, force_recalculate=False
    )
    
    # Print feature statistics
    print("\n" + "="*80)
    print("Feature Statistics (Training Set)")
    print("="*80)
    print(train_features.describe())
    
    print("\n" + "="*80)
    print("Sample Features (first 5 rows)")
    print("="*80)
    print(train_features.head())


if __name__ == "__main__":
    main()
