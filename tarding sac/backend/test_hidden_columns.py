"""
Test script to validate hidden columns (raw_close, timestamp) implementation.

This script tests:
1. Data pipeline saves raw_close correctly
2. Feature engineering preserves hidden columns
3. Trading environment uses raw_close for PnL calculations
4. Agent does NOT see hidden columns in observations
"""

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_pipeline():
    """Test that data pipeline saves and loads raw_close."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Data Pipeline - raw_close preservation")
    logger.info("="*80)

    from data_pipeline import DataPipeline

    pipeline = DataPipeline()
    train_data, val_data, test_data = pipeline.get_processed_data()

    # Check EURUSD has raw_close
    assert 'raw_close' in train_data['EURUSD'].columns, "raw_close not found in train_data"
    assert 'raw_close' in val_data['EURUSD'].columns, "raw_close not found in val_data"
    assert 'raw_close' in test_data['EURUSD'].columns, "raw_close not found in test_data"

    # Check raw_close values are same as close (non-normalized)
    train_df = train_data['EURUSD']
    assert np.allclose(train_df['raw_close'].values, train_df['close'].values), \
        "raw_close should be identical to close"

    logger.info("✅ Data pipeline correctly saves raw_close")
    logger.info(f"   Sample raw_close values: {train_df['raw_close'].head(3).values}")
    logger.info(f"   Range: [{train_df['raw_close'].min():.5f}, {train_df['raw_close'].max():.5f}]")

    return train_data, val_data, test_data


def test_feature_engineering(train_data, val_data, test_data):
    """Test that feature engineering preserves hidden columns."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Feature Engineering - hidden columns preservation")
    logger.info("="*80)

    from feature_engineering import FeaturePipeline

    feature_pipeline = FeaturePipeline()
    train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
        train_data, val_data, test_data, force_recalculate=False
    )

    # Check hidden columns are loaded
    assert 'raw_close' in train_features.columns, "raw_close not found in train_features"
    assert 'timestamp' in train_features.columns, "timestamp not found in train_features"

    # Check raw_close values match original data
    assert np.allclose(
        train_features['raw_close'].values[:100],
        train_data['EURUSD']['raw_close'].values[:100]
    ), "raw_close values don't match between features and data"

    logger.info("✅ Feature engineering correctly preserves hidden columns")
    logger.info(f"   Features shape: {train_features.shape}")
    logger.info(f"   Has raw_close: {('raw_close' in train_features.columns)}")
    logger.info(f"   Has timestamp: {('timestamp' in train_features.columns)}")
    logger.info(f"   Sample raw_close: {train_features['raw_close'].head(3).values}")

    return train_features, val_features, test_features


def test_trading_environment(train_data, train_features):
    """Test that trading environment uses raw_close correctly."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Trading Environment - raw_close usage and PnL precision")
    logger.info("="*80)

    from trading_env import TradingEnvironment

    env = TradingEnvironment(
        data=train_data['EURUSD'],
        features=train_features,
        eval_mode=False
    )

    # Check that raw_close is extracted
    assert hasattr(env, 'raw_close'), "env.raw_close not found"
    assert len(env.raw_close) > 0, "env.raw_close is empty"

    # Check that hidden columns are removed from features
    assert 'raw_close' not in env.features.columns, \
        "raw_close should NOT be in env.features (visible to agent)"
    assert 'timestamp' not in env.features.columns, \
        "timestamp should NOT be in env.features (visible to agent)"

    # Check observation space excludes hidden columns
    obs = env.reset()
    assert obs.shape[0] == 30, f"Observation should have 30 features, got {obs.shape[0]}"

    logger.info("✅ Trading environment correctly uses raw_close")
    logger.info(f"   env.raw_close shape: {env.raw_close.shape}")
    logger.info(f"   env.features shape: {env.features.shape}")
    logger.info(f"   Observation shape: {obs.shape}")
    logger.info(f"   Hidden columns in features: {[col for col in ['raw_close', 'timestamp'] if col in env.features.columns]}")

    # Test PnL calculation with raw_close
    logger.info("\n   Testing PnL calculation precision...")

    # Reset and take a few steps
    obs = env.reset()
    episode_start_idx = env.episode_start

    # Get initial price (should use raw_close)
    initial_price = env.raw_close[episode_start_idx]
    logger.info(f"   Initial price (raw_close): {initial_price:.5f}")

    # Take a long position
    action = np.array([0.5])  # 50% position
    obs, reward, done, info = env.step(action)

    if env.position != 0:
        logger.info(f"   Position opened: {env.position:.4f} lots at {env.entry_price:.5f}")

        # Take a few more steps to accumulate PnL
        for _ in range(5):
            obs, reward, done, info = env.step(action)
            if done:
                break

        logger.info(f"   Current equity: ${info['equity']:.2f}")
        logger.info(f"   Total trades: {info['total_trades']}")

        # Verify entry price is from raw_close
        entry_idx = episode_start_idx + 1  # After first step
        expected_entry_price = env.raw_close[entry_idx]
        assert np.isclose(env.entry_price, expected_entry_price, rtol=1e-4), \
            f"Entry price {env.entry_price:.5f} should match raw_close {expected_entry_price:.5f}"

        logger.info(f"   ✅ Entry price matches raw_close: {env.entry_price:.5f}")
    else:
        logger.info("   ⚠️  No position opened (action too small or constraints)")

    return env


def test_observation_space(env):
    """Test that observations don't leak hidden columns."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Observation Space - no information leakage")
    logger.info("="*80)

    # Reset environment
    obs = env.reset()

    # Check observation shape
    assert obs.shape == (30,), f"Observation should be (30,), got {obs.shape}"

    # Check observation bounds
    assert np.all(obs >= env.config.obs_min), "Observation contains values below minimum"
    assert np.all(obs <= env.config.obs_max), "Observation contains values above maximum"

    # Take random steps and verify observations
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        assert obs.shape == (30,), f"Observation shape changed to {obs.shape}"
        assert np.all(np.isfinite(obs)), "Observation contains non-finite values"

        if done:
            break

    logger.info("✅ Observation space validated")
    logger.info(f"   Shape: {obs.shape}")
    logger.info(f"   Range: [{obs.min():.2f}, {obs.max():.2f}]")
    logger.info(f"   No hidden columns exposed to agent")


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("HIDDEN COLUMNS VALIDATION TEST SUITE")
    logger.info("="*80)
    logger.info("\nThis test validates that:")
    logger.info("1. raw_close is saved in data pipeline")
    logger.info("2. raw_close is preserved in feature engineering")
    logger.info("3. raw_close is used for PnL calculations")
    logger.info("4. raw_close is NOT visible to the agent")
    logger.info("\n")

    try:
        # Test 1: Data Pipeline
        train_data, val_data, test_data = test_data_pipeline()

        # Test 2: Feature Engineering
        train_features, val_features, test_features = test_feature_engineering(
            train_data, val_data, test_data
        )

        # Test 3: Trading Environment
        env = test_trading_environment(train_data, train_features)

        # Test 4: Observation Space
        test_observation_space(env)

        logger.info("\n" + "="*80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*80)
        logger.info("\nSummary:")
        logger.info("  ✓ Data pipeline saves raw_close")
        logger.info("  ✓ Feature engineering preserves hidden columns")
        logger.info("  ✓ Trading environment uses raw_close for PnL")
        logger.info("  ✓ Agent observations don't leak hidden columns")
        logger.info("\nHidden columns implementation is working correctly!")

    except AssertionError as e:
        logger.error("\n" + "="*80)
        logger.error(f"❌ TEST FAILED: {e}")
        logger.error("="*80)
        raise
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error(f"❌ UNEXPECTED ERROR: {e}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
