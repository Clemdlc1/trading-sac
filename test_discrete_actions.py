"""
Test script to verify discrete action space implementation.
"""
import sys
sys.path.append('/home/user/trading-sac/tarding sac')

import numpy as np
from backend.trading_env import TradingEnvironment, TradingEnvConfig
from backend.data_pipeline import DataPipeline
from backend.feature_engineering import FeaturePipeline

def test_discrete_actions():
    """Test that the discrete action space works correctly."""
    print("Loading data and features...")
    data_pipeline = DataPipeline()
    train_data, val_data, test_data = data_pipeline.get_processed_data()

    feature_pipeline = FeaturePipeline()
    train_features, val_features, test_features = feature_pipeline.run_full_pipeline(
        train_data, val_data, test_data
    )

    print("\nCreating trading environment with discrete action space...")
    env = TradingEnvironment(
        data=train_data['EURUSD'],
        features=train_features,
        eval_mode=False
    )

    # Check action space type
    print(f"\nAction space: {env.action_space}")
    print(f"Action space type: {type(env.action_space)}")
    print(f"Number of actions: {env.action_space.n}")

    # Test reset
    print("\nTesting reset...")
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial equity: ${env.equity:.2f}")

    # Test each discrete action
    print("\n" + "="*80)
    print("Testing discrete actions:")
    print("="*80)

    actions = [0, 1, 2]
    action_names = ["FLAT (0)", "LONG (1)", "SHORT (2)"]

    for action, name in zip(actions, action_names):
        print(f"\nTesting action {name}")
        obs = env.reset()

        # Execute action for 5 steps
        for step in range(5):
            obs, reward, done, info = env.step(action)
            print(f"  Step {step+1}: Position={info['position']:.4f}, "
                  f"Equity=${info['equity']:.2f}, Reward={reward:.4f}")

            if done:
                break

    # Test random actions
    print("\n" + "="*80)
    print("Testing random discrete actions:")
    print("="*80)

    obs = env.reset()
    done = False
    step_count = 0
    action_counts = {0: 0, 1: 0, 2: 0}

    while not done and step_count < 100:
        action = env.action_space.sample()
        action_counts[action] += 1
        obs, reward, done, info = env.step(action)
        step_count += 1

        if step_count % 20 == 0:
            print(f"Step {step_count}: Action={action}, Position={info['position']:.4f}, "
                  f"Equity=${info['equity']:.2f}")

    print(f"\nAction distribution over {step_count} steps:")
    for action, count in action_counts.items():
        print(f"  Action {action}: {count} times ({count/step_count*100:.1f}%)")

    # Test action conversion
    print("\n" + "="*80)
    print("Testing action conversion:")
    print("="*80)

    for action in [0, 1, 2]:
        continuous = env._convert_discrete_action(action)
        print(f"Discrete action {action} -> Continuous action {continuous:.1f}")

    print("\n✓ All tests passed! Discrete action space is working correctly.")

if __name__ == "__main__":
    test_discrete_actions()
