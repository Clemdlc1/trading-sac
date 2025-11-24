"""
Action Discretizer Wrapper for Trading Environment
=================================================

This wrapper allows continuous action agents (like SAC) to work with
discrete action environments by discretizing continuous actions.

Author: SAC EUR/USD Project
Version: 1.0
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict


class ActionDiscretizerWrapper(gym.Wrapper):
    """
    Wrapper that converts continuous actions to discrete actions.

    This allows continuous action agents (like SAC) to work with
    discrete action environments. The wrapper:
    1. Presents a continuous Box action space [-1, 1] to the agent
    2. Discretizes the continuous action into {0, 1, 2}
    3. Passes the discrete action to the wrapped environment

    Discretization mapping:
    - [-1.0, -0.33) -> 2 (short position)
    - [-0.33, 0.33] -> 0 (flat, no position)
    - (0.33, 1.0]   -> 1 (long position)
    """

    def __init__(self, env):
        """
        Initialize wrapper.

        Args:
            env: Environment with discrete action space
        """
        super().__init__(env)

        # Replace discrete action space with continuous one for the agent
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

    def _discretize_action(self, continuous_action: np.ndarray) -> int:
        """
        Convert continuous action to discrete action.

        Args:
            continuous_action: Continuous action in [-1, 1]

        Returns:
            Discrete action:
            - 0: flat (no position)
            - 1: long position
            - 2: short position
        """
        # Extract scalar value
        if isinstance(continuous_action, np.ndarray):
            action_value = float(continuous_action[0])
        else:
            action_value = float(continuous_action)

        # Clip to valid range
        action_value = np.clip(action_value, -1.0, 1.0)

        # Discretize with thresholds
        if action_value < -0.33:
            return 2  # Short
        elif action_value > 0.33:
            return 1  # Long
        else:
            return 0  # Flat

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute step with discretized action.

        Args:
            action: Continuous action from agent

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Discretize the continuous action
        discrete_action = self._discretize_action(action)

        # Pass to underlying environment
        return self.env.step(discrete_action)

    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)


def test_wrapper():
    """Test the action discretizer wrapper."""
    from trading_env import TradingEnvironment, TradingEnvConfig
    from data_pipeline import DataPipeline
    from feature_engineering import FeaturePipeline

    print("Loading data...")
    data_pipeline = DataPipeline()
    train_data, _, _ = data_pipeline.get_processed_data()

    feature_pipeline = FeaturePipeline()
    train_features, _, _ = feature_pipeline.run_full_pipeline(
        train_data, train_data, train_data
    )

    print("\nCreating discrete environment...")
    base_env = TradingEnvironment(
        data=train_data['EURUSD'],
        features=train_features,
        eval_mode=False
    )
    print(f"Base env action space: {base_env.action_space}")

    print("\nWrapping with ActionDiscretizerWrapper...")
    wrapped_env = ActionDiscretizerWrapper(base_env)
    print(f"Wrapped env action space: {wrapped_env.action_space}")

    print("\nTesting continuous actions:")
    obs = wrapped_env.reset()

    test_actions = [
        np.array([-0.8]),  # Should -> 2 (short)
        np.array([0.0]),   # Should -> 0 (flat)
        np.array([0.7]),   # Should -> 1 (long)
    ]

    for action in test_actions:
        discrete = wrapped_env._discretize_action(action)
        obs, reward, done, info = wrapped_env.step(action)
        print(f"Continuous action {action[0]:.1f} -> Discrete {discrete} -> "
              f"Position: {info['position']:.4f}")

    print("\n✓ Wrapper test complete!")


if __name__ == "__main__":
    test_wrapper()
