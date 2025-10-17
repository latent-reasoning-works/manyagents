"""
Stable-Baselines3 implementation of RLModule.

This wraps SB3 algorithms (PPO, SAC, TD3, etc.) with the RLModule interface.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .rl_module import RLModule


class SB3Module(RLModule):
    """
    RLModule implementation using Stable-Baselines3.

    Supported algorithms: PPO, SAC, TD3, A2C, DQN, etc.

    Example:
        >>> from gymnasium import make
        >>> env = make("CartPole-v1")
        >>> model = SB3Module(algorithm="ppo")
        >>> model.fit(env, n_timesteps=10000)
        >>> action, _ = model.predict(env.reset())
    """

    def __init__(
        self,
        algorithm: str = "ppo",
        policy_type: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """
        Initialize SB3-backed RL module.

        Args:
            algorithm: SB3 algorithm (ppo, sac, td3, a2c, dqn)
            policy_type: Policy network type
            learning_rate: Learning rate
            **kwargs: Algorithm-specific hyperparameters (passed to SB3)
        """
        super().__init__(algorithm, policy_type, learning_rate, **kwargs)

        # Lazy import to avoid dependency if not using SB3
        self._sb3_module = None
        self._algorithm_class = None

    def _get_algorithm_class(self):
        """Lazy load SB3 algorithm class."""
        if self._algorithm_class is not None:
            return self._algorithm_class

        # Import SB3
        import stable_baselines3 as sb3

        # Map algorithm name to class
        algorithm_map = {
            "ppo": sb3.PPO,
            "sac": sb3.SAC,
            "td3": sb3.TD3,
            "a2c": sb3.A2C,
            "dqn": sb3.DQN,
        }

        algo_lower = self.algorithm.lower()
        if algo_lower not in algorithm_map:
            raise ValueError(
                f"Unknown SB3 algorithm '{self.algorithm}'. "
                f"Supported: {list(algorithm_map.keys())}"
            )

        self._algorithm_class = algorithm_map[algo_lower]
        return self._algorithm_class

    def fit(self, env, n_timesteps: int, **kwargs) -> "SB3Module":
        """
        Train the RL policy using SB3.

        Args:
            env: Gymnasium environment
            n_timesteps: Total training timesteps
            **kwargs: SB3 training options (callback, eval_env, etc.)

        Returns:
            Self for method chaining
        """
        AlgorithmClass = self._get_algorithm_class()

        # Create SB3 model if not exists
        if self._model is None:
            self._model = AlgorithmClass(
                self.policy_type,
                env,
                learning_rate=self.learning_rate,
                **self.hyperparameters,
                verbose=kwargs.pop("verbose", 1)
            )

        # Train
        self._model.learn(total_timesteps=n_timesteps, **kwargs)
        self._is_fitted = True

        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get action from trained policy.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy

        Returns:
            (action, state) tuple
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        return self._model.predict(observation, deterministic=deterministic)

    def save(self, path: Path) -> None:
        """Save SB3 model checkpoint."""
        if self._model is None:
            raise RuntimeError("No model to save. Train first with fit().")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))

    def load(self, path: Path) -> "SB3Module":
        """Load SB3 model checkpoint."""
        AlgorithmClass = self._get_algorithm_class()
        self._model = AlgorithmClass.load(str(path))
        self._is_fitted = True
        return self
