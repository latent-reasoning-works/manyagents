"""
Base class for RL algorithms following manylatents LatentModule pattern.

This provides a unified interface for different RL libraries (SB3, RLlib, etc.)
to work seamlessly with manyAgents orchestration.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class RLModule(ABC):
    """
    Abstract base class for RL algorithms in manyAgents.

    Mirrors the LatentModule pattern from manylatents:
    - fit() for training
    - predict() for inference
    - State tracking with _is_fitted flag
    - Checkpointing support

    This enables swappable RL backends (Stable-Baselines3, RLlib, custom, etc.)
    while maintaining a consistent interface.
    """

    def __init__(
        self,
        algorithm: str,
        policy_type: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """
        Initialize RL module.

        Args:
            algorithm: RL algorithm name (e.g., "ppo", "sac", "dqn")
            policy_type: Policy network architecture (e.g., "MlpPolicy", "CnnPolicy")
            learning_rate: Learning rate for training
            **kwargs: Algorithm-specific hyperparameters
        """
        self.algorithm = algorithm
        self.policy_type = policy_type
        self.learning_rate = learning_rate
        self.hyperparameters = kwargs
        self._is_fitted = False
        self._model = None  # Backend-specific model object

    @abstractmethod
    def fit(self, env, n_timesteps: int, **kwargs) -> "RLModule":
        """
        Train the RL policy.

        Args:
            env: Environment to train on (gym.Env or custom)
            n_timesteps: Number of training timesteps
            **kwargs: Training-specific parameters (callbacks, eval_env, etc.)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get action from trained policy.

        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic or stochastic policy

        Returns:
            Tuple of (action, state) where state is for recurrent policies
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> "RLModule":
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Self for method chaining
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            "algorithm": self.algorithm,
            "policy_type": self.policy_type,
            "learning_rate": self.learning_rate,
            **self.hyperparameters
        }

    def set_params(self, **params) -> "RLModule":
        """Set model hyperparameters."""
        if "algorithm" in params:
            self.algorithm = params.pop("algorithm")
        if "policy_type" in params:
            self.policy_type = params.pop("policy_type")
        if "learning_rate" in params:
            self.learning_rate = params.pop("learning_rate")
        self.hyperparameters.update(params)
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted
