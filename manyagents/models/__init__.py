"""
Trainable models for RL optimization.

NOTE: This module is being migrated to Geomancy. The RL training loop,
reward computation, and G-vector extraction are Geomancy's domain.

This module provides base classes that may be useful for manyAgents
internal training, but full RL infrastructure lives in Geomancy.
"""

from .rl_module import RLModule
from .sb3_module import SB3Module

__all__ = ["RLModule", "SB3Module"]
