"""
Trainable models for Phase 3 RL optimization.

This module provides:
1. RLModule: Abstract base class for RL algorithms (library-agnostic)
2. SB3Module: Stable-Baselines3 implementation
3. Future: RLlibModule, custom GRPO implementations, etc.

The training loop:
  models/ → produces embeddings → MetricComputer audits → reward signal → trains models/ via RL

Architecture follows manylatents.LatentModule pattern:
- fit() for training
- predict() for inference
- Checkpointing support
- Swappable backends
"""

from .rl_module import RLModule
from .sb3_module import SB3Module

__all__ = ["RLModule", "SB3Module"]
