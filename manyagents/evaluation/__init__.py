"""
Evaluation pipeline for Phase 3 RL training.

This module provides infrastructure for comparing embedding methods and computing
reward signals based on geometric metric differences.
"""

from .pipeline import EvaluationPipeline

__all__ = ["EvaluationPipeline"]
