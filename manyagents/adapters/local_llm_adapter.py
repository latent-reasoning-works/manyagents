"""Backward compatibility — LocalLLMAdapter is now HFAdapter."""
from .hf_adapter import HFAdapter as LocalLLMAdapter

__all__ = ["LocalLLMAdapter"]
