"""Agent adapters for manyAgents orchestration."""

from .base import AgentAdapter
from .manylatents_adapter import ManyLatentsAdapter
from .placeholder_adapter import PlaceholderAdapter

__all__ = ["AgentAdapter", "ManyLatentsAdapter", "PlaceholderAdapter"]