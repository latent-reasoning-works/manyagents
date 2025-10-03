"""Agent adapters for manyAgents orchestration."""

from .base import AgentAdapter
from .manylatents_adapter import ManyLatentsAdapter
from .placeholder_adapter import PlaceholderAdapter

# Alias for backwards compatibility
BioDiscoveryAgentAdapter = PlaceholderAdapter

__all__ = [
    "AgentAdapter",
    "ManyLatentsAdapter",
    "PlaceholderAdapter",
    "BioDiscoveryAgentAdapter"
]