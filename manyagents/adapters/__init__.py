"""Agent adapters for manyAgents orchestration."""

from .base import AgentAdapter
from .manylatents_adapter import ManyLatentsAdapter
from .placeholder_adapter import PlaceholderAdapter
from .claude_adapter import ClaudeAdapter
from .openai_adapter import OpenAIAdapter
from .biomni_adapter import BiomniAdapter
from .kosmos_adapter import KosmosAdapter
from .local_llm_adapter import LocalLLMAdapter

# Alias for backwards compatibility
BioDiscoveryAgentAdapter = PlaceholderAdapter

__all__ = [
    "AgentAdapter",
    "ManyLatentsAdapter",
    "PlaceholderAdapter",
    "BioDiscoveryAgentAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "BiomniAdapter",
    "KosmosAdapter",
    "LocalLLMAdapter",
]