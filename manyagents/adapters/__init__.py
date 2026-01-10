"""
ManyAgents adapter registry.

Provides a centralized registry of all available adapters.
"""

from .base import AgentAdapter
from .mock_adapter import MockAdapter
from .claude_adapter import ClaudeAdapter
from .openai_adapter import OpenAIAdapter
from .local_llm_adapter import LocalLLMAdapter
from .cellforge_adapter import CellForgeAdapter
from .kosmos_adapter import KosmosAdapter
from .placeholder_adapter import PlaceholderAdapter

# Core adapters (always available)
ADAPTER_REGISTRY = {
    "mock": MockAdapter,
    "claude": ClaudeAdapter,
    "openai": OpenAIAdapter,
    "local_llm": LocalLLMAdapter,
    "cellforge": CellForgeAdapter,
    "kosmos": KosmosAdapter,
}

# Optional adapters (require additional dependencies)
try:
    from .manylatents_adapter import ManyLatentsAdapter
    ADAPTER_REGISTRY["manylatents"] = ManyLatentsAdapter
except ImportError:
    ManyLatentsAdapter = None

try:
    from .biomni_adapter import BiomniAdapter
    ADAPTER_REGISTRY["biomni"] = BiomniAdapter
except ImportError:
    BiomniAdapter = None  # biomni package not installed

# Backwards compatibility alias
BioDiscoveryAgentAdapter = PlaceholderAdapter

__all__ = [
    "AgentAdapter",
    "ADAPTER_REGISTRY",
    "MockAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "LocalLLMAdapter",
    "CellForgeAdapter",
    "KosmosAdapter",
    "PlaceholderAdapter",
    "BioDiscoveryAgentAdapter",
]
