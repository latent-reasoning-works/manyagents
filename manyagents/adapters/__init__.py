"""
ManyAgents adapter registry.

Provides a centralized registry of all available adapters.
"""

from .base import AgentAdapter
from .mock_adapter import MockAdapter
from .claude_adapter import ClaudeAdapter
from .openai_adapter import OpenAIAdapter
from .hf_adapter import HFAdapter
from .cellforge_adapter import CellForgeAdapter
from .kosmos_adapter import KosmosAdapter
from .placeholder_adapter import PlaceholderAdapter

# Core adapters (always available)
ADAPTER_REGISTRY = {
    "mock": MockAdapter,
    "claude": ClaudeAdapter,
    "openai": OpenAIAdapter,
    "hf": HFAdapter,
    "local_llm": HFAdapter,  # backward compat alias
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

# Backwards compatibility aliases
BioDiscoveryAgentAdapter = PlaceholderAdapter
LocalLLMAdapter = HFAdapter

__all__ = [
    "AgentAdapter",
    "ADAPTER_REGISTRY",
    "MockAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "HFAdapter",
    "LocalLLMAdapter",
    "CellForgeAdapter",
    "KosmosAdapter",
    "PlaceholderAdapter",
    "BioDiscoveryAgentAdapter",
]
