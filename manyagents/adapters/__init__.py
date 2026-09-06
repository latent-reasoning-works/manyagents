"""
ManyAgents adapter registry.

Provides a centralized registry of all available adapters.
"""

from .base import AgentAdapter
from .mock_adapter import MockAdapter
from .claude_adapter import ClaudeAdapter
from .openai_adapter import OpenAIAdapter, OllamaAdapter
from .hf_adapter import HFAdapter
from .vllm_adapter import VLLMAdapter
from .cellforge_adapter import CellForgeAdapter
from .kosmos_adapter import KosmosAdapter
from .placeholder_adapter import PlaceholderAdapter

# Core adapters (always available). The vLLM adapter imports `vllm` lazily at
# run() time, so registering it here is safe even without vllm installed — a
# missing dependency surfaces as a clean error_response, mirroring HFAdapter.
ADAPTER_REGISTRY = {
    "mock": MockAdapter,
    "claude": ClaudeAdapter,
    "openai": OpenAIAdapter,
    "ollama": OllamaAdapter,  # OpenAI-compatible local server (laptop dev, no GPU)
    "hf": HFAdapter,
    "local_llm": HFAdapter,  # backward compat alias
    "vllm": VLLMAdapter,
    "cellforge": CellForgeAdapter,
    "kosmos": KosmosAdapter,
    "placeholder": PlaceholderAdapter,
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
    "OllamaAdapter",
    "HFAdapter",
    "VLLMAdapter",
    "LocalLLMAdapter",
    "CellForgeAdapter",
    "KosmosAdapter",
    "PlaceholderAdapter",
    "BioDiscoveryAgentAdapter",
]
