# tests/test_hf_adapter.py
"""Tests for HFAdapter (renamed from LocalLLMAdapter)."""

from manyagents.adapters import ADAPTER_REGISTRY
from manyagents.adapters.hf_adapter import HFAdapter


def test_hf_adapter_in_registry():
    """HFAdapter is registered under both 'hf' and 'local_llm' keys."""
    assert "hf" in ADAPTER_REGISTRY
    assert "local_llm" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["hf"] is HFAdapter
    assert ADAPTER_REGISTRY["local_llm"] is HFAdapter


def test_hf_adapter_importable_from_old_path():
    """Backward compat: LocalLLMAdapter still importable from old module."""
    from manyagents.adapters.local_llm_adapter import LocalLLMAdapter
    assert LocalLLMAdapter is HFAdapter
