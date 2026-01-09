"""
Smoke tests for CI - verify basic functionality works.

These tests are fast and don't require API keys or GPUs.
Run with: pytest tests/test_smoke.py -v
"""

import pytest


class TestAdapterImports:
    """Verify all adapters can be imported without errors."""

    def test_import_base(self):
        from manyagents.adapters.base import AgentAdapter
        assert AgentAdapter is not None

    def test_import_manylatents(self):
        from manyagents.adapters.manylatents_adapter import ManyLatentsAdapter
        assert ManyLatentsAdapter is not None

    def test_import_mock(self):
        from manyagents.adapters.mock_adapter import MockAdapter
        assert MockAdapter is not None

    def test_import_claude(self):
        from manyagents.adapters.claude_adapter import ClaudeAdapter
        assert ClaudeAdapter is not None

    def test_import_openai(self):
        from manyagents.adapters.openai_adapter import OpenAIAdapter
        assert OpenAIAdapter is not None

    def test_import_local_llm(self):
        from manyagents.adapters.local_llm_adapter import LocalLLMAdapter
        assert LocalLLMAdapter is not None


class TestAdapterInstantiation:
    """Verify adapters can be instantiated."""

    def test_mock_adapter_init(self):
        from manyagents.adapters.mock_adapter import MockAdapter
        adapter = MockAdapter()
        assert adapter.name == "mock"

    def test_claude_adapter_init(self):
        from manyagents.adapters.claude_adapter import ClaudeAdapter
        adapter = ClaudeAdapter()
        assert adapter.name == "claude"

    def test_openai_adapter_init(self):
        from manyagents.adapters.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter()
        assert adapter.name == "openai"


class TestMockAdapterExecution:
    """Verify mock adapter can execute without errors."""

    def test_mock_run(self):
        import asyncio
        from manyagents.adapters.mock_adapter import MockAdapter

        adapter = MockAdapter()
        result = asyncio.run(adapter.run(
            task_config={"scenario_name": "test"},
            input_files={}
        ))

        assert result["success"] is True
        assert "response" in result["output_files"]
        assert "methods_mentioned" in result["metadata"]


class TestRegistryLoads:
    """Verify adapter registry loads correctly."""

    def test_registry_has_adapters(self):
        from manyagents.main import ADAPTER_REGISTRY

        expected = ["mock", "claude", "openai", "local_llm", "manylatents"]
        for name in expected:
            assert name in ADAPTER_REGISTRY, f"Missing adapter: {name}"
