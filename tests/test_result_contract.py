"""Adapter validity and evaluation response contracts."""

from unittest.mock import AsyncMock

import numpy as np
import pytest
from omegaconf import OmegaConf

from manyagents.adapters import ADAPTER_REGISTRY
from manyagents.adapters.base import AdapterResult
from manyagents.experiment import run_experiment
from manyagents.types import validate_adapter_result


def test_compute_adapter_result_still_validates():
    result = {
        "success": True,
        "summary": "DR complete",
        "output_files": {"embeddings": np.zeros((3, 2)), "scores": {"trustworthiness": 0.9}},
    }
    assert validate_adapter_result(result, "manylatents") is result


def test_adapter_result_is_canonical():
    from manyagents.types import AdapterResult as PublicAdapterResult

    assert PublicAdapterResult is AdapterResult


@pytest.mark.parametrize("output_files", [None, [], "outputs"])
def test_adapter_result_requires_output_files_dict(output_files):
    with pytest.raises(ValueError, match="manylatents.*output_files.*dict"):
        validate_adapter_result(
            {"success": True, "summary": "DR complete", "output_files": output_files},
            "manylatents",
        )


def test_adapter_result_requires_output_files():
    with pytest.raises(ValueError, match="manylatents.*output_files"):
        validate_adapter_result({"success": True, "summary": "DR complete"}, "manylatents")


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_name", ["manylatents", "cellforge", "kosmos", "placeholder"])
async def test_compute_adapter_rejected_for_evaluation(adapter_name, monkeypatch, tmp_path, caplog):
    from manyagents.adapters import PlaceholderAdapter

    monkeypatch.setitem(ADAPTER_REGISTRY, "placeholder", PlaceholderAdapter)
    run = AsyncMock(return_value={"success": True, "output_files": {}})
    for name in ["mock", adapter_name]:
        monkeypatch.setattr(ADAPTER_REGISTRY[name], "run", run)
    cfg = OmegaConf.create({
        "name": "contract", "output_dir": str(tmp_path), "system_prompt": "",
        "active_agents": ["mock", adapter_name],
        "agents": {name: {"adapter": name, "config": {}} for name in ["mock", adapter_name]},
        "prompts": {"test": {"text": "Recommend methods."}},
    })
    with pytest.raises(SystemExit) as exc:
        await run_experiment(cfg)
    assert exc.value.code == 1
    assert f"Adapter '{adapter_name}' does not produce text responses" in caplog.text
    run.assert_not_awaited()


@pytest.fixture
def response_agent(monkeypatch):
    from manyagents.experiment import _run_agent

    async def evaluate(output_files):
        monkeypatch.setattr(ADAPTER_REGISTRY["mock"], "run", AsyncMock(return_value={
            "success": True, "summary": "Done", "output_files": output_files,
        }))
        return await _run_agent(OmegaConf.create({"adapter": "mock", "config": {}}), "prompt", "")

    return evaluate


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, {}, []], ids=["None", "dict", "list"])
async def test_extract_response_rejects_non_text(value, response_agent):
    result = await response_agent({"raw_response": value})
    assert result["success"] is False
    assert result["raw_response"] is None
    assert "mock" in result["error"]
    assert type(value).__name__ in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", " \n\t"], ids=["empty", "whitespace"])
@pytest.mark.parametrize("as_path", [False, True], ids=["str", "Path"])
async def test_extract_response_rejects_empty_and_whitespace(value, as_path, response_agent, tmp_path):
    if as_path:
        path = tmp_path / "response.txt"
        path.write_text(value)
        value = path
    result = await response_agent({"raw_response": value})
    assert result["success"] is False
    assert "mock" in result["error"]
    assert "empty" in result["error"]
    assert type(value).__name__ in result["error"]


@pytest.mark.asyncio
async def test_extract_response_accepts_path_and_str(response_agent, tmp_path):
    content = "Use Leiden and UMAP."
    path = tmp_path / "response.txt"
    path.write_text(content)
    for value in [path, content]:
        result = await response_agent({"raw_response": value})
        assert result["success"] is True
        assert result["raw_response"] == content
        assert set(result["extracted_methods"]) == {"leiden", "umap"}


@pytest.mark.asyncio
async def test_valid_response_with_no_methods_is_success(response_agent):
    result = await response_agent({"raw_response": "I cannot recommend a method."})
    assert result["success"] is True
    assert result["extracted_methods"] == []


@pytest.mark.asyncio
async def test_extract_response_rejects_missing_and_unreadable(response_agent, tmp_path):
    for output_files in [{}, {"raw_response": tmp_path / "missing.txt"}]:
        result = await response_agent(output_files)
        assert result["success"] is False
        assert "mock" in result["error"]


def test_save_response_returns_readable_path(tmp_path):
    from manyagents.adapters import MockAdapter

    adapter = MockAdapter()
    adapter.config.output_base_dir = tmp_path
    output = adapter.save_response("Use Leiden.")
    assert set(output) == {"raw_response"}
    assert output["raw_response"].read_text() == "Use Leiden."
    custom = adapter.save_response("Use PCA.", "custom.txt")
    assert custom["raw_response"].name == "custom.txt"
    assert custom["raw_response"].read_text() == "Use PCA."


@pytest.mark.asyncio
@pytest.mark.parametrize("as_dict", [False, True], ids=["objects", "dicts"])
async def test_claude_joins_all_text_blocks(as_dict, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from manyagents.adapters import ClaudeAdapter

    blocks = [
        {"type": "thinking", "thinking": "Private reasoning"},
        {"type": "text", "text": "First recommendation: Leiden."},
        {"type": "thinking", "thinking": "More reasoning"},
        {"type": "text", "text": "Second recommendation: SLINGSHOT."},
    ]
    response = SimpleNamespace(
        content=blocks if as_dict else [SimpleNamespace(**block) for block in blocks],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20), stop_reason="end_turn",
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))
    adapter = ClaudeAdapter()
    adapter.config.output_base_dir = tmp_path
    monkeypatch.setattr(adapter, "_get_client", lambda: client)
    result = await adapter.run({"prompt": "Recommend methods."}, {})
    assert result["success"] is True
    assert result["output_files"]["raw_response"].read_text() == (
        "First recommendation: Leiden.\nSecond recommendation: SLINGSHOT."
    )
