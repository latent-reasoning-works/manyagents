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
