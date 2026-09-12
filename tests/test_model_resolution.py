"""Cluster aliases reach both inference backends without global resolver state."""

from pathlib import Path
from unittest.mock import MagicMock

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pytest

from manyagents import inference
from manyagents.adapters.hf_adapter import HFAdapter
from manyagents.adapters.vllm_adapter import VLLMAdapter


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "manyagents/configs")


@pytest.fixture(autouse=True)
def remote_launcher_config():
    # Compose the optional Shop config without installing or launching Shop.
    ConfigStore.instance().store(group="hydra/launcher", name="shop_remote_slurm", node={})


def config(cluster, agent):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="main", overrides=[
            f"cluster={cluster}", "experiment=trace_extraction", f"agent={agent}",
        ])


@pytest.mark.parametrize("cluster", ["mila_slurm", "mila_remote", "mila_sweep"])
@pytest.mark.parametrize("agent", ["hf", "vllm"])
def test_cluster_aliases_reach_task_config(cluster, agent):
    cfg = config(cluster, agent)
    task = dict(cfg.agent.config)
    aliases = task["available_models"]
    assert set(aliases) >= {"olmo-1b", "olmo-7b", "llama-3.1-8b", "qwen3-0.6b"}
    assert inference.resolve_model_path("olmo-7b", aliases) == cfg.available_models["olmo-7b"]
    assert inference.resolve_model_path("qwen3-0.6b", aliases) == "Qwen/Qwen3-0.6B"


@pytest.mark.parametrize("agent", ["hf", "vllm"])
def test_local_config_has_portable_default_and_no_site_aliases(agent):
    cfg = config("local", agent)
    assert cfg.agent.config.model == "Qwen/Qwen3-0.6B"
    with pytest.raises(ValueError, match="cluster.*available_models"):
        inference.resolve_model_path("llama-3.1-8b", cfg.agent.config.available_models)


def test_alias_maps_are_explicit_and_do_not_leak_between_calls():
    assert inference.resolve_model_path("olmo-7b", {"olmo-7b": "/models/olmo-7b"}) == "/models/olmo-7b"
    assert inference.resolve_model_path("olmo-7b", {"olmo-7b": "org/other"}) == "org/other"
    with pytest.raises(ValueError, match="cluster.*available_models"):
        inference.resolve_model_path("olmo-7b")
    for model in ("Qwen/Qwen3-0.6B", "/models/olmo-7b", "./weights"):
        assert inference.resolve_model_path(model) == model


def test_hf_default_is_portable():
    assert HFAdapter.DEFAULT_MODEL == VLLMAdapter.DEFAULT_MODEL == "Qwen/Qwen3-0.6B"


@pytest.mark.parametrize("adapter_class", [HFAdapter, VLLMAdapter])
@pytest.mark.parametrize("build_trace", [False, True])
async def test_adapters_load_resolved_path_for_generation_and_traces(
    adapter_class, build_trace, monkeypatch, tmp_path,
):
    monkeypatch.chdir(tmp_path)
    # Use composed config, as the runner does, with neutral site paths for CPU tests.
    cfg = config("mila_slurm", "hf" if adapter_class is HFAdapter else "vllm")
    cfg.available_models = {"olmo-7b": "/models/olmo-7b"}
    task = dict(cfg.agent.config)
    task.update(model="olmo-7b", prompt="2+2?", build_trace=build_trace,
                capture_hidden_states=False)
    load = MagicMock(return_value=(MagicMock(), MagicMock(), None))
    engine = MagicMock(return_value=MagicMock())
    trace = MagicMock(response_text="4")
    trace.to_json.return_value = "{}"
    extract = MagicMock(return_value=(trace, None))
    monkeypatch.setattr(inference, "get_model", load)
    monkeypatch.setattr(inference, "get_vllm_engine", engine)
    monkeypatch.setattr(inference, "extract_trace", extract)
    monkeypatch.setattr(inference, "build_prompt", lambda *a: "2+2?")
    monkeypatch.setattr(inference, "generate", lambda *a, **kw: "4")
    monkeypatch.setattr(inference, "vllm_generate", lambda *a, **kw: [{"text": "4"}])
    result = await adapter_class().run(task, {})
    assert result["success"], result
    assert result["metadata"]["model_path"] == "/models/olmo-7b"
    if adapter_class is HFAdapter or build_trace:
        assert load.call_args.args == ("/models/olmo-7b",)
    else:
        load.assert_not_called()
    if adapter_class is VLLMAdapter:
        assert engine.call_args.args == ("/models/olmo-7b",)
    if build_trace:
        assert extract.call_args.kwargs["model_path"] == "/models/olmo-7b"


def test_alias_interpolation_survives_agent_repackaging():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="main", overrides=[
            "cluster=mila_slurm", "experiment=invariance_full",
        ])
    task = OmegaConf.to_container(cfg.agents.local_llm.agent.config, resolve=True)
    assert task["available_models"] == dict(cfg.available_models)


@pytest.mark.parametrize("overrides,path", [
    (["+agent=local_llm"], "agent.config"),
    (["experiment=invariance_full", "agent@agents.local_llm=local_llm"],
     "agents.local_llm.agent.config"),
])
def test_legacy_local_llm_alias_preserves_hf_config(overrides, path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="main", overrides=overrides)
    task = OmegaConf.select(cfg, path)
    assert task is not None
    assert task.model == "Qwen/Qwen3-0.6B"
    assert task.max_new_tokens == 2000


@pytest.mark.parametrize("name", [
    "reasoning_baseline", "llm_reasoning_sweep", "baseline_sweep",
])
def test_shipped_wandb_is_opt_in(name):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="main", overrides=[f"experiment={name}"])
        opted_in = compose(config_name="main", overrides=[
            f"experiment={name}", "wandb.enabled=true",
        ])
    assert cfg.wandb.enabled is False
    assert opted_in.wandb.enabled is True
