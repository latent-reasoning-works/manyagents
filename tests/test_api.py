from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir, version
from hydra.core.global_hydra import GlobalHydra
from hydra.core.utils import JobRuntime
from hydra.errors import MissingConfigException

import manyagents.api as api


@pytest.fixture(autouse=True)
def restore_test_hydra_globals(monkeypatch):
    # The caller initialized by these tests also changes Hydra's version/job
    # globals. Isolate that setup so it cannot change later CLI test behavior.
    base = version.VersionBase.instance()
    monkeypatch.setattr(base, "version_base", base.getbase())
    runtime = JobRuntime()
    monkeypatch.setattr(runtime, "conf", deepcopy(runtime.conf))


def test_run_calls_experiment(monkeypatch):
    seen = {}
    def fake_compose(config_name, overrides):
        seen["overrides"] = overrides
        return {"name": "t", "overrides": overrides}
    async def fake_run_experiment(cfg):
        seen["cfg"] = cfg
        return {"ok": True}
    monkeypatch.setattr(api, "_compose", fake_compose)
    monkeypatch.setattr(api, "run_experiment", fake_run_experiment)
    out = api.run(["experiment=demo", "seed=42"])
    assert out == {"ok": True}
    assert seen["overrides"] == ["experiment=demo", "seed=42"]


@pytest.mark.parametrize("failure", [None, "compose", "experiment", "exit"])
def test_run_preserves_callers_real_hydra_context(tmp_path, monkeypatch, failure):
    (tmp_path / "caller.yaml").write_text("caller_value: 42\n")

    async def execute(cfg):
        assert cfg.name == "test_wandb_integration"
        if failure == "experiment":
            raise RuntimeError("experiment failed")
        if failure == "exit":
            raise SystemExit(1)
        return {"ok": True}

    monkeypatch.setattr(api, "run_experiment", execute)
    with initialize_config_dir(config_dir=str(tmp_path), job_name="caller", version_base="1.1"):
        caller = GlobalHydra.instance()
        caller_hydra = caller.hydra
        caller_version = version.getbase()
        caller_runtime = JobRuntime().conf
        assert compose(config_name="caller").caller_value == 42
        config_name = "missing_config" if failure == "compose" else "main"
        try:
            if failure:
                exception = {"compose": MissingConfigException, "experiment": RuntimeError, "exit": SystemExit}[failure]
                with pytest.raises(exception):
                    api.run(["experiment=test_wandb"], config_name=config_name)
            else:
                result, cfg = api.run(["experiment=test_wandb"], return_cfg=True)
                assert result == {"ok": True}
                assert cfg.active_agents == ["mock"]
        finally:
            assert version.getbase() == caller_version
            assert JobRuntime().conf is caller_runtime
            assert JobRuntime().get("name") == "caller"
            assert GlobalHydra.instance() is caller
            assert GlobalHydra.instance().hydra is caller_hydra
            assert compose(config_name="caller").caller_value == 42
    assert not GlobalHydra.instance().is_initialized()


def test_run_without_caller_leaves_hydra_uninitialized(monkeypatch):
    async def execute(cfg):
        return cfg.name
    monkeypatch.setattr(api, "run_experiment", execute)
    assert api.run(["experiment=test_wandb"]) == "test_wandb_integration"
    assert not GlobalHydra.instance().is_initialized()
