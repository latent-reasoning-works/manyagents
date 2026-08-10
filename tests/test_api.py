import manyagents.api as api


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
