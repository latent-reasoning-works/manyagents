"""Programmatic entry point mirroring manylatents.api.run(). A thin per-beat
script composes overrides and calls run() — no orchestration logic in scripts."""
from __future__ import annotations

import asyncio
from typing import List, Optional

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

# Register shop's Hydra launchers (mirrors manyagents/main.py) so `cluster=`
# overrides naming a shop launcher resolve even when main.py is bypassed.
try:
    from shop.hydra.config_store import register_shop_launchers
    register_shop_launchers()
except ImportError:
    pass

from manyagents.experiment import run_experiment


def _compose(config_name: str, overrides: List[str]):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_module(config_module="manyagents.configs", version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def run(overrides: Optional[List[str]] = None, *, config_name: str = "main",
        return_cfg: bool = False):
    cfg = _compose(config_name, list(overrides or []))
    result = asyncio.run(run_experiment(cfg))
    return (result, cfg) if return_cfg else result
