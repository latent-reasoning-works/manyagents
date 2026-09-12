"""Programmatic entry point mirroring manylatents.api.run(). A thin per-beat
script composes overrides and calls run() — no orchestration logic in scripts."""
from __future__ import annotations

import asyncio
from copy import copy, deepcopy
from typing import List, Optional

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.utils import JobRuntime
from hydra.version import VersionBase

# Register shop's Hydra launchers (mirrors manyagents/main.py) so `cluster=`
# overrides naming a shop launcher resolve even when main.py is bypassed.
try:
    from shop.hydra.config_store import register_shop_launchers
    register_shop_launchers()
except ImportError:
    pass

from manyagents.experiment import run_experiment


def _compose(config_name: str, overrides: List[str]):
    # Hydra's initialize context restores only GlobalHydra, and uses a deep copy.
    # Preserve caller identity as well as the version/job globals it also changes.
    caller = GlobalHydra.instance()
    caller_version = VersionBase.instance()
    runtime = JobRuntime()
    caller_runtime = runtime.conf
    isolated = copy(caller)
    isolated.clear()
    try:
        GlobalHydra.set_instance(isolated)
        VersionBase.set_instance(copy(caller_version))
        runtime.conf = deepcopy(caller_runtime)
        with initialize_config_module(config_module="manyagents.configs", version_base=None):
            return compose(config_name=config_name, overrides=overrides)
    finally:
        GlobalHydra.set_instance(caller)
        VersionBase.set_instance(caller_version)
        runtime.conf = caller_runtime


def run(overrides: Optional[List[str]] = None, *, config_name: str = "main",
        return_cfg: bool = False):
    cfg = _compose(config_name, list(overrides or []))
    result = asyncio.run(run_experiment(cfg))
    return (result, cfg) if return_cfg else result
