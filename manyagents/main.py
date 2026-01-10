"""
ManyAgents CLI - LLM agent testing framework.

Tests whether LLMs can reason about data geometry by dispatching
prompts to multiple agents and evaluating their responses.
"""

import asyncio

import hydra
from omegaconf import DictConfig

# Register shop's Hydra launchers
try:
    from shop.hydra.config_store import register_shop_launchers
    register_shop_launchers()
except ImportError:
    pass


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    """Run experiment with configured prompts and agents."""
    from manyagents.experiment import run_experiment
    return asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    main()
