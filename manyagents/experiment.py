"""
Experiment runner for LLM evaluation.

Called by manyagents.main when prompts config is detected.

Usage (via main CLI):
    manyagents experiment=invariance_golden
    manyagents experiment=invariance_full active_agents=[claude,openai]
    manyagents experiment=invariance_golden wandb.enabled=true
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .metrics.extractor import extract_methods, check_ground_truth_match
from .metrics.llm import compute_system_metrics, format_metric, generate_summary_table
from .types import validate_adapter_result
from .utils.logger import ExperimentLogger, NullLogger

log = logging.getLogger(__name__)


# ============================================================================
# RESULT BUILDERS
# ============================================================================

def _extract_raw_response(output_files: Dict[str, Any]) -> str:
    """Read a response Path or inline string, rejecting missing or empty text."""
    response = output_files.get('raw_response')
    response_type = type(response).__name__
    if isinstance(response, Path):
        try:
            content = response.read_text()
        except (OSError, UnicodeError) as e:
            raise ValueError(f"Unreadable raw_response ({response_type}): {e}") from e
    elif isinstance(response, str):
        content = response
    else:
        raise ValueError(f"raw_response must be Path or str, got {response_type}")
    if not content.strip():
        raise ValueError(f"raw_response is empty or whitespace-only ({response_type})")
    return content


def _build_success_result(raw_response: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build a success result dict from raw response."""
    extraction = extract_methods(raw_response)
    return {
        'success': True,
        'raw_response': raw_response,
        'extracted_methods': extraction.get('extracted_methods', []),
        'mentions_clustering': extraction.get('mentions_clustering', False),
        'mentions_trajectory': extraction.get('mentions_trajectory', False),
        'mentions_data_inspection': extraction.get('mentions_data_inspection', False),
        'metadata': metadata
    }


def _build_error_result(error: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build an error result dict."""
    result = {'success': False, 'error': error, 'raw_response': None}
    if metadata:
        result['metadata'] = metadata
    return result


def _add_ground_truth_matching(result: Dict[str, Any], prompt_config: Dict[str, Any]) -> None:
    """Add ground truth matching info to a result dict (mutates in place)."""
    if not result.get('success'):
        return
    is_match, match_details = check_ground_truth_match(
        result.get('extracted_methods', []),
        prompt_config.get('ground_truth_methods', []),
        prompt_config.get('failure_indicators', [])
    )
    result['matches_ground_truth'] = is_match
    result['ground_truth_details'] = match_details


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def _save_results(experiment_results: Dict[str, Any], output_dir: Path, experiment_id: str) -> None:
    """Save experiment results and summary to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # Generate and save markdown summary
    summary_md = generate_summary_table(experiment_results, format='markdown')
    summary_path = output_dir / "summary.md"

    name = experiment_results.get('config', {}).get('name', 'LLM Evaluation')
    with open(summary_path, 'w') as f:
        f.write(f"# {name}\n\n")
        f.write(f"**Experiment ID:** {experiment_id}\n")
        f.write(f"**Timestamp:** {experiment_results['timestamp']}\n\n")
        f.write("## Results\n\n")
        f.write(summary_md)
        f.write("\n\n## Interpretation\n\n")
        f.write("- **Jaccard (Invariance):** Higher = same recommendations across prompts (BAD)\n")
        f.write("- **Ground Truth Match:** Higher = geometry-aware recommendations (GOOD)\n")
        f.write("- **Clustering-for-All:** Higher = always recommends clustering (BAD)\n")

    log.info(f"Summary saved to {summary_path}")


def _print_summary(metrics: Dict[str, Dict[str, float | None]]) -> None:
    """Print metrics summary to console."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    for agent_name, m in metrics.items():
        print(f"\n{agent_name}:")
        print(f"  Jaccard Similarity: {format_metric(m.get('jaccard_similarity_across_prompts'), '.2f')}")
        print(f"  Ground Truth Match: {format_metric(m.get('ground_truth_match_rate'), '.1%')}")
        print(f"  Clustering-for-All: {format_metric(m.get('clustering_for_all_rate'), '.1%')}")


# ============================================================================
# TRACE EXTRACTION
# ============================================================================

def _load_dataset_tasks(dataset: str, n_samples: int) -> list:
    """Load task samples from a dataset."""
    if dataset == "gsm8k":
        from datasets import load_dataset as hf_load
        ds = hf_load("openai/gsm8k", "main", split="train")
        tasks = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            answer = row["answer"].split("####")[-1].strip()
            tasks.append({
                "task_id": f"gsm8k_train_{i}",
                "prompt": row["question"],
                "expected_answer": answer,
                "domain": "math",
                "logic_type": "arithmetic",
            })
        return tasks
    raise ValueError(f"Unknown dataset: {dataset}")


async def _run_trace_extraction(cfg: DictConfig) -> Dict[str, Any]:
    """Run trace extraction experiment — generates ReasoningTraces with hidden states.

    This is the Hydra-driven equivalent of the old scripts/extract_traces.py.
    """
    from manyagents.adapters import ADAPTER_REGISTRY
    from manyagents.schemas.reasoning import TraceStore, ReasoningTrace

    trace_cfg = cfg.trace_extraction
    agent_config = cfg.agent if hasattr(cfg, "agent") else _get_agent_config(cfg, cfg.active_agents[0])

    dataset_name = trace_cfg.get("dataset", "gsm8k")
    n_samples = trace_cfg.get("n_samples", 10)
    system_prompt = trace_cfg.get("system_prompt", "Solve the problem step by step.")

    tasks = _load_dataset_tasks(dataset_name, n_samples)
    log.info(f"Loaded {len(tasks)} tasks from {dataset_name}")

    adapter_name = agent_config.adapter if hasattr(agent_config, "adapter") else "hf"
    adapter = ADAPTER_REGISTRY[adapter_name]()

    output_dir = Path(cfg.output_dir)
    store_dir = output_dir / "traces"

    summary = {
        "total_traces": 0, "with_tensors": 0, "traces_failed": 0,
        "backends": {}, "datasets": {}, "success": 0, "failure": 0, "unjudged": 0,
    }
    with TraceStore(store_dir) as store:
        for i, task in enumerate(tasks):
            log.info(f"[{i + 1}/{len(tasks)}] {task['task_id']}")
            task_config = dict(agent_config.config) if hasattr(agent_config, "config") else {}
            task_config["prompt"] = task["prompt"]
            task_config["system_prompt"] = system_prompt
            task_config["dataset"] = dataset_name
            task_config["task_id"] = task["task_id"]
            task_config["expected_answer"] = task.get("expected_answer")
            task_config["domain"] = task.get("domain")
            task_config["logic_type"] = task.get("logic_type")

            try:
                result = validate_adapter_result(await adapter.run(task_config, {}), adapter_name)
                if not result["success"]:
                    raise ValueError(result.get("summary", "Adapter failed"))
                output_files = result.get("output_files", {})
                if "trace" not in output_files:
                    raise ValueError("Successful adapter result has no trace")
                trace = ReasoningTrace.from_json(Path(output_files["trace"]).read_text())

                hs = None
                if "hidden_states" in output_files:
                    with np.load(output_files["hidden_states"], allow_pickle=False) as tensors:
                        hs = dict(tensors)
                    if hs and not all(
                        array.ndim >= 2 and array.size > 0
                        and np.issubdtype(array.dtype, np.floating)
                        and np.isfinite(array).all()
                        for array in hs.values()
                    ):
                        raise ValueError("Hidden states must be nonempty, finite float state arrays (at least 2D)")
                    if not hs:
                        hs = None
                if task_config.get("capture_hidden_states", False) and hs is None:
                    raise ValueError("capture_hidden_states requested but trace has no tensors")

                if hs is not None and len(trace.steps) < 2:
                    raise ValueError(
                        "Hidden-state geometry requires at least two steps; "
                        "try segmentation=delimiter or increase max_new_tokens"
                    )

                store.append(trace, hidden_states=hs)
                summary["total_traces"] += 1
                summary["with_tensors"] += int(hs is not None)
                backend = trace.model.backend.value
                summary["backends"][backend] = summary["backends"].get(backend, 0) + 1
                dataset = trace.task.dataset
                summary["datasets"][dataset] = summary["datasets"].get(dataset, 0) + 1
                outcome = "unjudged" if trace.success is None else "success" if trace.success else "failure"
                summary[outcome] += 1
                log.info(f"  -> {len(trace.steps)} steps, {trace.output_tokens} tokens")
            except Exception as e:
                summary["traces_failed"] += 1
                log.error(f"  {adapter_name}/{task['task_id']} FAILED: {e}", exc_info=True)

    log.info(f"Trace extraction complete: {json.dumps(summary, indent=2)}")
    if summary["total_traces"] == 0:
        log.error(f"Trace extraction failed: 0 persisted, {summary['traces_failed']} failed")
        raise SystemExit(1)

    return {"experiment_id": f"trace_{dataset_name}", "summary": summary, "output_dir": str(store_dir)}


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def _create_logger(cfg: DictConfig, experiment_id: str) -> ExperimentLogger:
    """Create an experiment logger based on config."""
    wandb_cfg = getattr(cfg, 'wandb', None)

    if wandb_cfg is None or not getattr(wandb_cfg, 'enabled', False):
        return NullLogger()

    return ExperimentLogger(
        project=getattr(wandb_cfg, 'project', 'manyagents'),
        experiment_name=experiment_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        entity=getattr(wandb_cfg, 'entity', None),
        tags=list(getattr(wandb_cfg, 'tags', [])) or None,
        enabled=True,
    )


async def _run_agent(agent_config: DictConfig, prompt: str, system_prompt: str) -> Dict[str, Any]:
    """Run a single agent with the given prompt."""
    from manyagents.adapters import ADAPTER_REGISTRY

    adapter_name = agent_config.adapter
    if adapter_name not in ADAPTER_REGISTRY:
        return _build_error_result(f"Unknown adapter: {adapter_name}")

    adapter = ADAPTER_REGISTRY[adapter_name]()
    task_config = dict(agent_config.config)
    task_config['prompt'] = prompt
    task_config['system_prompt'] = system_prompt

    try:
        result = validate_adapter_result(await adapter.run(task_config, {}), adapter_name)
        if result['success']:
            raw_response = _extract_raw_response(result.get('output_files', {}))
            return _build_success_result(raw_response, result.get('metadata', {}))
        return _build_error_result(result.get('summary', 'Unknown error'))
    except Exception as e:
        log.error(f"Error running {adapter_name}: {e}", exc_info=True)
        return _build_error_result(f"{adapter_name}: {e}")


def _get_agent_config(cfg: DictConfig, agent_name: str) -> DictConfig:
    """Extract agent config, handling nested 'agent' key from Hydra defaults."""
    agent_entry = cfg.agents[agent_name]
    return agent_entry.agent if hasattr(agent_entry, 'agent') else agent_entry


async def run_experiment(cfg: DictConfig) -> Dict[str, Any]:
    """Run the full experiment based on Hydra config."""
    if "active_agents" not in cfg and not OmegaConf.select(cfg, "trace_extraction.enabled", default=False):
        available = sorted(path.stem for path in (Path(__file__).parent / "configs/experiment").glob("*.yaml"))
        raise SystemExit(
            "No experiment selected. Run: manyagents experiment=<name>. Available: "
            + ", ".join(available)
        )

    # Dispatch to trace extraction if configured
    if hasattr(cfg, "trace_extraction") and getattr(cfg.trace_extraction, "enabled", False):
        return await _run_trace_extraction(cfg)

    from manyagents.adapters import ADAPTER_REGISTRY

    if not cfg.active_agents:
        log.error("active_agents must not be empty")
        raise SystemExit(1)
    unknown_agents = [name for name in cfg.active_agents if name not in cfg.agents]
    if unknown_agents:
        log.error(
            f"Unknown active agent(s): {', '.join(unknown_agents)}. "
            f"Configured agents: {', '.join(cfg.agents)}"
        )
        raise SystemExit(1)

    for agent_name in cfg.active_agents:
        adapter_name = _get_agent_config(cfg, agent_name).adapter
        adapter_class = ADAPTER_REGISTRY.get(adapter_name)
        if adapter_class is None:
            log.error(f"Unknown adapter: {adapter_name} (agent '{agent_name}')")
            raise SystemExit(1)
        if not adapter_class.PRODUCES_TEXT_RESPONSE:
            log.error(f"Adapter '{adapter_name}' does not produce text responses; cannot evaluate '{agent_name}'")
            raise SystemExit(1)

    experiment_id = f"{cfg.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = _create_logger(cfg, experiment_id)

    log.info(f"Starting experiment: {experiment_id}")
    log.info(f"Active agents: {cfg.active_agents}")
    log.info(f"Prompts: {list(cfg.prompts.keys())}")

    all_results = {agent: {} for agent in cfg.active_agents}
    prompts_dict = {
        prompt_id: OmegaConf.to_container(prompt_cfg, resolve=True)
        for prompt_id, prompt_cfg in cfg.prompts.items()
    }

    logger.log_config(prompts_dict, list(cfg.active_agents), cfg.system_prompt, None)

    for prompt_id, prompt_config in prompts_dict.items():
        prompt_text = prompt_config.get('text', prompt_config.get('prompt', ''))
        log.info(f"Running prompt: {prompt_id}")

        # Build and run agent tasks in parallel
        tasks = [
            (name, _run_agent(_get_agent_config(cfg, name), prompt_text, cfg.system_prompt))
            for name in cfg.active_agents
        ]

        gathered = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for (agent_name, _), result in zip(tasks, gathered):
            if isinstance(result, Exception):
                result = _build_error_result(str(result))

            _add_ground_truth_matching(result, prompt_config)
            all_results[agent_name][prompt_id] = result
            log.info(f"  {agent_name}: {'SUCCESS' if result.get('success') else 'FAILED'}")
            logger.log_prompt_result(agent_name, prompt_id, result)

    # Compute and log metrics
    metrics = {
        name: compute_system_metrics(all_results[name], prompts_dict)
        for name in cfg.active_agents if name in all_results
    }

    for agent_name, agent_metrics in metrics.items():
        logger.log_system_metrics(agent_name, agent_metrics)
    logger.log_summary_table(metrics)

    experiment_results = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'prompts': prompts_dict,
        'results': all_results,
        'metrics': metrics
    }

    output_dir = Path(cfg.output_dir)
    _save_results(experiment_results, output_dir, experiment_id)

    logger.create_visualizations(experiment_results)
    logger.save_artifacts(experiment_results, output_dir, experiment_id)

    if wandb_url := logger.finish():
        experiment_results['wandb_url'] = wandb_url
        log.info(f"wandb run: {wandb_url}")

    _print_summary(metrics)
    if not any(m['prompts_evaluated'] for m in metrics.values()):
        log.error("No successful evaluations")
        for agent_name, m in metrics.items():
            log.error(f"{agent_name}: {m['prompts_evaluated']} succeeded, {m['prompts_failed']} failed")
        raise SystemExit(1)
    return experiment_results
