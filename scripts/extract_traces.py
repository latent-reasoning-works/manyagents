#!/usr/bin/env python3
"""Extract reasoning traces with hidden states from a local LLM.

Loads a model from cluster weights, runs CoT reasoning on a task set,
captures per-step hidden states, and saves via TraceStore.

Usage:
    python scripts/extract_traces.py \
        --model olmo-7b \
        --n-samples 10 \
        --output-dir outputs/traces/olmo_gsm8k_test

    # Full run
    python scripts/extract_traces.py \
        --model olmo-7b \
        --n-samples 500 \
        --layers -1 \
        --output-dir outputs/traces/olmo_gsm8k_500
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent to path so we can import manyagents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manyagents import inference
from manyagents.schemas.reasoning import (
    ModelBackend,
    ModelInfo,
    StepKind,
    TaskInfo,
    ReasoningStep,
    ReasoningTrace,
    TraceStore,
)

log = logging.getLogger(__name__)

COT_SYSTEM = "Solve the problem step by step. Show your reasoning clearly, with each step on a new line."

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(n_samples: int) -> list[dict]:
    """Load GSM8K training set samples."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    samples = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        # Extract numeric answer after ####
        answer = row["answer"].split("####")[-1].strip()
        samples.append({
            "task_id": f"gsm8k_train_{i}",
            "prompt": row["question"],
            "expected_answer": answer,
            "domain": "math",
            "logic_type": "arithmetic",
        })
    return samples


def load_tasks(dataset: str, n_samples: int) -> list[dict]:
    """Load task samples from a dataset."""
    loaders = {
        "gsm8k": load_gsm8k,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(loaders.keys())}")
    return loaders[dataset](n_samples)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_trace(
    model,
    tokenizer,
    task: dict,
    model_name: str,
    model_path: str,
    max_new_tokens: int,
    temperature: float,
    layers: list[int] | None,
    step_delimiter: str,
) -> tuple[ReasoningTrace, dict[str, np.ndarray] | None]:
    """Run one task through the model and produce a ReasoningTrace + hidden states."""
    prompt = inference.build_prompt(tokenizer, task["prompt"], system_prompt=COT_SYSTEM)

    gen = inference.generate_with_hidden_states(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        layers=layers,
    )

    # Split response into reasoning steps
    step_defs = inference.split_into_steps(gen["text"], tokenizer, step_delimiter)
    if not step_defs:
        # Fallback: treat entire response as one step
        step_defs = [{"text": gen["text"], "token_start": 0,
                       "token_end": gen["n_new_tokens"]}]

    # Pool hidden states per step
    pooled = inference.pool_hidden_states_per_step(gen["token_hidden_states"], step_defs)

    # Build ReasoningSteps
    steps = []
    for i, sd in enumerate(step_defs):
        kind = StepKind.OUTPUT if i == len(step_defs) - 1 else StepKind.THINKING
        steps.append(ReasoningStep(
            index=i,
            text=sd["text"],
            kind=kind,
            token_count=sd["token_end"] - sd["token_start"],
            has_hidden_states=True,
            layers_captured=gen["layers_captured"],
        ))

    trace = ReasoningTrace(
        model=ModelInfo(
            name=model_name,
            backend=ModelBackend.LOCAL,
            path=model_path,
            generation_config={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        ),
        task=TaskInfo(
            dataset="gsm8k",
            task_id=task["task_id"],
            prompt=task["prompt"],
            expected_answer=task.get("expected_answer"),
            domain=task.get("domain"),
            logic_type=task.get("logic_type"),
        ),
        steps=steps,
        response_text=gen["text"],
        input_tokens=gen["input_length"],
        output_tokens=gen["n_new_tokens"],
        total_tokens=gen["input_length"] + gen["n_new_tokens"],
        duration_ms=gen["generation_time_ms"],
    )

    # hidden_states dict for TraceStore
    hs = {
        "pooled_steps": pooled.astype(np.float16),        # (n_steps, n_layers, d_model)
        "token_level": gen["token_hidden_states"].astype(np.float16),  # (n_tokens, n_layers, d_model)
    }

    return trace, hs


def main():
    parser = argparse.ArgumentParser(description="Extract reasoning traces with hidden states")
    parser.add_argument("--model", default="olmo-7b", help="Model name or path")
    parser.add_argument("--dataset", default="gsm8k", help="Dataset to use")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--output-dir", default="outputs/traces/test", help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layer indices to capture (default: all). Use -1 for last layer.")
    parser.add_argument("--step-delimiter", default="\n", help="How to split CoT into steps")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--no-tensors", action="store_true", help="Skip hidden state capture")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load tasks
    log.info(f"Loading {args.n_samples} samples from {args.dataset}...")
    tasks = load_tasks(args.dataset, args.n_samples)
    log.info(f"Loaded {len(tasks)} tasks")

    # Load model
    import torch
    torch_dtype = getattr(torch, args.dtype)
    model_path = inference.resolve_model_path(args.model)
    model, tokenizer, _ = inference.load_model(model_path, dtype=torch_dtype)

    # Extract traces
    log.info(f"Extracting traces → {args.output_dir}")
    with TraceStore(args.output_dir) as store:
        for i, task in enumerate(tasks):
            log.info(f"[{i+1}/{len(tasks)}] {task['task_id']}")
            try:
                trace, hs = extract_trace(
                    model, tokenizer, task,
                    model_name=args.model,
                    model_path=model_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    layers=args.layers,
                    step_delimiter=args.step_delimiter,
                )
                store.append(trace, hidden_states=None if args.no_tensors else hs)
                log.info(f"  → {len(trace.steps)} steps, {trace.output_tokens} tokens, "
                         f"{trace.duration_ms}ms")
            except Exception as e:
                log.error(f"  FAILED: {e}", exc_info=True)
                continue

    # Print summary
    store = TraceStore(args.output_dir, mode="r")
    summary = store.summary()
    print("\n" + "=" * 60)
    print("TRACE EXTRACTION COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    # Print a sample trace
    for trace in store:
        print(f"\n--- Sample trace: {trace.trace_id} ---")
        print(f"Task: {trace.task.prompt[:80]}...")
        print(f"Steps: {len(trace.steps)}")
        for s in trace.steps[:3]:
            print(f"  [{s.kind.value}] {s.text[:60]}...")
        if len(trace.steps) > 3:
            print(f"  ... ({len(trace.steps) - 3} more)")
        print(f"Tokens: {trace.input_tokens} in, {trace.output_tokens} out")
        print(f"Time: {trace.duration_ms}ms")

        if trace.has_tensors:
            tensors = store.load_tensors(trace.trace_id)
            if tensors:
                ps = tensors["pooled_steps"]
                print(f"Hidden states: pooled_steps {ps.shape} "
                      f"({ps.nbytes / 1024:.0f}KB)")
        break


if __name__ == "__main__":
    main()
