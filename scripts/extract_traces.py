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
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path so we can import manyagents
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# Cluster model paths
MODEL_PATHS = {
    "olmo-7b": "/network/weights/olmo/OLMo-7B-Twin-2T",
    "olmo-1b": "/network/weights/olmo/OLMo-1B-Twin-2T",
    "olmoe-1b-7b": "/network/weights/olmoe/OLMoE-1B-7B-0924",
    "llama-3.1-8b": "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct",
}

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
# Model loading and generation with hidden state capture
# ---------------------------------------------------------------------------

def load_model(model_name: str, dtype: str = "bfloat16"):
    """Load model and tokenizer from cluster weights."""
    model_path = MODEL_PATHS.get(model_name, model_name)
    log.info(f"Loading {model_name} from {model_path}...")

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Model loaded. Layers: {model.config.num_hidden_layers}, "
             f"d_model: {model.config.hidden_size}")
    return model, tokenizer


def build_prompt(tokenizer, question: str) -> str:
    """Build a CoT prompt using the model's chat template or fallback."""
    messages = [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback
    return f"System: {COT_SYSTEM}\n\nUser: {question}\n\nAssistant:"


@torch.no_grad()
def generate_with_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    layers: list[int] | None = None,
) -> dict:
    """Generate a response and capture hidden states for each new token.

    Returns dict with:
        response_text: str
        token_hidden_states: np.ndarray of shape (n_new_tokens, n_layers, d_model)
            or (n_new_tokens, len(layers), d_model) if layers specified
        input_length: int
        generation_time_ms: int
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    input_length = input_ids.shape[1]

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    d_model = model.config.hidden_size

    if layers is None:
        layer_indices = list(range(n_layers))
    else:
        # Resolve negative indices
        layer_indices = [l % n_layers for l in layers]

    # Collect hidden states token by token via generate hooks
    all_hidden_states = []

    start = time.time()

    # Use model.generate with output_hidden_states=True
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    generation_time_ms = int((time.time() - start) * 1000)

    # outputs.hidden_states is a tuple of length n_new_tokens
    # Each element is a tuple of (n_layers+1) tensors of shape (batch, seq_len, d_model)
    # For generate, seq_len=1 for each step after the first
    for step_hidden in outputs.hidden_states:
        # step_hidden is tuple of (n_layers+1) tensors
        # Take selected layers, squeeze batch and seq dims
        step_layers = []
        for li in layer_indices:
            h = step_hidden[li]  # (batch, seq_len, d_model)
            # For the first step, seq_len = input_length; take last token
            # For subsequent steps, seq_len = 1
            step_layers.append(h[0, -1, :].cpu().float().numpy())
        all_hidden_states.append(np.stack(step_layers))  # (n_selected_layers, d_model)

    # Stack: (n_new_tokens, n_selected_layers, d_model)
    token_hidden_states = np.stack(all_hidden_states)

    # Decode response
    new_token_ids = outputs.sequences[0, input_length:]
    response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    return {
        "response_text": response_text,
        "token_hidden_states": token_hidden_states,
        "input_length": input_length,
        "generation_time_ms": generation_time_ms,
        "n_new_tokens": len(all_hidden_states),
        "layers_captured": layer_indices,
    }


# ---------------------------------------------------------------------------
# Step boundary detection and per-step pooling
# ---------------------------------------------------------------------------

def split_into_steps(text: str, tokenizer, delimiter: str = "\n") -> list[dict]:
    """Split response text into reasoning steps.

    Returns list of dicts with:
        text: str
        token_start: int (inclusive, relative to response start)
        token_end: int (exclusive)
    """
    lines = text.split(delimiter)
    steps = []
    char_pos = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            char_pos += len(line) + len(delimiter)
            continue

        # Tokenize up to and including this line to find token boundaries
        prefix = text[:char_pos + len(line)]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)

        if not steps:
            token_start = 0
        else:
            token_start = steps[-1]["token_end"]

        steps.append({
            "text": stripped,
            "token_start": token_start,
            "token_end": len(prefix_tokens),
        })

        char_pos += len(line) + len(delimiter)

    return steps


def pool_hidden_states_per_step(
    token_hidden_states: np.ndarray,
    steps: list[dict],
) -> np.ndarray:
    """Mean-pool token hidden states within each reasoning step.

    Args:
        token_hidden_states: (n_tokens, n_layers, d_model)
        steps: list of dicts with token_start and token_end

    Returns:
        (n_steps, n_layers, d_model)
    """
    n_tokens = token_hidden_states.shape[0]
    pooled = []

    for step in steps:
        start = min(step["token_start"], n_tokens - 1)
        end = min(step["token_end"], n_tokens)
        if end <= start:
            end = start + 1  # at least one token
        pooled.append(token_hidden_states[start:end].mean(axis=0))

    return np.stack(pooled)


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
    prompt = build_prompt(tokenizer, task["prompt"])

    gen = generate_with_hidden_states(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        layers=layers,
    )

    # Split response into reasoning steps
    step_defs = split_into_steps(gen["response_text"], tokenizer, step_delimiter)
    if not step_defs:
        # Fallback: treat entire response as one step
        step_defs = [{"text": gen["response_text"], "token_start": 0,
                       "token_end": gen["n_new_tokens"]}]

    # Pool hidden states per step
    pooled = pool_hidden_states_per_step(gen["token_hidden_states"], step_defs)

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
        response_text=gen["response_text"],
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
    model, tokenizer = load_model(args.model, args.dtype)
    model_path = MODEL_PATHS.get(args.model, args.model)

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
