#!/usr/bin/env python3
"""vllm_trace_demo.py — minimal worked example of the vLLM backend.

vLLM generates a reasoning trajectory for Qwen3-0.6B; an HF forward pass over the
exact token ids vLLM emitted recovers per-step hidden states. Shows that every
sampling + engine knob is settable, and that the captured ReasoningTrace +
hidden-state tensors are identical in shape/semantics to the HF backend — so
downstream geometry (velocity-Gram, curvature in manylatents) consumes them
unchanged.

This stays inside manyagents' remit (inference + trace capture). Geometry metrics
and invariance-constrained training live downstream (manylatents / the
reasoning-geometry analysis scripts).

Run (GPU + `pip install manyagents[vllm]`):
    python scripts/vllm_trace_demo.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

from manyagents import inference
from manyagents.schemas.reasoning import TaskInfo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="If a train travels 60 km in 1.5 hours, what is its average speed?")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    # Engine capacity knobs — exposed straight through to vllm.LLM.
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--layers", type=int, nargs="+", default=[-1])
    p.add_argument("--segmentation", default="delimiter")
    args = p.parse_args()

    # 1. Build (cached) the vLLM engine with explicit scheduler/memory knobs.
    engine = inference.get_vllm_engine(
        args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 2. Load the HF model (hidden-state forward pass) + tokenizer.
    hf_model, tokenizer, _ = inference.get_model(args.model)

    # 3. Generate with vLLM, capture hidden states with HF — one call.
    task = TaskInfo(dataset="demo", task_id="demo-0", prompt=args.prompt)
    trace, hidden_states = inference.extract_trace(
        hf_model, tokenizer,
        prompt=args.prompt,
        task=task,
        model_name=args.model,
        model_path=inference.resolve_model_path(args.model),
        backend="vllm",
        vllm_engine=engine,
        system_prompt="Solve the problem step by step, one step per line.",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        layers=args.layers,
        segmentation=args.segmentation,
    )

    print(f"\nbackend={trace.model.backend.value}  steps={len(trace.steps)}  "
          f"output_tokens={trace.output_tokens}")
    print(f"pooled_steps {hidden_states['pooled_steps'].shape}  "
          f"token_level {hidden_states['token_level'].shape}")
    print("\n--- response ---")
    print(trace.response_text)


if __name__ == "__main__":
    main()
