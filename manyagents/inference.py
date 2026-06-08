"""Functional core for model loading, prompt building, and generation.

Plain functions — no classes, no async, no locking.  Both
``LocalLLMAdapter`` and ``scripts/extract_traces.py`` delegate here.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:

    import torch.nn as nn

    from .schemas.reasoning import ModelBackend, ReasoningTrace, TaskInfo

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple] = {}


def get_model(model: str, dtype=None, device_map: str = "auto"):
    """Cached model loading. Resolves name/HF ID, loads once per process.

    Args:
        model: Short name, HF Hub ID, or filesystem path.
        dtype: torch dtype (default: bfloat16).
        device_map: Device placement strategy.

    Returns:
        (network, tokenizer, hf_module) — same as load_model().
    """
    model_path = resolve_model_path(model)
    if model_path not in _model_cache:
        log.info(f"Cache miss for '{model}' -> loading from {model_path}")
        _model_cache[model_path] = load_model(model_path, dtype=dtype, device_map=device_map)
    else:
        log.debug(f"Cache hit for '{model}' ({model_path})")
    return _model_cache[model_path]


def clear_model_cache():
    """Free GPU memory by clearing the model cache (HF + vLLM engines)."""
    _model_cache.clear()
    _vllm_cache.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# vLLM engine cache
# ---------------------------------------------------------------------------
#
# A vLLM ``LLM`` engine is expensive to build and fixes its engine args at
# construction (``max_num_batched_tokens``, ``gpu_memory_utilization``, ...).
# We cache one engine per (resolved model path, engine-arg) combination so the
# same generation settings reuse a warm engine, mirroring ``_model_cache``.

_vllm_cache: dict[tuple, object] = {}


def _hashable(v):
    """Make engine-arg values hashable for the cache key (dicts/lists -> tuples)."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def get_vllm_engine(
    model: str,
    *,
    dtype: str = "bfloat16",
    max_num_batched_tokens: "int | None" = None,
    max_num_seqs: "int | None" = None,
    max_model_len: "int | None" = None,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    trust_remote_code: bool = True,
    seed: int = 0,
    **engine_kwargs,
):
    """Cached vLLM ``LLM`` engine. Resolves name/HF ID/path, builds once.

    Every engine-level knob is exposed and threaded into ``vllm.LLM``:
    scheduler capacity (``max_num_batched_tokens``, ``max_num_seqs``), context
    length (``max_model_len``), memory (``gpu_memory_utilization``), sharding
    (``tensor_parallel_size``), and CUDA-graph capture (``enforce_eager``).
    Anything else passes through via ``engine_kwargs`` (e.g. ``quantization``,
    ``kv_cache_dtype``, ``swap_space``).

    Args:
        model: Short name, HF Hub ID, or filesystem path (see resolve_model_path).
        dtype: vLLM weight/activation dtype ("auto", "bfloat16", "float16", ...).
        max_num_batched_tokens: Scheduler token budget per iteration (None = vLLM default).
        max_num_seqs: Max concurrent sequences per iteration.
        max_model_len: Max context length (prompt + generated).
        gpu_memory_utilization: Fraction of GPU memory for weights + KV cache.
        tensor_parallel_size: Number of GPUs to shard across.
        enforce_eager: Disable CUDA-graph capture (lower memory, slower).
        trust_remote_code: Allow custom modeling code (needed for some HF models).
        seed: Engine seed for reproducible sampling.
        **engine_kwargs: Any other ``vllm.LLM`` / ``EngineArgs`` field.

    Returns:
        A cached ``vllm.LLM`` instance.
    """
    model_path = resolve_model_path(model)

    args: dict = dict(
        model=model_path,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
        seed=seed,
    )
    # Only pass capacity knobs when set, so vLLM's own defaults apply otherwise.
    if max_num_batched_tokens is not None:
        args["max_num_batched_tokens"] = max_num_batched_tokens
    if max_num_seqs is not None:
        args["max_num_seqs"] = max_num_seqs
    if max_model_len is not None:
        args["max_model_len"] = max_model_len
    args.update(engine_kwargs)

    key = (model_path, tuple(sorted((k, _hashable(v)) for k, v in args.items())))
    if key not in _vllm_cache:
        from vllm import LLM

        log.info(f"vLLM cache miss for '{model}' -> building LLM({model_path})")
        _vllm_cache[key] = LLM(**args)
    else:
        log.debug(f"vLLM cache hit for '{model}' ({model_path})")
    return _vllm_cache[key]


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATHS: dict[str, str] = {
    "llama-3.3-70b": "/network/weights/llama.var/llama_3.3/Llama-3.3-70B-Instruct",
    "llama-3.1-70b": "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-70B-Instruct",
    "llama-3.1-8b": "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-405b-fp8": "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-405B-Instruct-FP8",
    "olmo-7b": "/network/weights/olmo/OLMo-7B-Twin-2T",
    "olmo-1b": "/network/weights/olmo/OLMo-1B-Twin-2T",
    "olmoe-1b-7b": "/network/weights/olmoe/OLMoE-1B-7B-0924",
}


def resolve_model_path(model_name: str) -> str:
    """Resolve a short model name to a filesystem path or HF Hub ID.

    Returns the corresponding local path for known aliases, the name
    itself if it looks like a HF Hub ID (contains ``/``), or an existing
    filesystem path.  Raises ``ValueError`` otherwise.
    """
    from pathlib import Path

    if model_name in DEFAULT_MODEL_PATHS:
        return DEFAULT_MODEL_PATHS[model_name]
    if "/" in model_name:
        return model_name  # HF Hub ID (e.g. "Qwen/Qwen3-4B")
    if Path(model_name).exists():
        return model_name
    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Available: {list(DEFAULT_MODEL_PATHS.keys())}, "
        "a HF Hub ID (org/model), or a full path."
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_path: str,
    device_map: str = "auto",
    dtype=None,
    trust_remote_code: bool = True,
):
    """Load a model and tokenizer via ``HFTrainerModule``.

    Returns ``(network, tokenizer, hf_module)`` where *network* is the
    underlying ``nn.Module`` ready for inference.
    """
    import torch
    from manylatents.lightning.hf_trainer import HFTrainerConfig, HFTrainerModule

    if dtype is None:
        dtype = torch.bfloat16

    config = HFTrainerConfig(
        model_name_or_path=model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    hf_module = HFTrainerModule(config)
    hf_module.configure_model()
    return hf_module.network, hf_module.tokenizer, hf_module


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(
    tokenizer,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """Build a chat-formatted prompt via ``tokenizer.apply_chat_template``
    with a plain-text fallback.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback for models without a chat template
    parts = [f"System: {system_prompt}\n\n"] if system_prompt else []
    parts.append(f"User: {prompt}\n\nAssistant:")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------

LAYER_PATH_DEFAULTS: dict[str, str] = {
    "olmo": "model.transformer.blocks[-1]",
    "llama": "model.layers[-1]",
    "gpt2": "transformer.h[-1]",
    "qwen": "model.layers[-1]",
}


def resolve_layer_specs(
    model_name: str,
    layer_specs: list[str] | None = None,
    reduce: str = "last_token",
):
    """Build a list of ``LayerSpec`` for ``ActivationExtractor``.

    If *layer_specs* is given, those paths are used directly.  Otherwise the
    model family is auto-detected from *model_name*.
    """
    from manylatents.lightning.hooks import LayerSpec

    if layer_specs:
        return [LayerSpec(path=p, reduce=reduce) for p in layer_specs]

    lower = model_name.lower()
    for prefix, path in LAYER_PATH_DEFAULTS.items():
        if prefix in lower:
            return [LayerSpec(path=path, reduce=reduce)]

    log.warning(
        f"No default layer path for '{model_name}'; "
        "pass layer_specs explicitly for hidden-state capture."
    )
    return []


# ---------------------------------------------------------------------------
# Generation — plain text
# ---------------------------------------------------------------------------


def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Plain text generation.  Returns the decoded response string."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Generation — hook-based hidden-state capture
# ---------------------------------------------------------------------------


def generate_with_hooks(
    model: nn.Module,
    tokenizer,
    prompt: str,
    layer_specs,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    """Generate text and capture hidden states via ``ActivationExtractor`` hooks.

    Returns::

        {
            "text": str,
            "hidden_states": {layer_path: ndarray(n_tokens, d)},
        }
    """
    import torch
    from manylatents.lightning.hooks import ActivationExtractor

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    extractor = ActivationExtractor(layer_specs)
    with torch.no_grad(), extractor.capture(model):
        outputs = model.generate(**gen_kwargs)

    acts = extractor.get_activations()
    hidden_states_np = {
        path: t.cpu().numpy().astype(np.float16)
        for path, t in acts.items()
    }

    input_length = inputs["input_ids"].shape[1]
    text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    return {"text": text, "hidden_states": hidden_states_np}


# ---------------------------------------------------------------------------
# Generation — HF built-in output_hidden_states capture
# ---------------------------------------------------------------------------


def generate_with_hidden_states(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    layers: list[int] | None = None,
) -> dict:
    """Generate text and capture hidden states via HF ``output_hidden_states=True``.

    Returns::

        {
            "text": str,
            "token_hidden_states": ndarray(n_new_tokens, n_layers, d_model),
            "input_length": int,
            "n_new_tokens": int,
            "generation_time_ms": int,
            "layers_captured": list[int],
        }
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    input_length = input_ids.shape[1]

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    layer_indices = (
        [layer % n_layers for layer in layers] if layers is not None else list(range(n_layers))
    )

    do_sample = temperature > 0
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generation_time_ms = int((time.time() - start) * 1000)

    # outputs.hidden_states: tuple of length n_new_tokens, each element a
    # tuple of (n_layers+1) tensors of shape (batch, seq_len, d_model).
    all_hidden_states = []
    for step_hidden in outputs.hidden_states:
        step_layers = []
        for li in layer_indices:
            h = step_hidden[li]  # (batch, seq_len, d_model)
            step_layers.append(h[0, -1, :].cpu().float().numpy())
        all_hidden_states.append(np.stack(step_layers))  # (n_selected, d_model)

    token_hidden_states = np.stack(all_hidden_states)  # (n_new_tokens, n_selected, d_model)

    new_token_ids = outputs.sequences[0, input_length:]
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    return {
        "text": text,
        "token_hidden_states": token_hidden_states,
        "input_length": input_length,
        "n_new_tokens": len(all_hidden_states),
        "generation_time_ms": generation_time_ms,
        "layers_captured": layer_indices,
    }


# ---------------------------------------------------------------------------
# Generation — vLLM (fast generation; hidden states via HF forward pass)
# ---------------------------------------------------------------------------


def build_sampling_params(
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: "int | None" = None,
    stop: "list[str] | None" = None,
    overrides: "dict | None" = None,
):
    """Build a ``vllm.SamplingParams`` from explicit knobs + a free-form override dict.

    The common decoding knobs are first-class arguments; ``overrides`` is merged
    last and passes through to *any* ``SamplingParams`` field (``min_p``,
    ``repetition_penalty``, ``presence_penalty``, ``frequency_penalty``, ``n``,
    ``logprobs``, ``stop_token_ids``, ...) and wins on conflict. ``max_new_tokens``
    maps onto vLLM's ``max_tokens``.

    Returns:
        A ``vllm.SamplingParams`` instance.
    """
    from vllm import SamplingParams

    params: dict = dict(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    if seed is not None:
        params["seed"] = seed
    if stop is not None:
        params["stop"] = stop
    if overrides:
        params.update(overrides)
    return SamplingParams(**params)


def vllm_generate(
    prompts: "str | list[str]",
    *,
    engine=None,
    model: "str | None" = None,
    sampling_params=None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: "int | None" = None,
    stop: "list[str] | None" = None,
    sampling_overrides: "dict | None" = None,
    engine_kwargs: "dict | None" = None,
) -> list[dict]:
    """Generate with vLLM. Returns token ids alongside text for exact re-encoding.

    Provide either a prebuilt ``engine`` (a ``vllm.LLM``) or a ``model`` name
    (an engine is fetched/built via ``get_vllm_engine(model, **engine_kwargs)``).
    Sampling is controlled by ``sampling_params`` (a ``vllm.SamplingParams`` or a
    plain dict of overrides) or the explicit knobs; see ``build_sampling_params``.

    Every result dict carries ``prompt_token_ids`` and ``completion_token_ids``
    so a downstream HF forward pass can re-encode the *exact* sequence vLLM saw —
    no string round-trip, no tokenizer drift (the key to backend="vllm" hidden
    states matching backend="hf").

    Returns:
        list of ``{text, prompt_token_ids, completion_token_ids, input_length,
        n_new_tokens, generation_time_ms, finish_reason}`` — one per prompt.
    """
    if engine is None:
        if model is None:
            raise ValueError("vllm_generate requires either `engine` or `model`.")
        engine = get_vllm_engine(model, **(engine_kwargs or {}))

    if sampling_params is None:
        sampling_params = build_sampling_params(
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, seed=seed, stop=stop,
            overrides=sampling_overrides,
        )
    elif isinstance(sampling_params, dict):
        sampling_params = build_sampling_params(
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, seed=seed, stop=stop,
            overrides={**sampling_params, **(sampling_overrides or {})},
        )

    single = isinstance(prompts, str)
    prompt_list = [prompts] if single else list(prompts)

    start = time.time()
    request_outputs = engine.generate(prompt_list, sampling_params)
    generation_time_ms = int((time.time() - start) * 1000)

    results: list[dict] = []
    for ro in request_outputs:
        out = ro.outputs[0]
        prompt_ids = list(ro.prompt_token_ids)
        completion_ids = list(out.token_ids)
        results.append({
            "text": out.text,
            "prompt_token_ids": prompt_ids,
            "completion_token_ids": completion_ids,
            "input_length": len(prompt_ids),
            "n_new_tokens": len(completion_ids),
            "generation_time_ms": generation_time_ms,
            "finish_reason": out.finish_reason,
        })
    return results


def forward_hidden_states(
    model: "nn.Module",
    input_ids: "list[int]",
    *,
    input_length: int,
    layers: "list[int] | None" = None,
) -> dict:
    """Teacher-forced forward pass over a fixed sequence; harvest per-token states.

    Used for backend="vllm": vLLM generates, then a single HF forward pass over
    ``prompt + completion`` token ids recovers hidden states. For a causal model
    over a fixed sequence, the state at position *t* equals the decode-time state
    (causal masking), so the completion-aligned slice reproduces what
    ``generate_with_hidden_states`` would return had HF generated the same tokens.

    Args:
        model: HF causal LM (provides hidden states).
        input_ids: Full sequence — ``prompt_token_ids + completion_token_ids``.
        input_length: Length of the prompt prefix (``len(prompt_token_ids)``).
        layers: Layer indices to keep (negatives allowed); None = all layers.

    Returns:
        Same shape contract as ``generate_with_hidden_states`` minus ``text``:
        ``{token_hidden_states (n_new, n_selected, d_model), input_length,
        n_new_tokens, generation_time_ms, layers_captured}`` (float32).
    """
    import torch

    ids = torch.tensor([list(input_ids)], device=model.device)
    total = ids.shape[1]
    n_new = total - input_length

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    layer_indices = (
        [layer % n_layers for layer in layers] if layers is not None else list(range(n_layers))
    )

    start = time.time()
    with torch.no_grad():
        outputs = model(ids, output_hidden_states=True, use_cache=False)
    forward_time_ms = int((time.time() - start) * 1000)

    # Position (input_length - 1 + s) is the state that predicts completion
    # token s — the same state generate() exposes at decode step s.
    all_hidden_states = []
    for pos in range(input_length - 1, total - 1):
        step_layers = [
            outputs.hidden_states[li][0, pos, :].cpu().float().numpy()
            for li in layer_indices
        ]
        all_hidden_states.append(np.stack(step_layers))  # (n_selected, d_model)

    if all_hidden_states:
        token_hidden_states = np.stack(all_hidden_states)
    else:  # empty completion — keep the shape contract
        d_model = model.config.hidden_size
        token_hidden_states = np.empty((0, len(layer_indices), d_model), dtype=np.float32)

    return {
        "token_hidden_states": token_hidden_states.astype(np.float32),
        "input_length": int(input_length),
        "n_new_tokens": int(n_new),
        "generation_time_ms": forward_time_ms,
        "layers_captured": layer_indices,
    }


# ---------------------------------------------------------------------------
# Step splitting + pooling
# ---------------------------------------------------------------------------


def segment_by_delimiter(
    text: str,
    tokenizer,
    delimiter: str = "\n",
) -> list[dict]:
    """Split response text into reasoning steps by delimiter with token boundaries.

    Returns a list of ``{"text", "token_start", "token_end", "kind"}``.
    """
    lines = text.split(delimiter)
    steps: list[dict] = []
    char_pos = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            char_pos += len(line) + len(delimiter)
            continue

        prefix = text[: char_pos + len(line)]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)

        token_start = steps[-1]["token_end"] if steps else 0

        steps.append({
            "text": stripped,
            "token_start": token_start,
            "token_end": len(prefix_tokens),
            "kind": "thinking",
        })

        char_pos += len(line) + len(delimiter)

    # Last step is output
    if steps:
        steps[-1]["kind"] = "output"

    return steps


# Backward-compatible alias
split_into_steps = segment_by_delimiter


# ---------------------------------------------------------------------------
# Tag-aware segmentation (<think>...</think>)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=\.)\s+(?=[A-Z])")


def segment_by_tags(
    text: str,
    tokenizer,
) -> list[dict]:
    """Parse ``<think>...</think>`` tags and sentence-split within.

    Within the think block, splits on sentence boundaries (period followed
    by whitespace and capital letter) and newlines.  Text after ``</think>``
    becomes a single OUTPUT step.

    Falls back to a single OUTPUT step when no tags are present.

    Returns a list of ``{"text", "token_start", "token_end", "kind"}``.
    """
    match = _THINK_RE.search(text)
    if not match:
        # No tags — single output step
        all_tokens = tokenizer.encode(text, add_special_tokens=False)
        return [{"text": text.strip(), "token_start": 0, "token_end": len(all_tokens), "kind": "output"}]

    think_content = match.group(1)
    after_think = text[match.end():].strip()

    # Split think content into sentences
    # First split on newlines, then on sentence boundaries within each line
    raw_parts: list[str] = []
    for line in think_content.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences = _SENTENCE_RE.split(line)
        raw_parts.extend(s.strip() for s in sentences if s.strip())

    steps: list[dict] = []
    consumed_text = text[:match.start()]  # text before <think>

    for part in raw_parts:
        consumed_text += part
        # Find this text in the original to compute token boundary
        prefix_tokens = tokenizer.encode(
            text[:text.find(part) + len(part)],
            add_special_tokens=False,
        )
        token_start = steps[-1]["token_end"] if steps else 0
        steps.append({
            "text": part,
            "token_start": token_start,
            "token_end": len(prefix_tokens),
            "kind": "thinking",
        })

    # Output step: text after </think>
    if after_think:
        all_tokens = tokenizer.encode(text, add_special_tokens=False)
        token_start = steps[-1]["token_end"] if steps else 0
        steps.append({
            "text": after_think,
            "token_start": token_start,
            "token_end": len(all_tokens),
            "kind": "output",
        })

    if not steps:
        all_tokens = tokenizer.encode(text, add_special_tokens=False)
        steps = [{"text": text.strip(), "token_start": 0, "token_end": len(all_tokens), "kind": "output"}]

    return steps


# ---------------------------------------------------------------------------
# Velocity-based segmentation (cosine distance peaks)
# ---------------------------------------------------------------------------


def segment_by_velocity(
    text: str,
    tokenizer,
    token_hidden_states: np.ndarray,
    *,
    layer: int = -1,
    min_segment_tokens: int = 5,
    prominence_factor: float = 1.0,
) -> list[dict]:
    """Segment by cosine-distance peaks between consecutive token hidden states.

    Args:
        text: Full generated text.
        tokenizer: HF tokenizer for decoding segments.
        token_hidden_states: ``(n_tokens, n_layers, d_model)``.
        layer: Which layer to compute velocity on (default: last).
        min_segment_tokens: Minimum tokens per segment.
        prominence_factor: Multiplied by median velocity to get peak prominence.

    Returns a list of ``{"text", "token_start", "token_end", "kind"}``.
    """
    from scipy.signal import find_peaks

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = min(len(token_ids), token_hidden_states.shape[0])

    if n_tokens < 2:
        return [{"text": text.strip(), "token_start": 0, "token_end": n_tokens, "kind": "output"}]

    # Extract the chosen layer's hidden states
    hs = token_hidden_states[:n_tokens, layer, :]  # (n_tokens, d_model)

    # Cosine distance between consecutive tokens
    from manylatents.metrics.trajectory_geometry import compute_cosine_velocity

    cos_dist = compute_cosine_velocity(hs)  # (n_tokens - 1,)

    median_vel = float(np.median(cos_dist))
    prominence = median_vel * prominence_factor

    peaks, _ = find_peaks(cos_dist, prominence=prominence, distance=min_segment_tokens)

    # Build segment boundaries: peaks mark the END of a segment
    # (the transition happens between token[peak] and token[peak+1])
    boundaries = sorted(set([0] + [int(p) + 1 for p in peaks] + [n_tokens]))

    steps: list[dict] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < 1:
            continue
        seg_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        if not seg_text:
            continue
        steps.append({
            "text": seg_text,
            "token_start": start,
            "token_end": end,
            "kind": "thinking",
        })

    # Last step is output
    if steps:
        steps[-1]["kind"] = "output"

    if not steps:
        return [{"text": text.strip(), "token_start": 0, "token_end": n_tokens, "kind": "output"}]

    return steps


# ---------------------------------------------------------------------------
# Hybrid segmentation (tags + velocity within thinking)
# ---------------------------------------------------------------------------


def segment_hybrid(
    text: str,
    tokenizer,
    token_hidden_states: np.ndarray,
    *,
    layer: int = -1,
    min_segment_tokens: int = 5,
    prominence_factor: float = 1.0,
) -> list[dict]:
    """Combine tag-aware structure with velocity-based sub-segmentation.

    Parses ``<think>...</think>`` for coarse THINKING vs OUTPUT regions.
    Within the THINKING region, applies velocity-based sub-segmentation.
    OUTPUT is kept as a single step.

    Falls back to pure velocity when no tags are present.
    """
    match = _THINK_RE.search(text)
    if not match:
        return segment_by_velocity(
            text, tokenizer, token_hidden_states,
            layer=layer, min_segment_tokens=min_segment_tokens,
            prominence_factor=prominence_factor,
        )

    after_think = text[match.end():].strip()

    # Tokenize full text to get token ranges
    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    think_start_text = text[:match.start()] + "<think>"
    think_start_tokens = len(tokenizer.encode(think_start_text, add_special_tokens=False))
    think_end_text = text[:match.end() - len("</think>")]
    think_end_tokens = len(tokenizer.encode(think_end_text, add_special_tokens=False))

    n_think_tokens = think_end_tokens - think_start_tokens
    if n_think_tokens < 2:
        # Not enough tokens in thinking region, fall back to tags
        return segment_by_tags(text, tokenizer)

    # Sub-segment the thinking region with velocity
    think_hs = token_hidden_states[think_start_tokens:think_end_tokens]
    think_token_ids = full_tokens[think_start_tokens:think_end_tokens]

    from scipy.signal import find_peaks

    hs = think_hs[:, layer, :]  # (n_think_tokens, d_model)
    from manylatents.metrics.trajectory_geometry import compute_cosine_velocity

    cos_dist = compute_cosine_velocity(hs)

    median_vel = float(np.median(cos_dist))
    prominence = median_vel * prominence_factor

    peaks, _ = find_peaks(cos_dist, prominence=prominence, distance=min_segment_tokens)
    boundaries = sorted(set([0] + [int(p) + 1 for p in peaks] + [len(think_token_ids)]))

    steps: list[dict] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < 1:
            continue
        seg_text = tokenizer.decode(think_token_ids[start:end], skip_special_tokens=True).strip()
        if not seg_text:
            continue
        steps.append({
            "text": seg_text,
            "token_start": think_start_tokens + start,
            "token_end": think_start_tokens + end,
            "kind": "thinking",
        })

    # Output step: text after </think>
    if after_think:
        steps.append({
            "text": after_think,
            "token_start": think_end_tokens,
            "token_end": len(full_tokens),
            "kind": "output",
        })

    if not steps:
        all_tokens = tokenizer.encode(text, add_special_tokens=False)
        return [{"text": text.strip(), "token_start": 0, "token_end": len(all_tokens), "kind": "output"}]

    return steps


# ---------------------------------------------------------------------------
# Segmentation dispatcher
# ---------------------------------------------------------------------------


def segment(
    text: str,
    tokenizer,
    segmentation: str = "delimiter",
    *,
    delimiter: str = "\n",
    token_hidden_states: np.ndarray | None = None,
    layer: int = -1,
    min_segment_tokens: int = 5,
    prominence_factor: float = 1.0,
) -> list[dict]:
    """Dispatch to the appropriate segmentation function.

    Args:
        segmentation: One of ``"delimiter"``, ``"tags"``, ``"velocity"``, ``"hybrid"``.
        token_hidden_states: Required for ``"velocity"`` and ``"hybrid"``.

    Returns a list of ``{"text", "token_start", "token_end", "kind"}``.
    """
    if segmentation == "delimiter":
        return segment_by_delimiter(text, tokenizer, delimiter)
    elif segmentation == "tags":
        return segment_by_tags(text, tokenizer)
    elif segmentation == "velocity":
        if token_hidden_states is None:
            raise ValueError("segment_by_velocity requires token_hidden_states")
        return segment_by_velocity(
            text, tokenizer, token_hidden_states,
            layer=layer, min_segment_tokens=min_segment_tokens,
            prominence_factor=prominence_factor,
        )
    elif segmentation == "hybrid":
        if token_hidden_states is None:
            raise ValueError("segment_hybrid requires token_hidden_states")
        return segment_hybrid(
            text, tokenizer, token_hidden_states,
            layer=layer, min_segment_tokens=min_segment_tokens,
            prominence_factor=prominence_factor,
        )
    else:
        raise ValueError(
            f"Unknown segmentation '{segmentation}'. "
            "Choose from: delimiter, tags, velocity, hybrid"
        )


def pool_hidden_states_per_step(
    token_hidden_states: np.ndarray,
    steps: list[dict],
) -> np.ndarray:
    """Mean-pool token-level hidden states within each reasoning step.

    Args:
        token_hidden_states: ``(n_tokens, n_layers, d_model)``
        steps: list of dicts with ``token_start`` and ``token_end``

    Returns:
        ``(n_steps, n_layers, d_model)``
    """
    n_tokens = token_hidden_states.shape[0]
    pooled = []

    for step in steps:
        start = min(step["token_start"], n_tokens - 1)
        end = min(step["token_end"], n_tokens)
        if end <= start:
            end = start + 1
        pooled.append(token_hidden_states[start:end].mean(axis=0))

    return np.stack(pooled)


# ---------------------------------------------------------------------------
# Trace building
# ---------------------------------------------------------------------------


def build_reasoning_trace(
    text: str,
    gen_metadata: dict,
    model_name: str,
    model_path: str,
    task: "TaskInfo",
    step_defs: list[dict],
    generation_config: dict,
    backend: "ModelBackend | None" = None,
) -> "ReasoningTrace":
    """Build a ReasoningTrace from generation output + step definitions.

    Single source of truth for local-model trace construction.

    Args:
        text: Full generated response text.
        gen_metadata: Dict from generate_with_hidden_states() with keys
            input_length, n_new_tokens, generation_time_ms, layers_captured.
        model_name: Short model name or HF Hub ID.
        model_path: Resolved filesystem path or HF Hub ID.
        task: TaskInfo for this trace.
        step_defs: List of dicts from split_into_steps(), each with
            text, token_start, token_end.
        generation_config: Dict of generation params (temperature, etc.)
            stored in ModelInfo for reproducibility.
        backend: Which backend produced the trace (default: ModelBackend.LOCAL).

    Returns:
        A ReasoningTrace with properly typed steps.
    """
    from manyagents.schemas.reasoning import (
        ModelBackend, ModelInfo, ReasoningStep, ReasoningTrace, StepKind,
    )

    if backend is None:
        backend = ModelBackend.LOCAL

    steps = []
    for i, sd in enumerate(step_defs):
        if "kind" in sd:
            kind = StepKind(sd["kind"])
        else:
            kind = StepKind.OUTPUT if i == len(step_defs) - 1 else StepKind.THINKING
        steps.append(ReasoningStep(
            index=i,
            text=sd["text"],
            kind=kind,
            token_count=sd["token_end"] - sd["token_start"],
            has_hidden_states=True,
            layers_captured=gen_metadata.get("layers_captured", []),
        ))

    return ReasoningTrace(
        model=ModelInfo(
            name=model_name,
            backend=backend,
            path=model_path,
            generation_config=generation_config,
        ),
        task=task,
        steps=steps,
        response_text=text,
        input_tokens=gen_metadata.get("input_length", 0),
        output_tokens=gen_metadata.get("n_new_tokens", 0),
        total_tokens=(
            gen_metadata.get("input_length", 0) + gen_metadata.get("n_new_tokens", 0)
        ),
        duration_ms=gen_metadata.get("generation_time_ms"),
    )


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------


def extract_trace(
    model: "nn.Module",
    tokenizer,
    prompt: str,
    task: "TaskInfo",
    model_name: str,
    model_path: str,
    *,
    backend: str = "hf",
    vllm_engine=None,
    sampling_params=None,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = -1,
    layers: list[int] | None = None,
    step_delimiter: str = "\n",
    segmentation: str = "delimiter",
) -> tuple["ReasoningTrace", dict[str, np.ndarray]]:
    """Full trace extraction pipeline: prompt -> generate -> segment -> pool -> trace.

    Composes build_prompt, generation, segment(), pool_hidden_states_per_step,
    and build_reasoning_trace into a single call.

    Args:
        backend: ``"hf"`` (HF generates + captures hidden states) or ``"vllm"``
            (vLLM generates fast; ``model`` runs one HF forward pass over the
            exact token ids vLLM emitted to recover hidden states). For
            ``"vllm"`` pass the generation engine as ``vllm_engine`` and the HF
            model (for hidden states) as ``model``.
        sampling_params: vLLM ``SamplingParams``/override-dict (vllm backend only).
        segmentation: ``"delimiter"``, ``"tags"``, ``"velocity"``, or ``"hybrid"``.

    Returns:
        (trace, hidden_states) where hidden_states has keys:
            "pooled_steps": ndarray (n_steps, n_layers, d_model) float16
            "token_level": ndarray (n_tokens, n_layers, d_model) float16
    """
    from manyagents.schemas.reasoning import ModelBackend

    formatted = build_prompt(tokenizer, prompt, system_prompt)
    gen_config: dict = {"max_new_tokens": max_new_tokens, "temperature": temperature}

    if backend == "hf":
        gen = generate_with_hidden_states(
            model, tokenizer, formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            layers=layers,
        )
        trace_backend = ModelBackend.LOCAL
    elif backend == "vllm":
        if vllm_engine is None:
            raise ValueError("extract_trace(backend='vllm') requires `vllm_engine`.")
        out = vllm_generate(
            formatted,
            engine=vllm_engine,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )[0]
        full_ids = out["prompt_token_ids"] + out["completion_token_ids"]
        hs = forward_hidden_states(
            model, full_ids, input_length=out["input_length"], layers=layers,
        )
        gen = {**hs, "text": out["text"]}
        gen["generation_time_ms"] = out["generation_time_ms"] + hs["generation_time_ms"]
        gen_config.update(top_p=top_p, top_k=top_k)
        trace_backend = ModelBackend.VLLM
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'hf' or 'vllm'.")

    step_defs = segment(
        gen["text"],
        tokenizer,
        segmentation,
        delimiter=step_delimiter,
        token_hidden_states=gen["token_hidden_states"],
    )
    if not step_defs:
        step_defs = [{
            "text": gen["text"],
            "token_start": 0,
            "token_end": gen["n_new_tokens"],
            "kind": "output",
        }]

    pooled = pool_hidden_states_per_step(gen["token_hidden_states"], step_defs)

    trace = build_reasoning_trace(
        text=gen["text"],
        gen_metadata=gen,
        model_name=model_name,
        model_path=model_path,
        task=task,
        step_defs=step_defs,
        generation_config=gen_config,
        backend=trace_backend,
    )

    hidden_states = {
        "pooled_steps": pooled.astype(np.float16),
        "token_level": gen["token_hidden_states"].astype(np.float16),
    }

    return trace, hidden_states
