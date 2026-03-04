"""Functional core for model loading, prompt building, and generation.

Plain functions — no classes, no async, no locking.  Both
``LocalLLMAdapter`` and ``scripts/extract_traces.py`` delegate here.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Dict, List, Optional

    import torch.nn as nn

log = logging.getLogger(__name__)

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
    d_model = model.config.hidden_size

    layer_indices = (
        [l % n_layers for l in layers] if layers is not None else list(range(n_layers))
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
# Step splitting + pooling
# ---------------------------------------------------------------------------


def split_into_steps(
    text: str,
    tokenizer,
    delimiter: str = "\n",
) -> list[dict]:
    """Split response text into reasoning steps with token boundaries.

    Returns a list of ``{"text": str, "token_start": int, "token_end": int}``.
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
        })

        char_pos += len(line) + len(delimiter)

    return steps


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
