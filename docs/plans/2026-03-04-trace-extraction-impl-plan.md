# Trace Extraction Consolidation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate scattered trace extraction logic into `inference.py` as a functional API, rename `LocalLLMAdapter` to `HFAdapter`, wire `ClaudeAdapter` for traces, and make trace extraction a Hydra experiment config.

**Architecture:** Move `extract_trace()` and `build_reasoning_trace()` into `inference.py` as plain functions with a module-level model cache. Adapters delegate to these functions. The standalone `extract_traces.py` script is replaced by a Hydra experiment config running through the existing `main.py` entry point.

**Tech Stack:** Python 3.12, Hydra, HuggingFace transformers, numpy, pytest

---

### Task 1: Add `build_reasoning_trace()` to `inference.py`

**Files:**
- Modify: `manyagents/inference.py` (append after line 392)
- Test: `tests/test_inference_trace.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_inference_trace.py
"""Tests for inference.py trace building functions."""

import numpy as np

from manyagents.inference import build_reasoning_trace
from manyagents.schemas.reasoning import (
    ModelBackend, StepKind, TaskInfo, ReasoningTrace,
)


def test_build_reasoning_trace_basic():
    """build_reasoning_trace produces a valid ReasoningTrace with correct steps."""
    task = TaskInfo("gsm8k", "train_0", "What is 2+2?", expected_answer="4", domain="math")
    step_defs = [
        {"text": "First, 2+2=4", "token_start": 0, "token_end": 5},
        {"text": "The answer is 4", "token_start": 5, "token_end": 10},
    ]
    gen_metadata = {
        "input_length": 20,
        "n_new_tokens": 10,
        "generation_time_ms": 500,
        "layers_captured": [32],
    }

    trace = build_reasoning_trace(
        text="First, 2+2=4\nThe answer is 4",
        gen_metadata=gen_metadata,
        model_name="olmo-7b",
        model_path="/network/weights/olmo/OLMo-7B-Twin-2T",
        task=task,
        step_defs=step_defs,
        generation_config={"max_new_tokens": 512, "temperature": 0.7},
    )

    assert isinstance(trace, ReasoningTrace)
    assert trace.model.name == "olmo-7b"
    assert trace.model.backend == ModelBackend.LOCAL
    assert trace.model.path == "/network/weights/olmo/OLMo-7B-Twin-2T"
    assert len(trace.steps) == 2
    assert trace.steps[0].kind == StepKind.THINKING
    assert trace.steps[1].kind == StepKind.OUTPUT
    assert trace.steps[0].token_count == 5
    assert trace.steps[1].layers_captured == [32]
    assert trace.input_tokens == 20
    assert trace.output_tokens == 10
    assert trace.total_tokens == 30
    assert trace.duration_ms == 500
    assert trace.task.dataset == "gsm8k"


def test_build_reasoning_trace_single_step():
    """Single-step trace marks the only step as OUTPUT."""
    task = TaskInfo("gsm8k", "train_0", "Q?")
    step_defs = [{"text": "Answer", "token_start": 0, "token_end": 3}]
    gen_metadata = {"input_length": 5, "n_new_tokens": 3, "generation_time_ms": 100, "layers_captured": []}

    trace = build_reasoning_trace(
        text="Answer",
        gen_metadata=gen_metadata,
        model_name="test",
        model_path="/test",
        task=task,
        step_defs=step_defs,
        generation_config={},
    )

    assert len(trace.steps) == 1
    assert trace.steps[0].kind == StepKind.OUTPUT
```

**Step 2: Run test to verify it fails**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_inference_trace.py::test_build_reasoning_trace_basic -v`
Expected: FAIL with `ImportError: cannot import name 'build_reasoning_trace'`

**Step 3: Write minimal implementation**

Append to `manyagents/inference.py` after line 392:

```python
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
) -> "ReasoningTrace":
    """Build a ReasoningTrace from generation output + step definitions.

    Single source of truth for local-model trace construction.
    Replaces extract_traces.py manual construction and
    LocalLLMAdapter._build_and_save_trace().

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

    Returns:
        A ReasoningTrace with properly typed steps.
    """
    from manyagents.schemas.reasoning import (
        ModelBackend, ModelInfo, ReasoningStep, ReasoningTrace, StepKind,
    )

    steps = []
    for i, sd in enumerate(step_defs):
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
            backend=ModelBackend.LOCAL,
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
```

**Step 4: Run test to verify it passes**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_inference_trace.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add manyagents/inference.py tests/test_inference_trace.py
git commit -m "feat: add build_reasoning_trace() to inference.py"
```

---

### Task 2: Add `extract_trace()` to `inference.py`

**Files:**
- Modify: `manyagents/inference.py` (append after build_reasoning_trace)
- Modify: `tests/test_inference_trace.py` (add tests)

**Step 1: Write the failing test**

Append to `tests/test_inference_trace.py`:

```python
from unittest.mock import patch, MagicMock
from manyagents.inference import extract_trace


def _mock_generate_with_hidden_states(model, tokenizer, prompt, **kwargs):
    """Mock that returns realistic generation output."""
    n_tokens = 10
    n_layers = 1
    d_model = 64
    return {
        "text": "Step one.\nStep two.\nThe answer is 4.",
        "token_hidden_states": np.random.randn(n_tokens, n_layers, d_model).astype(np.float32),
        "input_length": 20,
        "n_new_tokens": n_tokens,
        "generation_time_ms": 500,
        "layers_captured": [32],
    }


def _mock_tokenizer():
    """Mock tokenizer with encode method."""
    tok = MagicMock()
    # encode returns incrementing token IDs (1 token per ~5 chars)
    tok.encode = lambda text, add_special_tokens=False: list(range(len(text) // 5 + 1))
    tok.apply_chat_template = MagicMock(return_value="<formatted prompt>")
    return tok


def test_extract_trace_returns_trace_and_hidden_states():
    """extract_trace returns (ReasoningTrace, hidden_states_dict)."""
    model = MagicMock()
    tokenizer = _mock_tokenizer()
    task = TaskInfo("gsm8k", "train_0", "What is 2+2?", expected_answer="4")

    with patch("manyagents.inference.generate_with_hidden_states", _mock_generate_with_hidden_states):
        trace, hs = extract_trace(
            model, tokenizer,
            prompt="What is 2+2?",
            task=task,
            model_name="olmo-7b",
            model_path="/weights/olmo",
            system_prompt="Solve step by step.",
            max_new_tokens=512,
            temperature=0.7,
            layers=[-1],
        )

    # Trace checks
    assert isinstance(trace, ReasoningTrace)
    assert trace.model.name == "olmo-7b"
    assert len(trace.steps) >= 1
    assert trace.output_tokens == 10
    assert trace.duration_ms == 500

    # Hidden states checks
    assert "pooled_steps" in hs
    assert "token_level" in hs
    assert hs["pooled_steps"].dtype == np.float16
    assert hs["token_level"].dtype == np.float16
    assert hs["token_level"].shape[0] == 10  # n_tokens
    assert hs["pooled_steps"].shape[0] == len(trace.steps)


def test_extract_trace_fallback_single_step():
    """extract_trace falls back to single step when no delimiter matches."""
    model = MagicMock()
    tokenizer = _mock_tokenizer()
    task = TaskInfo("gsm8k", "train_0", "Q?")

    def _gen_no_newlines(model, tokenizer, prompt, **kwargs):
        return {
            "text": "The answer is simply four",
            "token_hidden_states": np.random.randn(5, 1, 64).astype(np.float32),
            "input_length": 10,
            "n_new_tokens": 5,
            "generation_time_ms": 200,
            "layers_captured": [32],
        }

    with patch("manyagents.inference.generate_with_hidden_states", _gen_no_newlines):
        trace, hs = extract_trace(
            model, tokenizer, prompt="Q?", task=task,
            model_name="test", model_path="/test",
        )

    assert len(trace.steps) >= 1
    assert hs["pooled_steps"].shape[0] == len(trace.steps)
```

**Step 2: Run test to verify it fails**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_inference_trace.py::test_extract_trace_returns_trace_and_hidden_states -v`
Expected: FAIL with `ImportError: cannot import name 'extract_trace'`

**Step 3: Write minimal implementation**

Append to `manyagents/inference.py` after `build_reasoning_trace`:

```python
def extract_trace(
    model: "nn.Module",
    tokenizer,
    prompt: str,
    task: "TaskInfo",
    model_name: str,
    model_path: str,
    *,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    layers: list[int] | None = None,
    step_delimiter: str = "\n",
) -> tuple["ReasoningTrace", dict[str, np.ndarray]]:
    """Full trace extraction pipeline: prompt -> generate -> split -> pool -> trace.

    Composes build_prompt, generate_with_hidden_states, split_into_steps,
    pool_hidden_states_per_step, and build_reasoning_trace into a single call.

    Args:
        model: Loaded HF model (nn.Module).
        tokenizer: Matching tokenizer.
        prompt: Raw user prompt (will be formatted via build_prompt).
        task: TaskInfo describing the task.
        model_name: Short name or HF Hub ID.
        model_path: Resolved path.
        system_prompt: Optional system prompt for CoT.
        max_new_tokens: Max generation length.
        temperature: Sampling temperature.
        layers: Which layers to capture (None = all, [-1] = last).
        step_delimiter: How to split CoT into reasoning steps.

    Returns:
        (trace, hidden_states) where hidden_states is::

            {
                "pooled_steps": ndarray (n_steps, n_layers, d_model) float16,
                "token_level": ndarray (n_tokens, n_layers, d_model) float16,
            }
    """
    formatted = build_prompt(tokenizer, prompt, system_prompt)

    gen = generate_with_hidden_states(
        model, tokenizer, formatted,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        layers=layers,
    )

    step_defs = split_into_steps(gen["text"], tokenizer, step_delimiter)
    if not step_defs:
        step_defs = [{
            "text": gen["text"],
            "token_start": 0,
            "token_end": gen["n_new_tokens"],
        }]

    pooled = pool_hidden_states_per_step(gen["token_hidden_states"], step_defs)

    trace = build_reasoning_trace(
        text=gen["text"],
        gen_metadata=gen,
        model_name=model_name,
        model_path=model_path,
        task=task,
        step_defs=step_defs,
        generation_config={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    )

    hidden_states = {
        "pooled_steps": pooled.astype(np.float16),
        "token_level": gen["token_hidden_states"].astype(np.float16),
    }

    return trace, hidden_states
```

**Step 4: Run test to verify it passes**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_inference_trace.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add manyagents/inference.py tests/test_inference_trace.py
git commit -m "feat: add extract_trace() to inference.py"
```

---

### Task 3: Add model cache to `inference.py`

**Files:**
- Modify: `manyagents/inference.py` (add after imports, before `resolve_model_path`)
- Modify: `tests/test_inference_trace.py` (add cache tests)

**Step 1: Write the failing test**

Append to `tests/test_inference_trace.py`:

```python
from manyagents.inference import get_model, clear_model_cache, _model_cache


def test_model_cache_stores_and_retrieves(monkeypatch):
    """get_model caches by resolved path; second call returns same objects."""
    call_count = 0
    sentinel_model = object()
    sentinel_tok = object()
    sentinel_hf = object()

    def _fake_load(path, **kwargs):
        nonlocal call_count
        call_count += 1
        return sentinel_model, sentinel_tok, sentinel_hf

    monkeypatch.setattr("manyagents.inference.load_model", _fake_load)
    monkeypatch.setattr("manyagents.inference.resolve_model_path", lambda m: f"/fake/{m}")
    clear_model_cache()

    m1, t1, h1 = get_model("test-model")
    m2, t2, h2 = get_model("test-model")

    assert m1 is sentinel_model
    assert m1 is m2
    assert call_count == 1  # load_model called only once

    clear_model_cache()
    assert len(_model_cache) == 0


def test_clear_model_cache_empties():
    """clear_model_cache empties the cache dict."""
    from manyagents.inference import _model_cache
    _model_cache["dummy"] = ("a", "b", "c")
    clear_model_cache()
    assert "dummy" not in _model_cache
```

**Step 2: Run test to verify it fails**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_inference_trace.py::test_model_cache_stores_and_retrieves -v`
Expected: FAIL with `ImportError: cannot import name 'get_model'`

**Step 3: Write minimal implementation**

Add to `manyagents/inference.py` after the imports section (after line 20), before `DEFAULT_MODEL_PATHS`:

```python
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
        log.info(f"Cache miss for '{model}' → loading from {model_path}")
        _model_cache[model_path] = load_model(model_path, dtype=dtype, device_map=device_map)
    else:
        log.debug(f"Cache hit for '{model}' ({model_path})")
    return _model_cache[model_path]


def clear_model_cache():
    """Free GPU memory by clearing the model cache."""
    _model_cache.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
```

**Step 4: Run test to verify it passes**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_inference_trace.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add manyagents/inference.py tests/test_inference_trace.py
git commit -m "feat: add module-level model cache to inference.py"
```

---

### Task 4: Rename `LocalLLMAdapter` to `HFAdapter`

**Files:**
- Create: `manyagents/adapters/hf_adapter.py` (copy + modify from `local_llm_adapter.py`)
- Modify: `manyagents/adapters/__init__.py`
- Modify: `manyagents/adapters/local_llm_adapter.py` (reduce to re-export)
- Test: `tests/test_hf_adapter.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_hf_adapter.py
"""Tests for HFAdapter (renamed from LocalLLMAdapter)."""

from manyagents.adapters import ADAPTER_REGISTRY
from manyagents.adapters.hf_adapter import HFAdapter


def test_hf_adapter_in_registry():
    """HFAdapter is registered under both 'hf' and 'local_llm' keys."""
    assert "hf" in ADAPTER_REGISTRY
    assert "local_llm" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["hf"] is HFAdapter
    assert ADAPTER_REGISTRY["local_llm"] is HFAdapter


def test_hf_adapter_importable_from_old_path():
    """Backward compat: LocalLLMAdapter still importable from old module."""
    from manyagents.adapters.local_llm_adapter import LocalLLMAdapter
    assert LocalLLMAdapter is HFAdapter
```

**Step 2: Run test to verify it fails**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_hf_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manyagents.adapters.hf_adapter'`

**Step 3: Create `hf_adapter.py`**

Create `manyagents/adapters/hf_adapter.py` — copy `local_llm_adapter.py` with these changes:

1. Class renamed: `LocalLLMAdapter` → `HFAdapter`
2. Docstring updated: mentions HF Hub IDs
3. `super().__init__("hf")` instead of `"local_llm"`
4. Replace `_build_and_save_trace()` with delegation to `inference.build_reasoning_trace()`
5. When `build_trace=True`: call `inference.extract_trace()` instead of manual pipeline
6. Use `inference.get_model()` for cached loading instead of own `_load_model()`

```python
"""HuggingFace adapter for running models via transformers.

Supports both local weight paths and HuggingFace Hub IDs.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import AgentAdapter, AdapterResult
from manyagents.utils.helpers import truncate_string
from manyagents import inference

log = logging.getLogger(__name__)


class HFAdapter(AgentAdapter):
    """Adapter for HuggingFace transformers inference.

    Supports loading models from local paths (e.g., /network/weights/)
    or HuggingFace Hub IDs (e.g., Qwen/Qwen3-4B).
    """

    DEFAULT_MODEL = "llama-3.1-8b"
    DEFAULT_MAX_NEW_TOKENS = 2000
    DEFAULT_TEMPERATURE = 0.1

    def __init__(self, model_name: Optional[str] = None, device_map: str = "auto"):
        super().__init__("hf")
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device_map = device_map

    async def run(
        self,
        task_config: Dict[str, Any],
        input_files: Dict[str, Path],
        input_data: Optional[Any] = None,
    ) -> AdapterResult:
        log.info(f"HFAdapter executing with config: {task_config}")

        if "prompt" not in task_config:
            return self.error_response(
                "HFAdapter requires 'prompt' parameter in task_config",
                error_type="missing_prompt",
            )

        model_name = task_config.get("model", self.model_name)
        max_new_tokens = task_config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)
        temperature = task_config.get("temperature", self.DEFAULT_TEMPERATURE)
        system_prompt = task_config.get("system_prompt")
        capture_hidden_states = task_config.get("capture_hidden_states", False)
        build_trace = task_config.get("build_trace", False)
        layers = task_config.get("layers")
        step_delimiter = task_config.get("step_delimiter", "\n")

        try:
            model_path = inference.resolve_model_path(model_name)

            def _run_inference():
                model, tokenizer, _ = inference.get_model(
                    model_name, device_map=self.device_map,
                )

                if build_trace or capture_hidden_states:
                    from manyagents.schemas.reasoning import TaskInfo

                    task = TaskInfo(
                        dataset=task_config.get("dataset", "unknown"),
                        task_id=task_config.get("task_id", uuid.uuid4().hex[:8]),
                        prompt=task_config["prompt"],
                        expected_answer=task_config.get("expected_answer"),
                        domain=task_config.get("domain"),
                        logic_type=task_config.get("logic_type"),
                    )
                    trace, hidden_states = inference.extract_trace(
                        model, tokenizer,
                        prompt=task_config["prompt"],
                        task=task,
                        model_name=model_name,
                        model_path=model_path,
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        layers=layers,
                        step_delimiter=step_delimiter,
                    )
                    return {
                        "text": trace.response_text,
                        "trace": trace,
                        "hidden_states": hidden_states,
                    }
                else:
                    formatted = inference.build_prompt(
                        tokenizer, task_config["prompt"], system_prompt,
                    )
                    text = inference.generate(
                        model, tokenizer, formatted,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    return {"text": text, "trace": None, "hidden_states": None}

            start_time = time.time()
            result = await asyncio.to_thread(_run_inference)
            response_time = time.time() - start_time

            content = result["text"]
            log.info(f"Inference completed in {response_time:.2f}s")

            unique_id = uuid.uuid4().hex[:8]
            output_files: Dict[str, Any] = {
                "raw_response": self.save_text_output(content, f"response_{unique_id}.txt"),
            }

            if result["hidden_states"] is not None:
                npz_path = self.output_dir / f"hidden_states_{unique_id}.npz"
                np.savez_compressed(npz_path, **result["hidden_states"])
                output_files["hidden_states"] = npz_path

            if result["trace"] is not None:
                trace_path = self.output_dir / f"trace_{unique_id}.json"
                trace_path.write_text(result["trace"].to_json())
                output_files["trace"] = trace_path

            return self.success_response(
                summary=f"HF inference completed. Response: {truncate_string(content, 100)}",
                output_files=output_files,
                metadata={
                    "model": model_name,
                    "model_path": model_path,
                    "response_time": response_time,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "captured_hidden_states": result["hidden_states"] is not None,
                    "built_trace": result["trace"] is not None,
                },
            )

        except ImportError as e:
            return self.error_response(
                f"Missing dependency: {e}. Run 'uv add transformers accelerate'",
                error_type="import_error",
                details=str(e),
            )

        except Exception as e:
            log.error(f"HF inference failed: {e}", exc_info=True)
            return self.error_response(
                f"HF inference failed: {e}",
                error_type="inference_error",
                details=str(e),
            )

    def unload_model(self):
        """Explicitly free GPU memory via the shared cache."""
        inference.clear_model_cache()
        log.info("Model cache cleared and GPU memory freed")
```

**Step 4: Update `local_llm_adapter.py` to re-export**

Replace entire content of `manyagents/adapters/local_llm_adapter.py`:

```python
"""Backward compatibility — LocalLLMAdapter is now HFAdapter."""
from .hf_adapter import HFAdapter as LocalLLMAdapter

__all__ = ["LocalLLMAdapter"]
```

**Step 5: Update `__init__.py` registry**

Replace `manyagents/adapters/__init__.py`:

```python
"""
ManyAgents adapter registry.

Provides a centralized registry of all available adapters.
"""

from .base import AgentAdapter
from .mock_adapter import MockAdapter
from .claude_adapter import ClaudeAdapter
from .openai_adapter import OpenAIAdapter
from .hf_adapter import HFAdapter
from .cellforge_adapter import CellForgeAdapter
from .kosmos_adapter import KosmosAdapter
from .placeholder_adapter import PlaceholderAdapter

# Core adapters (always available)
ADAPTER_REGISTRY = {
    "mock": MockAdapter,
    "claude": ClaudeAdapter,
    "openai": OpenAIAdapter,
    "hf": HFAdapter,
    "local_llm": HFAdapter,  # backward compat alias
    "cellforge": CellForgeAdapter,
    "kosmos": KosmosAdapter,
}

# Optional adapters (require additional dependencies)
try:
    from .manylatents_adapter import ManyLatentsAdapter
    ADAPTER_REGISTRY["manylatents"] = ManyLatentsAdapter
except ImportError:
    ManyLatentsAdapter = None

try:
    from .biomni_adapter import BiomniAdapter
    ADAPTER_REGISTRY["biomni"] = BiomniAdapter
except ImportError:
    BiomniAdapter = None  # biomni package not installed

# Backwards compatibility aliases
BioDiscoveryAgentAdapter = PlaceholderAdapter
LocalLLMAdapter = HFAdapter

__all__ = [
    "AgentAdapter",
    "ADAPTER_REGISTRY",
    "MockAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "HFAdapter",
    "LocalLLMAdapter",
    "CellForgeAdapter",
    "KosmosAdapter",
    "PlaceholderAdapter",
    "BioDiscoveryAgentAdapter",
]
```

**Step 6: Run tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_hf_adapter.py tests/test_inference_trace.py -v`
Expected: All pass

**Step 7: Run existing tests to check nothing broke**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/ -v`
Expected: All existing tests still pass

**Step 8: Commit**

```bash
git add manyagents/adapters/hf_adapter.py manyagents/adapters/local_llm_adapter.py manyagents/adapters/__init__.py tests/test_hf_adapter.py
git commit -m "refactor: rename LocalLLMAdapter to HFAdapter, delegate to inference.extract_trace()"
```

---

### Task 5: Wire `ClaudeAdapter` for trace building

**Files:**
- Modify: `manyagents/adapters/claude_adapter.py`
- Modify: `tests/test_hf_adapter.py` → rename to `tests/test_adapters_trace.py` (or append)

**Step 1: Write the failing test**

Create `tests/test_claude_trace.py`:

```python
# tests/test_claude_trace.py
"""Tests for ClaudeAdapter trace building."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from manyagents.adapters.claude_adapter import ClaudeAdapter
from manyagents.schemas.reasoning import ModelBackend, StepKind


@pytest.fixture
def adapter():
    a = ClaudeAdapter()
    a.api_key = "test-key"
    return a


@pytest.mark.asyncio
async def test_claude_adapter_build_trace(adapter):
    """ClaudeAdapter builds ReasoningTrace when build_trace=True."""
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(type="thinking", thinking="Let me think..."),
        MagicMock(type="text", text="The answer is 4."),
    ]
    mock_response.usage = MagicMock(input_tokens=50, output_tokens=20)
    mock_response.stop_reason = "end_turn"

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    adapter.client = mock_client

    result = await adapter.run(
        task_config={
            "prompt": "What is 2+2?",
            "build_trace": True,
            "dataset": "gsm8k",
            "task_id": "test_001",
        },
        input_files={},
    )

    assert result["success"]
    assert "trace" in result.get("output_files", {})
    # Verify the trace file was written
    trace_path = result["output_files"]["trace"]
    from manyagents.schemas.reasoning import ReasoningTrace
    trace = ReasoningTrace.from_json(trace_path.read_text())
    assert trace.model.backend == ModelBackend.ANTHROPIC
    assert len(trace.steps) == 2
    assert trace.steps[0].kind == StepKind.THINKING
    assert trace.steps[1].kind == StepKind.OUTPUT
    assert trace.input_tokens == 50
    assert trace.has_tensors is False
```

**Step 2: Run test to verify it fails**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_claude_trace.py -v`
Expected: FAIL (no trace in output_files)

**Step 3: Modify `claude_adapter.py`**

Add trace building after the API response is received. In `claude_adapter.py`, after line 112 (`output_files = {...}`), add:

```python
        # Optionally build a ReasoningTrace
        if task_config.get("build_trace", False):
            from manyagents.schemas.reasoning import TaskInfo, trace_from_anthropic

            task = TaskInfo(
                dataset=task_config.get("dataset", "unknown"),
                task_id=task_config.get("task_id", f"claude_{int(time.time())}"),
                prompt=task_config[PROMPT],
                expected_answer=task_config.get("expected_answer"),
                domain=task_config.get("domain"),
                logic_type=task_config.get("logic_type"),
            )
            trace = trace_from_anthropic(
                response, task,
                model_name=api_params["model"],
                duration_ms=int(response_time * 1000),
            )
            trace_path = self.output_dir / "trace.json"
            trace_path.write_text(trace.to_json())
            output_files["trace"] = trace_path
```

**Step 4: Run test to verify it passes**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/test_claude_trace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add manyagents/adapters/claude_adapter.py tests/test_claude_trace.py
git commit -m "feat: wire ClaudeAdapter to build ReasoningTrace via trace_from_anthropic()"
```

---

### Task 6: Add Hydra experiment config for trace extraction

**Files:**
- Create: `manyagents/configs/agent/hf.yaml`
- Create: `manyagents/configs/experiment/trace_extraction.yaml`
- Modify: `manyagents/experiment.py` (add trace extraction code path)
- Test: manual Hydra dry-run

**Step 1: Create `configs/agent/hf.yaml`**

```yaml
# @package _global_
# HuggingFace model agent configuration

agent:
  name: hf
  adapter: hf
  config:
    model: llama-3.1-8b
    temperature: 0.1
    max_new_tokens: 2000
  available_models:
    llama-3.3-70b: /network/weights/llama.var/llama_3.3/Llama-3.3-70B-Instruct
    llama-3.1-70b: /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-70B-Instruct
    llama-3.1-8b: /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct
    llama-3.1-405b-fp8: /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-405B-Instruct-FP8
    olmo-7b: /network/weights/olmo/OLMo-7B-Twin-2T
    olmoe-1b-7b: /network/weights/olmoe/OLMoE-1B-7B-0924
```

**Step 2: Create `configs/experiment/trace_extraction.yaml`**

```yaml
# @package _global_
# Trace extraction experiment — generates ReasoningTraces with hidden states

defaults:
  - /agent: hf

name: trace_extraction

# Trace-specific config
trace_extraction:
  enabled: true
  dataset: gsm8k
  n_samples: 10
  system_prompt: "Solve the problem step by step. Show your reasoning clearly, with each step on a new line."

# Override agent config for trace extraction
agent:
  config:
    build_trace: true
    capture_hidden_states: true
    layers: [-1]
    step_delimiter: "\n"
    temperature: 0.7
    max_new_tokens: 512
```

**Step 3: Add trace extraction path to `experiment.py`**

Add a new function and modify `run_experiment` to detect the trace extraction config. Insert before `run_experiment()`:

```python
async def _run_trace_extraction(cfg: DictConfig) -> Dict[str, Any]:
    """Run trace extraction experiment — generates ReasoningTraces with hidden states.

    This is the Hydra-driven equivalent of the old scripts/extract_traces.py.
    """
    from manyagents.adapters import ADAPTER_REGISTRY
    from manyagents.schemas.reasoning import TaskInfo, TraceStore

    trace_cfg = cfg.trace_extraction
    agent_config = cfg.agent if hasattr(cfg, "agent") else _get_agent_config(cfg, cfg.active_agents[0])

    # Load dataset
    dataset_name = trace_cfg.get("dataset", "gsm8k")
    n_samples = trace_cfg.get("n_samples", 10)
    system_prompt = trace_cfg.get("system_prompt", "Solve the problem step by step.")

    tasks = _load_dataset_tasks(dataset_name, n_samples)
    log.info(f"Loaded {len(tasks)} tasks from {dataset_name}")

    # Create adapter
    adapter_name = agent_config.adapter if hasattr(agent_config, "adapter") else "hf"
    adapter = ADAPTER_REGISTRY[adapter_name]()

    output_dir = Path(cfg.output_dir)
    store_dir = output_dir / "traces"

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
                result = await adapter.run(task_config, {})
                if result["success"] and "trace" in result.get("output_files", {}):
                    trace_path = result["output_files"]["trace"]
                    from manyagents.schemas.reasoning import ReasoningTrace
                    trace = ReasoningTrace.from_json(Path(trace_path).read_text())

                    # Load hidden states if saved
                    hs = None
                    if "hidden_states" in result.get("output_files", {}):
                        hs_path = result["output_files"]["hidden_states"]
                        hs = dict(np.load(hs_path, allow_pickle=False))

                    store.append(trace, hidden_states=hs)
                    log.info(f"  -> {len(trace.steps)} steps, {trace.output_tokens} tokens")
                else:
                    log.warning(f"  SKIPPED: {result.get('summary', 'unknown error')}")
            except Exception as e:
                log.error(f"  FAILED: {e}", exc_info=True)

    store_r = TraceStore(store_dir, mode="r")
    summary = store_r.summary()
    log.info(f"Trace extraction complete: {json.dumps(summary, indent=2)}")

    return {"experiment_id": f"trace_{dataset_name}", "summary": summary, "output_dir": str(store_dir)}


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
```

Then modify `run_experiment()` to detect trace extraction mode. At the top of `run_experiment()`, add:

```python
    # Dispatch to trace extraction if configured
    if hasattr(cfg, "trace_extraction") and getattr(cfg.trace_extraction, "enabled", False):
        return await _run_trace_extraction(cfg)
```

**Step 4: Delete `scripts/extract_traces.py`**

```bash
git rm scripts/extract_traces.py
```

**Step 5: Run the existing test suite**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/ -v`
Expected: All pass

**Step 6: Commit**

```bash
git add manyagents/configs/agent/hf.yaml manyagents/configs/experiment/trace_extraction.yaml manyagents/experiment.py
git commit -m "feat: trace extraction as Hydra experiment, delete standalone script"
```

---

### Task 7: Update `configs/agent/local_llm.yaml` to alias `hf.yaml`

**Files:**
- Modify: `manyagents/configs/agent/local_llm.yaml`

**Step 1: Update the config**

Replace content of `local_llm.yaml`:

```yaml
# @package _global_
# Backward compatibility alias — use agent=hf instead
# This file is kept for existing configs that reference agent=local_llm

defaults:
  - hf

# Override adapter name for backward compat with experiment logs
agent:
  adapter: hf
```

Also update `local_llm_70b.yaml` if it exists — check and update similarly.

**Step 2: Commit**

```bash
git add manyagents/configs/agent/local_llm.yaml
git commit -m "refactor: local_llm.yaml now aliases hf.yaml"
```

---

### Task 8: Final verification — run full test suite

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/agents && uv run pytest tests/ -v`
Expected: All pass

**Step 2: Verify imports work**

```bash
cd /network/scratch/c/cesar.valdez/lrw/agents && uv run python -c "
from manyagents.inference import extract_trace, build_reasoning_trace, get_model, clear_model_cache
from manyagents.adapters import ADAPTER_REGISTRY, HFAdapter, LocalLLMAdapter
from manyagents.adapters.hf_adapter import HFAdapter as HF
from manyagents.adapters.local_llm_adapter import LocalLLMAdapter as LLMA
assert ADAPTER_REGISTRY['hf'] is HFAdapter
assert ADAPTER_REGISTRY['local_llm'] is HFAdapter
assert HF is HFAdapter
assert LLMA is HFAdapter
print('All imports and registry checks passed')
"
```

**Step 3: Verify Hydra config loads**

```bash
cd /network/scratch/c/cesar.valdez/lrw/agents && uv run python -c "
from hydra import compose, initialize_config_dir
import os
config_dir = os.path.join(os.getcwd(), 'manyagents', 'configs')
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name='main', overrides=['experiment=trace_extraction'])
    print(f'Config loaded: {cfg.name}')
    print(f'Trace extraction enabled: {cfg.trace_extraction.enabled}')
    print(f'Agent adapter: {cfg.agent.adapter}')
    print(f'Dataset: {cfg.trace_extraction.dataset}')
"
```
