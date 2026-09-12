# Changelog

## 0.1.1

**0.1.1 is not a drop-in upgrade from 0.1.0.** The version remains 0.1.1 deliberately; the patch version does not imply API, result-schema, or experimental comparability. Review these migrations before upgrading custom adapters, readers, and stored trajectories.

### Adapter results and custom adapters

Every `AdapterResult` must contain `success: bool`, `summary: str`, and `output_files: dict`, including failed results. `metadata: dict` and `embeddings: dict` remain optional. Boundary validation rejects missing or incorrectly typed required fields. Import `AdapterResult` and `AgentAdapter` from `manyagents.adapters.base`.

Migrate custom adapters to the base helpers (inside an async `run` implementation):

```python
# Before: incomplete result, implicit success, or a bare response string
return {"summary": response}

# After: explicit outcome and a saved response for text evaluation
return self.success_response(
    summary="Response generated",
    output_files=self.save_response(response),
    metadata={"model": model_id},
)
# On an execution failure:
# return self.error_response("Generation failed", details=str(error))
```

A text adapter must provide nonempty response content under `output_files["raw_response"]`. Built-in text adapters return a `pathlib.Path`; the evaluator also accepts an inline string at that key. It does not turn a summary into a response. Artifact/compute adapters should declare `PRODUCES_TEXT_RESPONSE = False`; the text-evaluation runner rejects them before dispatch. Unknown adapter names and malformed results are failures rather than implicit successes.

### Mock response reader migration

Mock no longer returns inline text at `output_files["response"]`. It writes the response and returns a **`Path`** at `output_files["raw_response"]`:

```python
# 0.1.0 reader
text = result["output_files"]["response"]  # inline str

# 0.1.1 reader (also works for the other built-in text adapters)
if not result["success"]:
    raise RuntimeError(result["summary"])
response_path = result["output_files"]["raw_response"]  # pathlib.Path
text = response_path.read_text()
```

The old key is not an alias. Adapter response files can be overwritten by later calls; evaluation copies response text into `results.json`, which is the durable experiment record. Consume or copy individual response files before reusing an adapter.

### Exit status, partial failures, and missing measurements

Bare CLI invocation without an experiment, empty or unknown `active_agents`, unknown adapters, and artifact adapters used for text evaluation exit nonzero. Evaluation with zero successful prompt/agent pairs saves its results and then exits nonzero. A run with at least one successful evaluation may return normally even when other evaluations fail: inspect per-prompt `success`/`error` and each system's `prompts_evaluated`/`prompts_failed`. Metrics use successful evaluations, not failed responses padded with zeros.

Trace extraction similarly exits nonzero when it persists zero traces. Partial trace failures are counted in `traces_failed`; counts include only traces persisted by this run. Requested hidden states must be present, nonempty, finite floating arrays. This release has no answer judge: generated traces are `unjudged` (`success=None`, `judge="none"`); summary `success`/`failure` counts remain zero unless supplied by an external judge. Hidden-state traces with fewer than two steps are rejected before store persistence; the shipped default now uses newline segmentation.

`manyagents.api.run()` preserves the caller's Hydra composition context, including its config search path, job name, and compatibility version. It retains the CLI failure semantics: **`SystemExit` propagates through the Python API** (it is not caught by `except Exception`). Catch it explicitly if embedding the runner:

```python
from manyagents.api import run

try:
    results = run(["experiment=test_wandb"])
except SystemExit as exc:
    print(f"Experiment did not complete successfully: {exc.code}")
```

Unavailable aggregate metrics are `None` in Python, `null` in JSON, and `n/a` in summaries. Jaccard needs at least two successful prompts. Missing or empty `ground_truth_methods` makes the per-prompt ground-truth match and match ratio unavailable. If any successful prompt lacks that criterion, the system's entire `ground_truth_match_rate` is unavailable; the denominator is not silently narrowed. `prompts_failed` still counts execution failures, so missing criteria alone do not increase it.

Missing scoring or validation functions now raise `ValueError`. A validator returning no `_flag` columns also raises, in both `strict` and `flag` modes. For explicit simulations only, select `manyagents.utils.scoring:random_scores` and optionally pass `scoring_params={"seed": 7}`. Its random `score` values carry `score_simulated=True` on every output row, including after filtering/ranking. Structured manylatents metric dictionaries without a scalar, or with an unavailable status, display `n/a`; their container length is never a measurement.

Tag segmentation tracks successive occurrences of repeated sentences. Pooling rejects empty, reversed, negative, and out-of-range token intervals with `ValueError`; it no longer clips or widens them to fabricate a vector.

### Python and installation extras

Supported Python versions are **3.11 and 3.12** (`>=3.11,<3.13`). Core includes API clients, mock, and local Hugging Face generation; `accelerate` brings in torch, so core is still a large installation.

| Extra | Adds |
| --- | --- |
| `traces` | manylatents `>=0.1.6,<0.2`, datasets, and SciPy for hidden-state hooks and geometry |
| `wandb` | Optional W&B experiment logging |
| `vllm` | vLLM generation backend; GPU/backend platform requirements still apply |
| `full` | `traces`, `wandb`, and Biomni |
| `dev` | pytest, pytest-asyncio, ruff, and pre-commit |

**`full` excludes `vllm` and `dev`**, and does not install external CellForge/Kosmos executables or the Shop cluster launcher. Source checkouts sync a separate default development group (pytest, pytest-asyncio, ruff), which is not the `dev` extra. Plain vLLM generation needs `vllm`; vLLM hidden-state replay needs both `vllm` and `traces`. There is no `docs` extra. With uv, include all needed extras in the same sync command and use `uv run --no-sync` to retain them.

The metric registry is generated in memory at use time. Wheels no longer use a registry build hook or require manylatents to build. `manyagents-generate-registry` prints JSON; use `--output /writable/path/registry.json` for an explicit cache. Package metadata now declares the existing MIT license.

### Model defaults and comparable experiments

Claude and Biomni now default to `claude-opus-5`. Previously their YAML defaults named `claude-sonnet-4-20250514`; Claude's direct adapter fallback separately named `claude-opus-4-8`. OpenAI retains `gpt-4o`. HF and vLLM shipped configs select `Qwen/Qwen3-0.6B`, subject to experiment and cluster overrides. These describe this release's configuration, not service availability guarantees.

Do not compare runs across upgrades using implicit defaults. Pin the exact provider model ID (or local model/tokenizer snapshot) and retain the resolved config, dependency versions, backend, prompts, seeds, and generation settings. For example, to retain the earlier YAML choice where that provider ID remains available:

```bash
manyagents experiment=invariance_full 'active_agents=[claude,biomni]' agents.claude.agent.config.model=claude-sonnet-4-20250514 agents.biomni.agent.config.llm=claude-sonnet-4-20250514
```

For the single-agent trace path, use `agent.config.model=<model-or-snapshot>`; direct Python adapter calls use `task_config["model"]` for Claude/HF/vLLM/OpenAI and `task_config["llm"]` for Biomni. Pin temperature, token limits, repetition penalty, segmentation, captured layers, and dtypes too. The vLLM engine defaults to `bfloat16`; pre-Ampere GPUs require an explicit supported dtype such as `agent.config.dtype=float16`. Hidden-state storage dtype is a separate setting: direct `inference.extract_trace(state_dtype="float32")` preserves a wider range, while the Hydra adapter path currently stores float16. Changing dtypes or backends can change trajectory geometry even with the same model.

### GVector outcomes, arithmetic, and legacy data

`GVector.measurements` records named outcomes: `measured` with a finite `value`, `failed` with a `reason`, `not_requested`, or `unknown` for legacy provenance. The four numeric fields remain for schema compatibility. Their zeros for unavailable metrics are **padding only**, never measured zeros. A measured zero explicitly has `{"status": "measured", "value": 0.0}`.

Measurement metadata supplies the values used by `metric_value()`, `to_array()`, and trajectory `deltas`. A measured core field must equal its metadata value. Construction/deserialization and subsequent measurement reads or serialization validate this consistency, including nested dictionary edits, and raise `ValueError` on disagreement. When changing a measurement, update the field and outcome together before reading it, or construct a new vector.

`metric_value(name)` raises for failed, unrequested, unknown, or invalid outcomes. `to_array()` and trajectory `deltas` require all four core metrics and reject any requested failure, including additional metrics. Incomplete records still serialize for inspection; they cannot enter arithmetic as padded vectors.

**Legacy policy:** `from_dict()`/`from_json()` documents with missing or null `measurements` load every core outcome as `unknown`. Numeric fields remain readable and unknown status survives another round trip. They are not silently certified, regardless of whether their numbers are zero or nonzero. This deliberately replaces the old test contract that auto-certified legacy zeros.

Recompute legacy measurements when possible. If independent records establish a field's provenance, explicitly supply its verified outcome; keep unverifiable fields unknown. For example:

```python
from manyagents.schemas import GVector

record = {"beta_0": 0, "beta_1": 0, "participation_ratio": 0.0, "local_intrinsic_dim": 0.0}
g = GVector.from_dict(record)  # readable, unknown; g.to_array() raises
verified = g.to_dict()
# Only after independently verifying that beta_0 was actually measured as zero:
verified["measurements"]["beta_0"] = {"status": "measured", "value": 0}
g = GVector.from_dict(verified)  # beta_0 is readable via metric_value; other fields stay unknown
```

Direct `GVector(...)` construction and `from_array(...)` without metadata remain explicit caller assertions that the supplied numeric values are measured. Do not use those constructors to bypass provenance review when migrating old padded records.
