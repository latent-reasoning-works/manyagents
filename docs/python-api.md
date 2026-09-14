# Python API

Using manyagents from Python rather than the CLI. See the [README](../README.md) for
installation and the command-line quickstart.

## Adapters

11 classes behind 12 registry keys. Registration does not mean the dependency is
installed — a missing one surfaces as a failed `AdapterResult`, not an import error.

| key | class | runs |
|---|---|---|
| `claude` | `ClaudeAdapter` | Anthropic API (`claude-opus-5` default); text traces |
| `openai` | `OpenAIAdapter` | OpenAI-compatible Chat Completions (`gpt-4o` default) |
| `ollama` | `OllamaAdapter` | `OpenAIAdapter` pointed at a local Ollama server; no hidden states |
| `hf`, `local_llm` | `HFAdapter` | local transformers generation; hidden states |
| `vllm` | `VLLMAdapter` | vLLM generation; HF replay for hidden states (`vllm` + `traces`) |
| `manylatents` | `ManyLatentsAdapter` | in-process DR and geometric metrics (`traces`) |
| `biomni` | `BiomniAdapter` | in-process biomedical agent (`full`, Anthropic key) |
| `cellforge`, `kosmos` | `CellForgeAdapter`, `KosmosAdapter` | external local CLI installations |
| `mock` | `MockAdapter` | deterministic responses; no external service |
| `placeholder` | `PlaceholderAdapter` | development stub |

Built-in text adapters return a `pathlib.Path` at `raw_response`; the evaluator also accepts an inline string there. Compute adapters set `PRODUCES_TEXT_RESPONSE = False` and are rejected by the text-evaluation runner before dispatch. Response files can be overwritten by later calls; `results.json` is the durable record.

## Tool-calling loop

`manyagents.agent_loop.run_agent_loop(prompt, agent=..., tools=[...])` drives any adapter exposing `chat()` (OpenAI, Ollama, Claude) until it stops calling tools or hits `max_steps`. A `manyagents.tools.Tool` pairs a JSON Schema with a trusted sync or async callable; the returned `AgentResult` carries `answer`, the full `messages` transcript (pass it back as `history` to continue), `steps`, `stopped`, and the executed `tool_calls`. Tool bodies run with the caller's permissions.

## DR workflows

`manyagents.workflows.sequence.execute_sequence(workflow, dataset)` chains manylatents algorithms and records a `GVector` (β₀, β₁, participation ratio, local intrinsic dimension) after every step as a `TransformationTrajectory`. Needs `traces`; unavailable measurements have named outcomes, while fixed GVector numeric fields contain zero padding. Read through `gvector.metric_value(name)` to check validity; it raises for failed, unrequested, or unknown measurements.
