# Release-readiness pass, 2026-09-12

Base: `67b93ec`, branch `fix/release-gaps`. Fable's scratchpad `final-gaps.md`
was read before changing files. Work and tests used an archive of the committed
source in `/private/tmp/manyagents-release-gaps-20260912`, with a newly created
venv. The developer checkout's `.venv` was never used. No push, tag, merge, model
API call, or W&B run was performed.

## Fixes and regression evidence

| Finding | Result | Regression and failed-first evidence |
| --- | --- | --- |
| M1: shipped defaults | All seven experiments load `hf` into the existing `local_llm` name. README, CLAUDE, and both running/config guides no longer require the package override. The legacy `local_llm.yaml` stays, using `hf@_here_` and `_self_` so nested composition preserves HF config. | `test_shipped_default_invocation_completes`: all seven cases failed before fixes. It invokes the real Hydra entry point, mocks adapters only, leaves experiment/agent/model/W&B defaults intact, and verifies every job's persisted responses and metrics. The two default sweeps execute all 36 and 4 jobs. Existing README/CLAUDE sweep execution tests now run without the workaround. Named alias propagation and the repackaged legacy-alias case also failed first. |
| M1: portable sweep model | `llm_reasoning_sweep` defaults to `Qwen/Qwen3-0.6B` rather than three Mila-only aliases. Users can supply other models with CLI overrides. | The same default-invocation regression checks HF config and resolves the model name offline, preventing a mock from concealing invalid local aliases. |
| M2: W&B defaults | All three offending configs default off; `wandb.enabled=true` remains the opt-in. | All three `test_shipped_wandb_is_opt_in` cases failed before fixes; they also assert the explicit opt-in composes as true. W&B is stubbed during execution regressions, so no run can be created even on the broken baseline. |
| M3: supplied data | `validate_manylatents_config(..., input_data=...)` accepts a supplied array as the data source; the adapter forwards it during validation. No dummy dataset is needed. | Validator regression failed first with an unexpected-keyword error; real adapter/PCA regression failed first with the missing-data diagnostic. The passing integration compares pairwise distances with PCA of the supplied array, verifying its contents are used. |
| M3: trace-to-geometry bridge | README now contains the runnable capture → TraceStore → NPZ → layer selection → float32 → manyLatents bridge, linked from the running guide. Per-trace primitives and grouped `compute_metric` are explicit. The cached adapter registry's trajectory limitations are stated plainly. | Both README-snippet execution cases failed first because the section was absent. Passing tests execute its actual code with real manylatents 0.1.7, multiple traces/layers, unequal trace lengths, a text-only record, and an insufficient-steps store. Expected grouped metrics average per-trace means and exclude inter-trace boundaries. |
| M3: unusable short captures | The experiment rejects zero-/one-step traces with tensors before appending to TraceStore, records extraction failures, and exits nonzero if none persist. Default capture segmentation is newline-based. Two-step captures still permit velocity. | Three short-trace rejection cases failed first because the runner persisted them. Tests verify nonzero exit and absence of new JSONL/NPZ artifacts; the two-step case verifies persistence and unjudged accounting. Existing fixtures were given actual steps to meet the new contract. |
| M3: judging | README, running guide, and changelog state there is no answer judge: generated traces have `success=None`, `judge="none"`, and count as `unjudged`. | The accepted two-step capture asserts `unjudged=1`, `success=failure=0`; this assertion already passed before fixes. No judge was added. |
| M4: usage comments | Replaced `manyagents-experiment`, corrected `.agent.config.model` paths, used portable example model IDs, and quoted list overrides for shells such as zsh. | Documentation-only changes were inspected; no artificial failing test was added for comment wording. |
| S5: score interpretation | README explains why constant mock responses deliberately yield Jaccard 1.0 and clustering-for-all 100%. Console scores now state metric directions. | The console regression failed first with zero direction hints; it now verifies two “lower is better” labels and one “higher is better”, while preserving `n/a`. |

The initial targeted run recorded **19 failed / 55 passed**. After the config
fixes, the CLI/model subset passed **32 tests**; the geometry/accounting subset
passed **42 tests**. The console direction regression was separately observed
failing before its change. Documentation, docstrings, deletions of unused
examples, dependency metadata, and repository templates were reviewed and linted,
not given contrived behavioral tests.

## Public-release cleanup

- Added `CITATION.cff`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, bug/feature issue
  templates, and a PR template. Citation metadata uses the existing author and
  version; no DOI or release date was invented. Added project URLs and a CI badge.
- Added the promised local Ruff pre-commit hook and its usage instructions.
  Included `scripts/` in CI and documented lint commands; fixed its three unused
  import/assignment errors. The benchmark no longer supplies a dummy dataset.
- Explicitly declared the existing Ruff correctness checks (`E4`, `E7`, `E9`,
  `F`). Unlocked Ruff 0.16.7 enables a broader default policy than locked 0.13.1;
  declaring project policy prevents unrelated lint drift.
- Removed the unused packaged drug-discovery mock examples and
  `docs/adapters_vs_utils.md`, plus references to those examples and the missing
  templates directory. Removed the unfilled future-decision template.
- Reworded public package help and adapter comments to address downstream callers
  without private Geomancy/Shop guidance. Optional Shop launcher imports and
  explicitly site-specific configs remain because they implement real optional
  integration behavior.
- Updated metric-registry documentation to describe in-memory generation,
  explicit caches, and missing trajectory YAMLs. Corrected the vLLM demo's
  “unchanged tensors” claim and its required extras.
- Documented `api.run`, HF/vLLM `run`, registry accessors, trace serializers,
  tool methods, and remaining public functions. An AST scan finds no undocumented
  non-underscore functions in non-test package Python files.
- Documented the callable tool loop. Removed hydra-zen recommendations from package
  help and marked its unused builder as an unsupported prototype with a separate
  dependency requirement.

## Dependency decision and clean verification

The minimum and lock now use **manylatents 0.1.7**. The declared range is
`>=0.1.7,<0.2`: it matches the tested trace API and allows compatible 0.1 releases
without accepting a new minor API line. This is a compatibility boundary, not a
guarantee that all future patch releases will work. A daily/manual unlocked CI
workflow complements the existing locked core/traces jobs.

| Environment | Install | Result |
| --- | --- | --- |
| Archive work copy, Python 3.11.15 | Fresh venv; `uv pip install '.[traces,dev]'`, no lock resolution | **441 passed / 5 skipped**; wheel-build test included. manylatents 0.1.7, torch 2.14.0, transformers 5.17.0, datasets 5.0.1, Ruff 0.16.7. |
| Second `git archive` copy, Python 3.12.13 | `uv sync --locked --extra traces --extra dev` | **441 passed / 5 skipped**; wheel-build test included. manylatents 0.1.7, torch 2.13.0, transformers 5.10.2, datasets 5.0.0, Ruff 0.13.1. |

The built wheel was then installed into the locked archive's venv and exercised
from `/private/tmp`, outside the source tree. Its no-keys CLI quickstart persisted
2/2 successful mock responses with W&B disabled; all seven packaged experiment
configs composed with complete HF settings and portable model defaults.

Both environments use published manylatents distributions without `direct_url`
metadata, and neither has Shop installed. During the suite, the locked archive's editable
`manyagents` install referred to its own archive, not the developer tree. The
subsequent wheel probe imported `manyagents` from that venv's `site-packages`. Ruff passes on
`manyagents/ tests/ scripts/`; the pre-commit config validates and the hook passes
against all tracked files in the scratch Git checkout. Tests use offline model
adapters; these results do not claim a new GPU/vLLM or live provider evaluation.

## Deliberately deferred and clarifications

- **No answer judge and no cached-adapter trajectory registry redesign.** The
  requested documented direct manyLatents path is implemented and tested; an
  adapter redesign would need a separate layer/grouping/metric-discovery contract.
- **No automatic W&B authentication recovery.** Opt-in removes the unexpected
  calls. Users who explicitly enable W&B must install/authenticate it as documented.
- **No deletion of the three historical review-created W&B runs.** The report's
  imperative is review evidence, not an explicit request from the user to delete
  remote account data. No remote account state was changed.
- **No CODEOWNERS or dependency-bot configuration.** Ownership identities and bot
  maintenance policy need maintainer choices; they do not block runnable release
  paths. No broad classifier/keyword taxonomy or PyPI-relative-link rewrite was
  added; repository URLs now provide a canonical entry point.
- **No removal of generic scoring/validation/data utilities.** Their explicit
  extension contracts remain tested even though the obsolete mock example package
  was removed. Site-specific profiles and the unused `local_llm_70b` profile were
  not modernized; the seven shipped experiment defaults no longer depend on them.
- **No live model downloads or GPU validation.** Requested default-command
  regressions use offline adapters, while geometry/PCA runs use real manylatents.
  Existing GPU skips remain intentional.
- Fable's initial heading/count understated M1; seven shipped experiment configs
  carried the alias. `llm_reasoning_sweep` already had a partial `config.model`
  node, so its failure was not precisely the missing-`config` error seen in the
  flagship; inherited HF settings and portable model resolution were still broken.
- The report's ungrouped PCA example does execute, but its scores include
  concatenation boundaries. README distinguishes those from grouped measurements
  and shows how to group the returned embedding. The bridge also now handles
  text-only/short stores instead of failing on `None` or empty concatenation.
- A two-step trace has velocity but cannot have Menger curvature. The rejection
  threshold is two; the example requires three when calculating both metrics.

## Delivery

Implementation commits: `6ce4927` (defaults/W&B), `f5d4b74` (geometry),
`003d608` (public-release cleanup), `ce825b1` (dependency policy), and
`7fb386a` (shell-safe usage comments). This report is a separate final commit.

Commits are in the scratch checkout on `fix/release-gaps`, retaining ancestry from
`67b93ec`. A bundle accompanies the synchronized workspace files because the
workspace Git metadata is read-only. The workspace's HEAD therefore stays at the
base commit until the maintainer imports/applies the committed branch. No push,
tag, or merge was performed.
