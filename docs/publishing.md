# Publishing manyagents

Version **0.2.0** is the first PyPI release. Earlier versions were Git tags;
the release preparation changes packaging, metadata, and documentation only.

## What ships

Both Hatch targets use `only-include` in `pyproject.toml`:

- Wheel: `["manyagents"]`, excluding `manyagents/**/test_*.py`. Package data,
  including all Hydra YAML files, stays with the package.
- Sdist: `["manyagents", "tests", "README.md", "LICENSE", "CHANGELOG.md",
  "CITATION.cff", "pyproject.toml"]`.

Hatch also generates distribution metadata and includes the MIT license in the
wheel. Repository workflows, agent instructions, hooks, Python version file,
lockfile, `docs/`, and `scripts/` are development material and are not in the
sdist. Tests pin the include-lists and inspect built archive contents. Checks
that require the repository-only lockfile or agent instructions skip when run
from an extracted sdist.

The public-tree hygiene guard stores SHA-256 digests and hashes candidate tokens;
it reports counts without reproducing retired identifiers. The release scrub
changed **1 file**, removing **2 occurrences**; the subsequent sweep found **0**.

## Metadata and dependency review

The project and citation version remain 0.2.0; `manyagents.__version__` and the
lockfile agree. PyPI metadata identifies Latent Reasoning Works as the author,
uses the MIT license expression and license file, describes the scientific
workflow use case, and limits Python to `>=3.11,<3.13`. README links to repository
documents are absolute so they also work on PyPI.

Existing APIs require these corrected lower bounds:

| Dependency | Floor | Reason |
| --- | --- | --- |
| OpenAI | 1.58.0 | The adapter passes `reasoning_effort` to `chat.completions.create`; [1.58.0 release notes](https://github.com/openai/openai-python/releases/tag/v1.58.0) record this support. |
| Anthropic | 0.28.0 | The adapter uses stable `messages.create(tools=...)`; the [0.28.0 SDK source](https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/v0.28.0/src/anthropic/resources/messages.py) accepts it. |
| Transformers | 4.51.0 | The default Qwen3 model requires this version in its [model card](https://huggingface.co/Qwen/Qwen3-0.6B#quickstart). |
| vLLM (optional) | 0.8.5 | The same model card specifies this floor for Qwen3. |
| Hatchling (build) | 1.27 | SPDX license expressions and `license-files` metadata. |

The remaining floors cover the APIs used: Hydra composition, NumPy array
operations, pandas frame helpers, Accelerate model dispatch, datasets loading,
SciPy peak finding, and optional logging/adapters. The traces extra retains
`manylatents>=0.1.7,<0.2` for the existing trace/metric contract. All declared
dependencies are registry requirements; no Git or private sources are allowed.
The lock's existing versions satisfy every declared floor. Only its root
requirement metadata was edited; resolver validation and fresh installs require
network access. This review is not an execution test of every minimum version.

## PyPI and GitHub setup

At PyPI's [pending publisher form](https://pypi.org/manage/account/publishing/),
register a GitHub publisher with these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `manyagents` |
| GitHub owner | `latent-reasoning-works` |
| GitHub repository | `manyagents` |
| Workflow filename | `release.yml` |
| Environment name | `release` |

Create the GitHub environment `release` and configure its approval/deployment
rules for the organisation's release policy. PyPI publishing uses GitHub OIDC;
there is no PyPI token secret.

The publish job has **only** `id-token: write`. GitHub release uploads additionally
need repository Contents write access, so create an environment secret named
`GH_RELEASE_TOKEN`: a fine-grained GitHub token restricted to this repository
with **Contents: Read and write**. It is used only by the GitHub release steps,
never by the PyPI action. Build and verify have only `contents: read`, with no
OIDC access. The workflow checks that the release secret is present before
publishing to PyPI. Missing permissions or an expired credential can still make
the later GitHub upload fail; PyPI publication cannot be rolled back.

## Checks to run with network access

Run these from the checkout before tagging. They are required because the
restricted worktree cannot resolve build dependencies or install distributions:

```bash
uv lock --check
uv sync --locked --group dev --extra traces
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/ruff check manyagents/ tests/ scripts/
uv build --clear
uvx --from twine twine check --strict dist/*
uv run --no-project --python 3.11 scripts/verify_release.py dist
uv run --no-project --python 3.12 scripts/verify_release.py dist
```

`verify_release.py` creates two clean temporary venvs for each Python version,
installs the original wheel with its dependencies, imports the installed package
from outside the checkout with Python isolated mode, checks every Hydra YAML's
bytes, composes each experiment, and runs both console entry points with
`--help`. It rebuilds a wheel from the sdist, compares package contents, then
repeats the install probes in the second venv. No model downloads or provider
credentials are needed. The archive-content test in the full suite also checks
that development files do not enter the sdist.

## Workflow operation

`.github/workflows/release.yml` has three jobs: build, verify (Python 3.11 and
3.12), and publish. Build uploads the sdist and wheel as one immutable workflow
artifact and records their SHA-256 hashes in its job summary. Verify downloads
that artifact. Publish waits for both verification jobs and uploads the same
original distributions, with no rebuild in the privileged job.

Once the workflow is on the default branch, a manual verification-only run is:

```bash
gh workflow run release.yml --repo latent-reasoning-works/manyagents --ref main -F publish=false
```

Tag pushes matching `v*` publish after verification; the tag must equal
`v` plus the package version (for this release, `v0.2.0`). Manual dispatch also
publishes if `publish=true`. From a branch, that creates the corresponding
version tag at the dispatched commit when creating the GitHub release. From a
tag, the same version check applies. Configure environment rules accordingly.

The first real publish creates the PyPI project through its pending publisher.
The GitHub step then creates the release and attaches both distributions, or
attaches them to an existing release. Existing release assets are never
overwritten. PyPI versions are immutable: use verification-only dispatches for
rehearsals and publish each version once.
