# Experiment configurations

These are the experiment configs shipped with manyagents. Select one with `experiment=<name>`; bare `manyagents` lists the available names and exits nonzero. See [Running Experiments](../../../docs/running_experiments.md) for setup and execution details.

## Start here

| Config | Purpose |
| --- | --- |
| `test_wandb.yaml` | Two mock prompts; runs locally without keys, GPU, or model downloads |
| `geometric_reasoning.yaml` | Nine prompts across three domains and three information conditions |
| `invariance_golden.yaml` | Golden invariance evaluation configuration |
| `invariance_full.yaml` | Four-prompt evaluation with Claude, OpenAI, local LLM, and Biomni agents |
| `invariance_compare_models.yaml` | Model comparison configuration |
| `trace_extraction.yaml` | Reasoning trace extraction; defaults to HF and newline segmentation |
| `reasoning_baseline.yaml` | Local reasoning baseline |
| `baseline_sweep.yaml` | Baseline sweep settings |
| `llm_reasoning_sweep.yaml` | Local-model reasoning sweep settings |

Inspect each config's defaults and comments for models, datasets, and dependencies. API experiments require credentials; local models require accessible weights and suitable hardware. Sweeps may include site-specific choices.

```bash
manyagents experiment=test_wandb
manyagents experiment=geometric_reasoning 'active_agents=[mock]'
# Inspect configuration without executing:
manyagents experiment=trace_extraction --cfg job
```

The text-evaluation runner expects text-producing adapters. For dimensionality reduction workflows, use `manyagents.workflows.sequence.execute_sequence` or `ManyLatentsAdapter` with the `traces` extra; there are no shipped single-algorithm or multi-step manylatents experiment configs.

## The 3x3 geometric-reasoning design

The shipped single-cell evaluation suite asks models one question: does the method you recommend fit the shape of the data? The failure it catches is easy to see, a model that recommends the same pipeline for every dataset, and the expected and forbidden methods per prompt are YAML drawn from the fixed vocabulary above. The 3×3 design and its scoring are in [experiment configurations](manyagents/configs/experiment/README.md).

> 3 domains × 3 information conditions = 9 prompts, 3 scores

`geometric_reasoning` poses the same analysis question three ways for each of three single-cell scenarios:

| domain | expected geometry | ground truth includes | failure indicators include |
|---|---|---|---|
| immunology (PBMCs) | discrete clusters | leiden, louvain, kmeans, phenograph | pseudotime, monocle, slingshot |
| cancer (EMT time course) | branching trajectory | slingshot, monocle3, paga, cellrank, palantir | clustering, leiden, kmeans |
| developmental (organoids) | continuous manifold | phate, diffusion_map, umap, isomap | clustering, leiden, louvain |

Condition **A** gives biological context without explicit embedding hints. **B** adds what the embedding looks like. **C** emphasizes geometry and reduces biological context, though some remains (for example single-cell data and timepoints), and the shared system prompt is still computational biology. Better performance on B/C is a hypothesis to test, not a guaranteed consequence of reasoning about structure.

Three scores per agent, written to `summary.md`:

- **Ground-truth match rate:** fraction of successful prompts with at least one extracted expected method and no extracted failure indicator. Higher means more passes against the configured criteria.
- **Jaccard across prompts:** mean method-set overlap over **all successful prompt pairs**, including pairs within the same geometry. High overlap signals invariance; **lower is not always better**. With nine successful prompts there are 36 pairs, nine within a geometry. One consistent nonempty set per geometry, disjoint across geometries, scores **0.25**. Shared methods raise this value; inconsistent answers within a geometry can lower it. Two empty sets have similarity 1.0. This is an invariance signal, not an optimization objective.
- **Clustering-for-all:** fraction of successful prompts with an extracted clustering method or phrase, regardless of expected structure. High values flag broad clustering use in this mixed-geometry design; they do not establish whether individual uses are appropriate.

Scoring uses a vocabulary of named tools (clustering, trajectory, DR, cell-cycle, spatial, integration, annotation, differential expression) plus phrases such as "pseudotime analysis". Explicit local rejection cues — “do not use”, “avoid”, “instead of”, “rather than”, “not appropriate”, “would be wrong”, and related forms — filter individual occurrences, including coordinated lists. Prefix scope is limited to eight words after the cue, sentence/contrast boundaries, and new affirmative recommendation cues. A separate unrejected occurrence still counts. Thus “Avoid Leiden; use UMAP” can pass, while “Use Leiden and UMAP” fails a criterion that forbids Leiden.

This remains a heuristic, not a scientific answer judge: bare hedges (“might use”), quoted or hypothetical advice, distant negation, and complex scope can still be misread. Unrejected mentions need not be definite recommendations. `ground_truth_matches` and `match_ratio` describe vocabulary overlap even when `failure_matches` blocks the pass. Execution failures are excluded from all three scores; inspect `prompts_evaluated` and `prompts_failed` alongside rates. Measurements that cannot be computed are `null` in `results.json` and `n/a` in summaries; Jaccard needs at least two successful prompts, and a missing ground-truth criterion makes the whole match rate unavailable rather than quietly narrowing the denominator.

Other shipped experiments vary the framing: `invariance_full` adds periodic (cell cycle) and spatial-gradient geometries; `reasoning_baseline` uses descriptions of synthetic manifolds (swiss roll, torus) and embryoid-body data; `llm_reasoning_sweep` varies models and scenarios, while `baseline_sweep` varies adapters, datasets, algorithms, and dimensions.

## Add a project experiment

Copy a suitable shipped config into your own `configs/experiment/` directory, retain `# @package _global_`, and configure agents and prompts. For a checkout, add your directory to Hydra's search path:

```bash
manyagents 'hydra.searchpath=[file:///absolute/path/to/your_project/configs]' experiment=my_experiment
```

Keep project-specific data and model choices in project configs. Keep cluster paths and resource allocations in the `cluster=` and `resources=` config groups.


## Reading the scores

That is what failure looks like. The mock answers every prompt with the same three methods: identical sets across three expected geometries (Jaccard 1.0), clustering recommended for a continuous manifold (clustering-for-all 100%). Read the three together; match rate alone hides it.

**Scoring is a heuristic.** The extractor finds mentions of the single-cell methods in its fixed vocabulary, drops any mention under a local rejection cue (“avoid”, “do not use”, “instead of”), and passes a prompt when at least one expected method survives and no configured failure indicator does. A term outside the vocabulary is invisible to both lists, and hedges, quoted advice, and distant negation get through. Jaccard averages method-set overlap over all successful prompt pairs, same-geometry pairs included: one consistent answer per geometry, disjoint across geometries, scores 0.25 on the 3×3. Treat it as an invariance signal, never as something to minimise.
