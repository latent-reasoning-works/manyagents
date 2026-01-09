# Pipeline Invariance Experiment Results

**Experiment ID:** invariance_20260108_225126
**Timestamp:** 2026-01-08T22:53:02.771277

## Summary Metrics

| System | Jaccard (Invariance) | Ground Truth Match | Clustering-for-All |
|--------|---------------------|-------------------|-------------------|
| local_llm | 1.00 | 100.0% | 100.0% |

## Interpretation

- **Jaccard (Invariance):** Higher = more similar recommendations across prompts (BAD for geometry-aware systems)
- **Ground Truth Match:** Higher = recommendations match expected methods for each geometry type (GOOD)
- **Clustering-for-All:** Higher = always recommends clustering regardless of data structure (BAD)
