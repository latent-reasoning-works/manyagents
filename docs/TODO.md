# TODO: Future Improvements

## High Priority

### Fast Dev Run Mode
**Status**: Planned
**Priority**: High (needed for CI and local testing)

**Goal**: Add a `fast_dev_run` mode (like Lightning's) that configures everything for rapid testing/CI.

**Current State**:
We're almost there:
- ✅ SwissRoll dataset is fast (synthetic, no I/O)
- ✅ PCA algorithm is fast (simple linear algebra)
- ✅ test_metric is instant (returns zeros)
- ⚠️ wandb callbacks add significant overhead (logging, uploading)
- ⚠️ debug mode couples wandb disabling with verbose logging

**Proposed Implementation**:

```yaml
# manylatents/configs/fast_dev_run.yaml (or flag in main)
fast_dev_run: true  # When enabled:
  # 1. Remove wandb callback entirely (not just disable)
  # 2. Use minimal dataset (swissroll with n_samples=100)
  # 3. Use fast algorithm (PCA)
  # 4. Use test_metric (instant)
  # 5. Skip expensive operations (checkpointing, artifact saving)
  # 6. Keep normal log level (INFO, not DEBUG)
```

**Benefits**:
- **Fast CI**: Tests run in seconds, not minutes
- **Clean logs**: No wandb overhead, no debug verbosity
- **Developer friendly**: Quick validation before pushing
- **Explicit intent**: Clear that this is a test mode, not partial config

**Implementation Options**:

1. **Hydra flag approach**:
   ```bash
   python -m manylatents.main experiment=single_algorithm fast_dev_run=true
   ```
   - Conditionally exclude wandb callback in config
   - Override dataset/algorithm params for speed

2. **Dedicated experiment config**:
   ```yaml
   # configs/experiment/fast_dev_run.yaml
   defaults:
     - override /algorithms/latent: pca
     - override /data: swissroll_tiny  # n_samples=100
     - override /metrics: test_metric
     - override /callbacks/embedding: minimal  # No wandb
   ```

3. **Callback group approach** (RECOMMENDED):
   ```yaml
   # configs/callbacks/embedding/minimal.yaml
   defaults:
     - save_embeddings
     - plot_embeddings
     # NO wandb_log_scores
   ```

**Recommended Approach**: Option 3 (callback group)
- Most composable
- Doesn't require new infrastructure
- Works with existing configs
- Easy to maintain

**Changes Needed**:

1. **In manylatents**:
   ```yaml
   # configs/callbacks/embedding/minimal.yaml
   defaults:
     - save_embeddings
     - plot_embeddings
     - _self_

   # configs/experiment/fast_dev_run.yaml
   defaults:
     - override /algorithms/latent: pca
     - override /data: swissroll
     - override /callbacks/embedding: minimal
     - override /metrics: test_metric

   seed: 42
   project: fast_dev_test

   # Small, fast dataset
   data:
     n_distributions: 10
     n_points_per_distribution: 10  # Only 100 samples total
     rotate_to_dim: 50  # Smaller dimension

   # Fast algorithm
   algorithms:
     latent:
       n_components: 2
   ```

2. **In manyAgents**:
   ```yaml
   # configs/experiment/manylatents_fast_dev_run.yaml
   name: manylatents_fast_dev_run

   workflow:
     steps:
       - name: test_step
         agent: manylatents
         config:
           experiment: fast_dev_run  # Uses the fast config above
   ```

3. **In GitHub Actions**:
   ```yaml
   # .github/workflows/integration-tests.yml
   - name: Test manylatents integration
     run: |
       python -m manyagents.main experiment=manylatents_fast_dev_run
   ```

**Estimated Time**: 1-2 hours

**Testing**:
```bash
# Should complete in < 10 seconds
time python -m manylatents.main experiment=fast_dev_run

# Through manyAgents
time python -m manyagents.main experiment=manylatents_fast_dev_run
```

---

## Medium Priority

### Separate Debug Concerns
**Status**: Related to fast_dev_run
**Priority**: Medium

**Problem**: Current `debug` flag couples two concerns:
- Disabling wandb
- Enabling verbose logging

**Solution**: Once `fast_dev_run` is implemented, deprecate `debug` flag or make it only control logging verbosity.

---

## Low Priority

### Orchestration-Level Metrics
**Status**: Future (Phase 2+)
**Priority**: Low

**Goal**: Metrics that evaluate entire workflows, not just individual components.

Examples:
- Pipeline efficiency (time per step)
- Data size reduction across steps
- Metric improvement across sequential steps
- Cross-tool consistency checks

This would require a different testing paradigm than component-level metrics.
