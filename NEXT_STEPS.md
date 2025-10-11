# Next Steps: WandB Integration for Merge Readiness

## Current Status: ⚠️ BLOCKED - Missing WandB Metrics/Plots

### What Works ✅
1. **ManyAgents → ManyLatents integration**: Fully functional
   - Orchestrator passes configs to adapter correctly
   - Adapter calls `manylatents.api.run()` successfully
   - Step-based wandb naming works: runs named `step0_pca_reduction`
   - In-memory data passing between steps works
   - Validation system prevents config errors

2. **Standalone manylatents CLI**: Unchanged and functional
   ```bash
   python -m manylatents.main experiment=single_algorithm
   ```

3. **Basic manyagents workflow**: Executes successfully
   ```bash
   python -m manyagents.main experiment=manylatents_single_algorithm
   # Creates wandb run: step0_pca_reduction
   # But: No metrics tables or embedding plots logged to wandb
   ```

### What's Broken ❌

**Issue**: Wandb runs are created but **metrics and plots are NOT logged**

**Root Cause**: The example configs don't specify callbacks/metrics properly

**Current behavior**:
- Wandb run created ✅
- Run has correct name (`step0_pca_reduction`) ✅
- But: Empty run - no metrics, no plots ❌

**Expected behavior**:
- Wandb run should show:
  - Geometric metrics table (trustworthiness, continuity, etc.)
  - Embedding plot visualization
  - Scores logged from manylatents evaluation

---

## The Problem: Two Approaches, Both Have Issues

### Approach 1: Direct Config (Current)
**File**: `manyagents/configs/experiment/manylatents_single_algorithm.yaml`

```yaml
workflow:
  steps:
    - name: pca_reduction
      agent: manylatents
      config:
        algorithm: pca
        data: swissroll
        n_components: 2
        project: manyagents_examples
        # MISSING: callbacks and metrics!
```

**Problem**: How do we add callbacks/metrics in YAML format that works with `api.run()`?

**Attempted solution** (failed):
```yaml
config:
  algorithm: pca
  callbacks:
    embedding:
      plot_embeddings:
        _target_: manylatents.callbacks.embedding.plot_embeddings.PlotEmbeddings
```

**Error**: `no viable alternative at input '{'embedding''`
**Reason**: `api.run()` tries to convert nested dicts to Hydra override strings, which fails

---

### Approach 2: Experiment Reference (Preferred but broken)
**Concept**: Reference a manylatents experiment that already has callbacks/metrics configured

```yaml
workflow:
  steps:
    - name: pca_reduction
      agent: manylatents
      config:
        experiment: single_algorithm  # Loads manylatents experiment
        project: manyagents_examples  # Override project name
```

**Problem**: Structured config merge error

**Error**: `Key 'embedding' is not in struct`
**Location**: `manylatents/api.py:114` - `OmegaConf.update(cfg, key, value, merge=True)`

**Root cause**:
1. `load_manylatents_experiment()` loads a **structured config** (from manylatents Config dataclass)
2. When we try to merge `project` override, it passes dict to `api.run()`
3. `api.run()` calls `OmegaConf.update()` which expects structured config
4. Structured configs don't allow adding fields not in the schema

---

## What Needs to be Fixed

### Option A: Fix in ManyLatents API (Recommended)
**File to modify**: `manylatents/api.py`

**Current code** (lines ~92-114):
```python
# Compose the base configuration
cfg = compose(config_name="config", overrides=override_list)

# Merge complex overrides (dicts and lists) directly
for key, value in overrides.items():
    if isinstance(value, (dict, list)):
        if OmegaConf.select(cfg, key) is None:
            OmegaConf.update(cfg, key, value, merge=False)
        else:
            OmegaConf.update(cfg, key, value, merge=True)  # <-- FAILS HERE
```

**Problem**: `cfg` is a **structured config** (from Config dataclass), so `OmegaConf.update()` validates against the schema

**Solution**: Set `struct=False` before merging
```python
# Compose the base configuration
cfg = compose(config_name="config", overrides=override_list)

# Allow flexible field additions (disable struct mode)
OmegaConf.set_struct(cfg, False)

# Merge complex overrides (dicts and lists) directly
for key, value in overrides.items():
    if isinstance(value, (dict, list)):
        if OmegaConf.select(cfg, key) is None:
            OmegaConf.update(cfg, key, value, merge=False)
        else:
            OmegaConf.update(cfg, key, value, merge=True)
```

**Impact**: Allows `api.run()` to accept arbitrary config overrides without schema violations

---

### Option B: Fix in ManyAgents Config (Workaround)
**File to modify**: `manyagents/configs/experiment/manylatents_single_algorithm.yaml`

**Workaround**: Use Hydra config group syntax instead of nested dicts

**Attempt this syntax**:
```yaml
workflow:
  steps:
    - name: pca_reduction
      agent: manylatents
      config:
        algorithm: pca
        data: swissroll
        n_components: 2
        project: manyagents_examples
        # Use Hydra defaults syntax
        defaults:
          - override /metrics: test_metric
          - override /callbacks/embedding: default
```

**Status**: Untested - may not work with current adapter implementation

**Alternative**: Pass Hydra override strings directly
```yaml
config:
  _hydra_overrides:
    - "algorithm=pca"
    - "data=swissroll"
    - "metrics=test_metric"
    - "callbacks/embedding=default"
```

**Status**: Would require adapter changes to handle `_hydra_overrides` key

---

## Recommended Action Plan

### Phase 1: Fix ManyLatents API (30 minutes)
1. **Edit `manylatents/api.py`** line ~105:
   ```python
   # Add before the merge loop:
   OmegaConf.set_struct(cfg, False)
   ```

2. **Test the fix**:
   ```bash
   # From manyAgents directory
   python -m manyagents.main experiment=manylatents_single_algorithm
   ```

3. **Verify wandb output**:
   - Check wandb run has metrics table
   - Check wandb run has embedding plot
   - Confirm scores are logged

4. **If successful**: Update `manylatents_single_algorithm.yaml` to use experiment reference:
   ```yaml
   config:
     experiment: single_algorithm
     project: manyagents_examples
   ```

### Phase 2: Test Multi-Step Pipeline (15 minutes)
1. **Update `manylatents_multi_step_pipeline.yaml`** to use experiment references:
   ```yaml
   workflow:
     steps:
       - name: pca_preprocessing
         agent: manylatents
         config:
           experiment: single_algorithm
           project: manyagents_examples
           algorithms:
             latent:
               n_components: 50

       - name: umap_visualization
         agent: manylatents
         config:
           algorithm: umap  # Direct config for 2nd step
           n_components: 2
           project: manyagents_examples
           defaults:
             - override /metrics: test_metric
             - override /callbacks/embedding: default
   ```

2. **Test**:
   ```bash
   python -m manyagents.main experiment=manylatents_multi_step_pipeline
   ```

3. **Verify**:
   - Two separate wandb runs created
   - Both have metrics and plots
   - In-memory data passing works between steps

### Phase 3: Documentation & Merge (15 minutes)
1. **Update example config comments** with working patterns
2. **Add note to README** about wandb integration
3. **Create merge summary** documenting what works
4. **Commit and push** both branches

---

## Quick Reference Commands

### Test Current State
```bash
cd /network/scratch/c/cesar.valdez/manyAgents
source .venv/bin/activate

# Test basic integration (works but no metrics)
python -m manyagents.main experiment=manylatents_single_algorithm

# Check wandb run
# Should see run at: https://wandb.ai/.../manyagents_examples/runs/...
# Currently: Empty run (no metrics/plots)
```

### Test Standalone ManyLatents (Reference)
```bash
cd /network/scratch/c/cesar.valdez/manyLatents
source .venv/bin/activate

# This works and logs metrics/plots correctly
python -m manylatents.main experiment=single_algorithm

# Check output - should see:
# - Metrics logged to wandb
# - Embedding plot in outputs/
```

### After Fix - Expected Behavior
```bash
cd /network/scratch/c/cesar.valdez/manyAgents
source .venv/bin/activate

python -m manyagents.main experiment=manylatents_single_algorithm

# Should now show in wandb:
# - Metrics table (trustworthiness, continuity, etc.)
# - Embedding plot
# - Scores logged from evaluation
```

---

## Files to Modify

### Critical (Must fix for merge)
- [ ] **`manylatents/api.py:~105`** - Add `OmegaConf.set_struct(cfg, False)`

### Nice to have (Can be done after merge)
- [ ] `manyagents/configs/experiment/manylatents_single_algorithm.yaml` - Switch to experiment reference
- [ ] `manyagents/configs/experiment/manylatents_multi_step_pipeline.yaml` - Add proper configs
- [ ] `manyagents/configs/experiment/README.md` - Document working patterns

---

## Key Insights for Future

1. **Structured configs are brittle**: The Config dataclass in manylatents creates a rigid schema
   - This is good for validation
   - But bad for flexible API usage
   - Solution: `OmegaConf.set_struct(cfg, False)` before merging arbitrary configs

2. **API flexibility matters**: An API should accept both:
   - Experiment names (high-level, convenient)
   - Direct configs (low-level, flexible)
   - Our design supports both, but needs the struct fix

3. **Validation vs Flexibility tradeoff**:
   - Schema-on-write (structured configs): Catches errors early, inflexible
   - Schema-on-read (our approach): Flexible, validates at runtime
   - Hybrid: Use structured configs internally, but allow flexible API inputs

---

## Success Criteria for Merge

- [ ] Wandb run shows metrics table
- [ ] Wandb run shows embedding plot
- [ ] `experiment=manylatents_single_algorithm` works with full logging
- [ ] `experiment=manylatents_multi_step_pipeline` works with two logged runs
- [ ] Standalone manylatents CLI still works unchanged
- [ ] Documentation updated with working examples

**Estimated time to completion**: ~1 hour (mostly testing and verification)

---

## Contact Points

**When you continue:**
1. Start with the manylatents API fix (`set_struct(False)`)
2. Test thoroughly with both single and multi-step workflows
3. Verify wandb output has all expected data
4. If successful, we're ready to merge!

**Blockers**: None - all issues are understood and have clear solutions
