# Phase 3 Integration Planning: External Adapters for Metric Comparison

**Date**: 2025-10-16
**Status**: Planning integration modes for external tools (STELLA, CellForge, etc.)

---

## Current State: Phase 3 Architecture Summary

### What We've Built So Far

#### 1. **Metric Infrastructure** ✅
- **manylatents.metrics.api**: Lightweight in-memory metric computation
  - Dynamic discovery of all metric functions
  - No Hydra overhead, just pure computation
  - Returns geometric quality scores (Trustworthiness, Continuity, etc.)

- **manyagents.metrics.MetricComputer**: Core feature for auditing embeddings
  - ONLY for non-manylatents workflows
  - Takes embeddings + original data → computes metrics
  - Compares multiple embeddings at once

#### 2. **Evaluation Pipeline** ✅
- **manyagents.evaluation.EvaluationPipeline**: Orchestrates comparisons
  - Takes anchor embeddings + comparison embeddings
  - Computes metrics for both
  - Calculates reward signal (metric differences)
  - Supports weighted, single-metric, and threshold reward functions

#### 3. **Config-Driven Workflow** ✅
- **phase3_anchor_vs_comparison.yaml**: Test config
  - Anchor: manylatents PCA → metrics computed natively
  - Comparison: manylatents UMAP → metrics computed natively
  - Works perfectly for **both manylatents** scenarios
  - Successfully tested: Both workflows execute, metrics computed, ready for reward calculation

#### 4. **RL Training Infrastructure** ✅ (Skeleton)
- **manyagents.models.RLModule**: Abstract base for RL algorithms
- **manyagents.models.SB3Module**: Stable-Baselines3 implementation
- Ready for GRPO training on metric vectors

#### 5. **Naming Convention** ✅
- **Adopted: `anchor` / `comparison`**
  - `anchor`: Fixed point for comparison (no quality implication, just a reference)
  - `comparison`: Method being evaluated relative to anchor
- **Rationale**: "reference" implies proven/validated baseline, but anchor can be arbitrary
- **Aligns with**: Contrastive learning terminology, geometric thinking, future flexibility

---

## The Integration Challenge: External Tools vs manylatents

### Key Insight: Not All Tools Produce Embeddings Directly

**STELLA** (https://github.com/zaixizhang/STELLA) is an **LLM-based research assistant** that:
- Takes natural language research questions
- Performs literature review, experimental design, data interpretation
- Outputs: Text-based research insights, NOT numerical embeddings
- Architecture: Multi-agent system (manager, developer, critic) with tool ocean
- Use case: Biomedical research automation

**CellForge** is a **multi-agent computational method design framework** that:
- Phase 1: Task analysis (analyze dataset and research objectives)
- Phase 2: Method design (expert discussion to design approach)
- Phase 3: Code generation (generate executable code)
- Current blocker: Installation issues (syntax errors in package)
- Use case: Generates new computational methods

### The Problem

Our current Phase 3 workflow requires:
```
Anchor: Embeddings (n_samples × n_components) → Metrics (vector of floats)
Comparison: Embeddings (n_samples × n_components) → Metrics (vector of floats)
Reward: Metric difference
```

But tools like STELLA produce:
```
Input: "Design a DR method for single-cell data"
Output: Research insights (text), possibly code suggestions
```

---

## Possible Integration Modes

### **Option 1: STELLA/CellForge as Method Designer → Execute → Evaluate**
**Workflow:**
1. **Anchor**: manylatents PCA → embeddings + metrics
2. **STELLA/CellForge**: Ask "Design a dimensionality reduction method for this dataset"
   - Tool generates code or method description
3. **Execute tool's output**: Run the generated code to produce embeddings
4. **MetricComputer**: Compute metrics on generated embeddings
5. **Reward**: Compare generated metrics vs anchor metrics

**Challenges:**
- Need to parse tool's text output and extract executable code
- Need to safely execute generated code (sandboxing, error handling)
- Tool might not always produce valid/executable methods
- Failure modes: syntax errors, runtime errors, wrong output format

**Benefits:**
- True agentic method design
- Tests if LLM-designed DR beats classical methods
- Very aligned with "learning to design better algorithms"
- Ultimate Phase 3 goal: RL agent learns to prompt LLMs to design better methods

**Implementation Requirements:**
- Code parser (extract Python code from markdown/text)
- Sandbox execution environment
- Output validator (check embeddings shape, type, etc.)
- Error recovery strategy
- Adapter returns: `{'embeddings': np.ndarray, 'generated_code': str, 'success': bool}`

---

### **Option 2: STELLA for Hyperparameter Suggestions**
**Workflow:**
1. **Anchor**: manylatents PCA (default params) → embeddings + metrics
2. **STELLA**: Ask "What hyperparameters should I use for UMAP on this dataset?"
   - STELLA suggests: `n_neighbors=20, min_dist=0.05`
3. **Execute**: Run manylatents UMAP with STELLA's suggested params
4. **Reward**: Compare STELLA-tuned UMAP vs anchor PCA

**Challenges:**
- STELLA might not be great at hyperparameter tuning (not its primary use case)
- Needs structured output parsing (extract key-value pairs from text)
- Quality of suggestions might be inconsistent

**Benefits:**
- Simpler integration (no code execution, just parameter extraction)
- Still tests LLM-guided optimization
- Lower risk (parameters validated by existing manylatents algorithms)

**Implementation Requirements:**
- Output parser (extract hyperparameters from natural language)
- Parameter validator (check types, ranges)
- Fallback to defaults if parsing fails

---

### **Option 3: STELLA for Dataset Analysis → Inform Method Selection**
**Workflow:**
1. **STELLA**: Analyze dataset characteristics
   - "What are the key properties of this single-cell dataset?"
   - Output: "High sparsity, 10k cells, 2k genes, batch effects present"
2. **Method Selection**: Based on STELLA's analysis, choose appropriate DR method
3. **Comparison**: Run selected method vs anchor
4. **Reward**: Metric difference

**Challenges:**
- Very indirect connection to embeddings
- Hard to measure STELLA's contribution to improvement
- Unclear mapping from analysis → method selection

**Benefits:**
- Tests dataset understanding capability
- Could inform meta-learning strategies

---

### **Option 4: Use CellForge Instead (Method Design Tool)**
**Workflow:**
1. **Anchor**: manylatents PCA
2. **CellForge**: Multi-agent method design
   - Phase 1: Task analysis
   - Phase 2: Method design
   - Phase 3: Code generation
3. **Execute CellForge output**: Run generated DR code
4. **MetricComputer**: Evaluate embeddings
5. **Reward**: Compare metrics

**Challenges:**
- CellForge has installation issues (syntax errors)
- Need to fix those first
- Similar code execution challenges as Option 1

**Benefits:**
- Designed specifically for computational method creation
- More direct path to executable code than STELLA
- Multi-phase design could provide richer feedback for RL

**Blocker**: Fix installation first

---

### **Option 5: Simpler Approach - External Adapter with Fixed Algorithm** ⭐ RECOMMENDED FIRST
**Workflow:**
1. **Anchor**: manylatents PCA
2. **Comparison**: External tool with known algorithm (e.g., scanpy, scikit-learn, custom implementation)
3. **MetricComputer**: Evaluate comparison embeddings
4. **Reward**: Metric difference

**Example adapters:**
- **ScikitLearnAdapter**: Wrap sklearn.manifold.TSNE, sklearn.decomposition.PCA
- **ScanpyAdapter**: Wrap scanpy DR methods (sc.tl.pca, sc.tl.umap, sc.tl.tsne)
- **CustomDRAdapter**: Your own DR implementation
- **OpenTSNEAdapter**: Wrap opentsne (optimized t-SNE)

**Benefits:**
- ✅ Validates the infrastructure immediately
- ✅ No LLM complexity
- ✅ Proves MetricComputer works with external tools
- ✅ No installation blockers (most libraries already available)
- ✅ Foundation for more complex integrations

**Implementation:**
```python
class ScikitLearnAdapter(AgentAdapter):
    async def run(self, task_config, input_files):
        algorithm = task_config['algorithm']  # 'TSNE', 'PCA', 'MDS'
        params = task_config.get('params', {})
        input_data = task_config['input_data']

        # Import and instantiate
        if algorithm == 'TSNE':
            from sklearn.manifold import TSNE
            model = TSNE(**params)

        # Fit and transform
        embeddings = model.fit_transform(input_data.cpu().numpy())

        return {
            'embeddings': torch.from_numpy(embeddings),
            'summary': f"{algorithm} completed",
            'success': True,
            'metadata': {'algorithm': algorithm, 'params': params}
        }
```

---

## Recommended Implementation Path

### Phase 3A: Validate Infrastructure (IMMEDIATE) ⭐
1. **Create ScikitLearnAdapter** (simple, no installation needed)
2. **Test anchor vs comparison**: manylatents PCA vs sklearn TSNE
3. **Prove end-to-end**:
   - Anchor workflow executes → metrics computed natively
   - Comparison workflow executes → MetricComputer computes metrics
   - EvaluationPipeline calculates reward
   - Results logged/saved
4. **Branch**: `feat/sklearn-adapter`

### Phase 3B: LLM-Based Method Design (NEXT)
1. **Fix CellForge installation** OR **implement STELLA integration**
2. **Build code execution pipeline**:
   - Code parser
   - Sandbox executor
   - Output validator
3. **Test method design workflow**:
   - Anchor: manylatents PCA
   - Comparison: CellForge/STELLA-generated method
   - MetricComputer evaluates generated embeddings
   - Reward signal for RL training
4. **Branch**: `feat/cellforge-adapter` or `feat/stella-adapter`

### Phase 3C: RL Training Loop (FUTURE)
1. **Create custom gym environment** wrapping manyAgents orchestrator
2. **Implement GRPO training**:
   - Policy generates method design prompts
   - Execute workflow (anchor vs comparison)
   - Reward = metric improvement
   - Update policy via GRPO
3. **Branch**: `feat/grpo-training`

---

## Open Questions for Brainstorming

### Question 1: Integration Mode Priority
Which integration mode should we pursue first?
- [ ] Option 1: STELLA as method designer (parse code, execute, evaluate)
- [ ] Option 4: Fix CellForge and use it for method design
- [ ] Option 5: Simple external adapter first (sklearn, scanpy, custom) ⭐
- [ ] Other approach?

### Question 2: Code Execution Strategy (for Options 1/4)
If we pursue LLM-based method design, how should we handle:
- **Parsing**: Extract code from natural language output
  - Regex for ```python ... ``` blocks?
  - Structured prompting for JSON output?
  - LLM-based code extractor?
- **Execution**: Run generated code safely
  - Docker container?
  - subprocess with timeout?
  - RestrictedPython?
- **Validation**: Ensure output is valid embeddings
  - Shape checking
  - Type checking
  - NaN/Inf detection
- **Failure handling**: What if code doesn't work?
  - Retry with error feedback to LLM?
  - Fall back to default method?
  - Mark as failed and log?

### Question 3: Reward Function Evolution
Current reward is simple metric difference. Future options:
- **Weighted combination**: Already implemented
- **GRPO sequence-based**: Train on full metric vector trajectory
- **Multi-objective**: Pareto frontier of Trustworthiness vs Continuity
- **Threshold-based**: Reward only if improvement > threshold
- **Relative improvement**: Percentage change rather than absolute difference

### Question 4: Adapter Development Strategy
Branch-based development:
- [ ] `feat/sklearn-adapter` (Phase 3A)
- [ ] `feat/scanpy-adapter` (Phase 3A alternative)
- [ ] `feat/stella-adapter` (Phase 3B)
- [ ] `feat/cellforge-adapter` (Phase 3B, needs installation fix)
- [ ] Merge to main after each adapter validated?
- [ ] Keep in separate branches for experimentation?

---

## Technical Specifications

### Adapter Interface for Phase 3
All Phase 3 comparison adapters must return:
```python
{
    'embeddings': torch.Tensor,  # Shape: (n_samples, n_components)
    'summary': str,              # Human-readable description
    'success': bool,             # Execution success flag
    'metadata': {                # Additional info
        'algorithm': str,
        'params': dict,
        'execution_time': float,
        'generated_code': str,   # Optional: for LLM-based adapters
    }
}
```

### MetricComputer Integration
When comparison adapter is NOT manylatents:
```python
# In orchestrator (manyagents/main.py)
comparison_result = await adapter.run(task_config, input_files)

if comparison_result['success']:
    from manyagents.metrics import MetricComputer

    computer = MetricComputer(
        metric_names=['Trustworthiness', 'Continuity'],
        metric_params={'k': 25}
    )

    comparison_metrics = computer.compute(
        x=original_data,
        embeddings={'comparison': comparison_result['embeddings']}
    )['comparison']

    # Add metrics to result for evaluation
    comparison_result['metrics'] = comparison_metrics
```

### Evaluation Pipeline Usage
```python
from manyagents.evaluation import EvaluationPipeline

pipeline = EvaluationPipeline(
    metric_names=['Trustworthiness', 'Continuity'],
    metric_params={'k': 25}
)

results = pipeline.evaluate_embeddings(
    x_original=original_data,
    anchor_embedding=anchor_result['embeddings'],
    comparison_embedding=comparison_result['embeddings'],
    anchor_name='pca',
    comparison_name='stella_generated'
)

# results contains:
# - anchor_metrics: {'Trustworthiness': 0.85, ...}
# - comparison_metrics: {'Trustworthiness': 0.87, ...}
# - metric_differences: {'Trustworthiness': 0.02, ...}
# - reward: 0.018 (weighted combination)
```

---

## Files Modified in This Session

### Created
- `manylatents/metrics/api.py` - Lightweight metrics API
- `manyagents/metrics/compute.py` - MetricComputer for external adapters
- `manyagents/metrics/__init__.py` - Metrics module exports
- `manyagents/evaluation/pipeline.py` - EvaluationPipeline
- `manyagents/evaluation/__init__.py` - Evaluation module exports
- `manyagents/models/rl_module.py` - Abstract RL base class
- `manyagents/models/sb3_module.py` - Stable-Baselines3 implementation
- `manyagents/models/__init__.py` - Models module exports
- `manyagents/configs/logger/minimal.yaml` - Minimal logger config
- `manyagents/configs/logger/wandb.yaml` - WandB logger config
- `manyagents/configs/experiment/phase3_anchor_vs_comparison.yaml` - Phase 3 test config

### Updated
- `CLAUDE.md` - Added Phase 3 architecture documentation
- Renamed: `phase3_baseline_vs_variant.yaml` → `phase3_anchor_vs_comparison.yaml`

### Tested
- ✅ phase3_anchor_vs_comparison workflow (PCA vs UMAP, both manylatents)
- ✅ In-memory state threading (PCA output → UMAP input)
- ✅ Metrics computation via test_metric
- ✅ WandB integration with minimal callbacks

---

## Next Steps (After Brainstorming)

1. **Decision**: Choose integration mode (Option 1, 4, or 5)
2. **Implementation**: Build selected adapter
3. **Configuration**: Create phase3_anchor_vs_[tool].yaml config
4. **Testing**: Run end-to-end workflow with MetricComputer
5. **Validation**: Verify reward calculation works correctly
6. **Documentation**: Update CLAUDE.md with adapter integration patterns
7. **Branch Strategy**: Decide on merge vs keep-separate approach

---

## Notes
- STELLA is LLM-based research assistant, not DR tool
- CellForge has installation blockers
- ScikitLearnAdapter is simplest path to validate infrastructure
- Future: GRPO training on metric vectors from these comparisons
