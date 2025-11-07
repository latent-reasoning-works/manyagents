# ManyAgents Architecture: Adapters vs Utils

**Date**: November 6, 2025  
**Decision**: Separate generic operations (utils) from model wrappers (adapters)

---

## Core Principle

**Adapters** wrap complete external models/tools.  
**Utils** provide reusable operations that models and workflows use.

---

## Directory Structure

```
manyagents/
├── adapters/              # Model-specific wrappers
│   ├── base.py           # AgentAdapter ABC
│   ├── manylatents_adapter.py
│   ├── cellforge_adapter.py
│   └── biodiscovery_adapter.py
│
├── utils/                # Generic, reusable operations
│   ├── data_ops.py       # DataFrame loading, merging, transforming
│   ├── scoring.py        # Ranking and scoring
│   └── validation.py     # Filtering and validation
│
├── examples/             # Domain-specific demonstrations
│   └── drug_discovery/
│       ├── data.py       # ChEMBL, LINCS, DrugBank loaders
│       ├── scoring.py    # Drug repurposing scoring
│       └── validation.py # Safety validation
│
└── main.py              # Orchestration engine
```

---

## What Goes Where?

### ✅ Adapters (Model Wrappers)

**Criteria**: Wraps a complete external model/tool/agent

**Examples**:
- `ManyLatentsAdapter` → Wraps entire manylatents package
- `CellForgeAdapter` → Wraps cellforge tool
- `STELLAAdapter` → Wraps STELLA model

**Interface**:
```python
class ModelAdapter(AgentAdapter):
    async def run(self, task_config, input_files, input_data):
        # Prepare inputs (may use utils)
        # Call the external model
        # Return standardized outputs
```

### ✅ Utils (Generic Operations)

**Criteria**: Reusable across ANY domain, not model-specific

**Examples**:
- `data_ops.load_and_merge_dataframes()` → Works for ANY tabular data
- `scoring.score_and_rank()` → Works for ANY ranking task
- `validation.validate_and_filter()` → Works for ANY filtering task

**Usage**:
1. Called by adapters internally
2. Called directly in workflows
3. Called by domain examples

### ✅ Examples (Domain-Specific Logic)

**Criteria**: Shows how to use utils for specific domain

**Examples**:
- `drug_discovery.data.fetch_chembl()` → Drug-specific data loading
- `drug_discovery.scoring.score_repurposing()` → Drug-specific scoring
- `finance.data.load_stock_prices()` → Finance-specific data loading

**Pattern**: Examples USE utils, they don't duplicate them

---

## Usage Patterns

### Pattern 1: Adapter Uses Utils Internally

Adapters can use utils to prepare data before calling their model:

```python
from manyagents.utils.data_ops import merge_dataframes

class ManyLatentsAdapter(AgentAdapter):
    async def run(self, task_config, input_files, input_data):
        # Use util to merge input DataFrames
        data = merge_dataframes([df1, df2], strategy="inner")
        
        # Run the model
        result = await manylatents.run(data, **task_config)
        return result
```

### Pattern 2: Workflow Uses Utils Directly

Workflows can call utils as standalone operations:

```yaml
workflow:
  steps:
    - name: prepare_data
      operation: data_ops.load_and_merge
      config:
        sources: [...]
    
    - name: run_model
      agent: manylatents  # Uses adapter
      config:
        algorithm: pca
```

### Pattern 3: Examples Call Utils

Domain examples show best practices for using utils:

```python
# examples/drug_discovery/data.py
from manyagents.utils.data_ops import load_and_merge_dataframes

async def fetch_all_sources():
    """Example of integrated drug data loading."""
    return await load_and_merge_dataframes(
        sources=[...],
        merge_strategy="inner_join"
    )
```

---

## Why This Separation?

### Before (Wrong)
```
adapters/
├── manylatents_adapter.py        # Model wrapper ✓
├── data_integration_adapter.py   # Generic operation ✗
├── hypothesis_adapter.py         # Generic operation ✗
└── validation_adapter.py         # Generic operation ✗
```
**Problem**: Mixing model wrappers with generic operations

### After (Right)
```
adapters/
└── manylatents_adapter.py        # Model wrappers only

utils/
├── data_ops.py                   # Generic operations
├── scoring.py
└── validation.py

examples/drug_discovery/
├── data.py                       # Domain-specific uses of utils
├── scoring.py
└── validation.py
```
**Benefit**: Clear separation of concerns

---

## Benefits

1. **Reusability**: Utils work across ANY domain (drug discovery, finance, e-commerce)
2. **Clarity**: Adapters are clearly model wrappers, not generic operations
3. **Composability**: Mix and match utils, adapters, and domain logic
4. **Maintainability**: Change utils once, benefit everywhere
5. **Extensibility**: Add new domains without touching core infrastructure

---

## Creating New Domain Examples

Want to add a new domain (e.g., finance)? Follow this pattern:

1. **Create domain module**:
   ```
   manyagents/examples/finance/
   ├── __init__.py
   ├── data.py         # Domain-specific data loaders
   ├── scoring.py      # Domain-specific scoring
   └── validation.py   # Domain-specific filtering
   ```

2. **Use existing utils**:
   ```python
   from manyagents.utils import data_ops, scoring, validation
   ```

3. **NO new adapters needed** unless wrapping a new model

---

## Example: Drug Discovery Workflow

```yaml
workflow:
  steps:
    # Use utils with domain-specific functions
    - name: integrate_data
      operation: data_ops.load_and_merge
      config:
        sources:
          - transform_function: "manyagents.examples.drug_discovery.data:fetch_chembl"
        merge_strategy: "inner_join"
    
    # Use model adapter
    - name: embed
      agent: manylatents
      config:
        algorithm: pca
        n_components: 50
    
    # Use utils with domain-specific functions
    - name: score
      operation: scoring.score_and_rank
      config:
        scoring_function: "manyagents.examples.drug_discovery.scoring:score_repurposing"
        threshold: 0.6
    
    # Use utils with domain-specific functions
    - name: validate
      operation: validation.validate_and_filter
      config:
        filter_function: "manyagents.examples.drug_discovery.validation:validate_safety"
        filter_mode: "strict"
```

---

## Key Takeaway

**Adapters = Model Wrappers** (manylatents, cellforge, STELLA)  
**Utils = Generic Operations** (data, scoring, validation)  
**Examples = Domain Demonstrations** (drug discovery, finance)

This keeps manyAgents clean, reusable, and extensible! 🚀
