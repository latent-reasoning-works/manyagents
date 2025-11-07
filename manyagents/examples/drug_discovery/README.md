"""
Drug Discovery Example for ManyAgents

This example demonstrates how to use manyAgents' generic utilities for
domain-specific workflows. It showcases the clean separation between:

1. **Generic utilities** (in manyagents/utils/)
2. **Model adapters** (in manyagents/adapters/)
3. **Domain-specific logic** (in manyagents/examples/)

## Architecture

### Utils (Generic, Reusable)
```
manyagents/utils/
├── data_ops.py      # DataFrame loading, merging, transforming
├── scoring.py       # Ranking and scoring operations
└── validation.py    # Filtering and validation logic
```

**Purpose**: Reusable across ANY domain (finance, e-commerce, drug discovery, etc.)

### Adapters (Model-Specific Wrappers)
```
manyagents/adapters/
├── manylatents_adapter.py    # Wraps manylatents package
├── cellforge_adapter.py      # Wraps cellforge tool
└── biodiscovery_adapter.py   # Wraps biodiscovery agent
```

**Purpose**: Each adapter wraps a COMPLETE external model/tool

### Examples (Domain-Specific Logic)
```
manyagents/examples/drug_discovery/
├── data.py         # Drug-specific data loading (ChEMBL, LINCS, DrugBank)
├── scoring.py      # Drug repurposing scoring logic
└── validation.py   # Drug safety validation logic
```

**Purpose**: Domain-specific implementations that USE the generic utils

## Usage Patterns

### Pattern 1: Adapters Use Utils Internally

```python
# Inside ManyLatentsAdapter
from manyagents.utils.data_ops import load_and_merge_dataframes

class ManyLatentsAdapter(AgentAdapter):
    async def run(self, task_config, input_files, input_data):
        # Use utils to prepare data
        data = await load_and_merge_dataframes(sources=[...])
        
        # Run the model
        embeddings = await manylatents.run(data)
        return embeddings
```

### Pattern 2: Workflows Use Utils Directly

```yaml
workflow:
  steps:
    # Step 1: Use utils directly for data integration
    - name: integrate_data
      operation: data_ops.load_and_merge
      config:
        sources:
          - transform_function: "manyagents.examples.drug_discovery.data:fetch_chembl"
            params: {query: "kinase_inhibitors", limit: 500}
        merge_strategy: "inner_join"
    
    # Step 2: Use model adapter
    - name: embed
      agent: manylatents
      config:
        algorithm: pca
        n_components: 50
    
    # Step 3: Use utils for scoring
    - name: rank_candidates
      operation: scoring.score_and_rank
      config:
        scoring_function: "manyagents.examples.drug_discovery.scoring:score_repurposing"
        threshold: 0.6
```

### Pattern 3: Domain Examples Call Utils

```python
# manyagents/examples/drug_discovery/data.py
from manyagents.utils.data_ops import load_and_merge_dataframes

async def fetch_all_sources():
    sources = [
        {"transform_function": "...:fetch_chembl", "params": {...}},
        {"transform_function": "...:fetch_expression", "params": {...}}
    ]
    return await load_and_merge_dataframes(sources, merge_strategy="inner")
```

## Benefits of This Architecture

1. **Generic = Reusable**: Utils work for ANY domain
2. **Adapters = Model Wrappers**: Clean separation of model orchestration
3. **Examples = Show How**: Domain examples demonstrate best practices
4. **Composable**: Mix and match utils, adapters, and domain logic

## Creating Your Own Domain Example

To create a new domain example (e.g., finance, e-commerce):

1. **Create domain module**:
   ```
   manyagents/examples/finance/
   ├── __init__.py
   ├── data.py         # Load stock prices, fundamentals, etc.
   ├── scoring.py      # Score investment opportunities
   └── validation.py   # Filter by risk criteria
   ```

2. **Use generic utils**:
   ```python
   from manyagents.utils.data_ops import load_and_merge_dataframes
   from manyagents.utils.scoring import score_and_rank
   from manyagents.utils.validation import validate_and_filter
   ```

3. **Reference in workflows**:
   ```yaml
   transform_function: "manyagents.examples.finance.data:load_stock_prices"
   scoring_function: "manyagents.examples.finance.scoring:score_investments"
   filter_function: "manyagents.examples.finance.validation:filter_by_risk"
   ```

## Drug Discovery Functions

### Data Loading
- `fetch_chembl(query, limit)`: Drug-target interactions
- `fetch_expression(dataset, limit)`: Gene expression signatures
- `fetch_drugbank(limit)`: Approved drug information
- `fetch_all_sources(...)`: Integrated data pipeline

### Scoring
- `score_repurposing(data, embeddings, method, failure_mode)`: 
  - Methods: "embedding_similarity", "target_overlap", "combined"
  - Failure modes: None, "overfit", "target_bias" (for demos)

### Validation
- `validate_safety(data, checks, confidence_boost)`:
  - Checks: "toxicity", "ddi", "adverse_events", "approval_status"
- `validate_strict(data)`: Reject any with flags
- `validate_permissive(data)`: Flag but keep all

## Example Workflow

See configs/experiment/drug_repurposing_pipeline.yaml for complete example.
"""
