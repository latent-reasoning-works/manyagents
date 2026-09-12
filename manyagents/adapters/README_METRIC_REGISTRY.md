# Metric Registry

The metric registry provides fast lookup of manyLatents metrics for use in RL training loops.

## Overview

The registry maps simple metric names (e.g., `participation_ratio`) to their:
- Full class path
- Default parameters
- Group (embedding/dataset/module)
- Source (core manylatents or extension package)

## Usage

### In Code

```python
from manyagents.adapters.metric_registry import MetricRegistry

# Load registry
registry = MetricRegistry()

# List available metrics
all_metrics = registry.list_metrics()
embedding_metrics = registry.list_metrics(group='embedding')

# Get metric information
info = registry.get_metric_info('participation_ratio')
print(info['class'])     # manylatents.metrics.participation_ratio.ParticipationRatio
print(info['defaults'])  # {'n_neighbors': 25, 'return_per_sample': True}
print(info['group'])     # 'embedding'

# Get metric class for instantiation
metric_class = registry.get_metric_class('participation_ratio')
```

## Generation

### During Development

Generate the registry manually:
```bash
# Using CLI command
manyagents-generate-registry

# With verbose output
manyagents-generate-registry --verbose

# Force regeneration
manyagents-generate-registry --force
```

### At Runtime and During Package Build

`MetricRegistry()` discovers the installed manylatents metrics and algorithms in memory. It requires the `traces` extra at use time and does not write into the installed package. There is no Hatch registry build hook; building a wheel does not require manylatents.

The CLI prints JSON to stdout by default. To persist an explicit cache in a writable location:

```bash
manyagents-generate-registry --output /tmp/manyagents-registry.json
```

```python
from pathlib import Path
registry = MetricRegistry(Path("/tmp/manyagents-registry.json"))
```

An explicit cache write failure is logged and the generated in-memory registry remains usable.

## Extension Support

The registry automatically discovers and includes metrics from installed manyLatents extensions:

### Automatic Discovery

Any installed package matching `manylatents-*` or `manylatents_*` is scanned for metrics.

Example with `manylatents-omics`:
```bash
# Install extension
uv add manylatents-omics

# Regenerate registry (picks up extension metrics)
manyagents-generate-registry

# Check what was found
manyagents-generate-registry --verbose
```

Output:
```
Scanning manyLatents core metrics...
Discovered extension: manylatents-omics at .../configs/metrics
Scanning extension 'manylatents-omics' metrics...
✅ Generated registry with 40 metrics
   Extensions: ['manylatents-omics']
   Sources: {manylatents: 26, manylatents-omics: 14}
```

### Registry Metadata

The generated registry includes extension information:
```json
{
  "_metadata": {
    "manylatents_version": "0.1.0",
    "metrics_scanned": 40,
    "extensions_found": 1,
    "extensions": ["manylatents-omics"],
    "sources": {
      "manylatents": 26,
      "manylatents-omics": 14
    }
  }
}
```

### Metric Source Tracking

Each metric tracks its source:
```python
info = registry.get_metric_info('admixture_preservation')
print(info['source'])  # 'manylatents-omics'
```

### Override Behavior

If an extension defines a metric with the same name as a core metric:
- The extension metric **overrides** the core metric
- A warning is logged during generation
- The registry tracks which source won

## File Location

**Generated file**: `manyagents/adapters/data/metric_registry.json`

**Note**: This file is `.gitignore`d as it's auto-generated.

## Version-Based Regeneration

The registry only regenerates when the manyLatents version changes:

```bash
# First run - generates registry
manyagents-generate-registry
# ✅ Generated registry with 26 metrics (manylatents v0.1.0)

# Second run - skips (version unchanged)
manyagents-generate-registry
# ℹ️ Metric registry up-to-date (manylatents v0.1.0). Skipping regeneration.

# After manylatents upgrade - regenerates automatically
pip install --upgrade manylatents
manyagents-generate-registry
# ✅ Generated registry with 30 metrics (manylatents v0.2.0)
```

Force regeneration:
```bash
manyagents-generate-registry --force
```

## Testing

Run registry tests:
```bash
pytest tests/test_metric_registry.py -v
```

All tests should pass:
```
18 passed in ~12s
```
