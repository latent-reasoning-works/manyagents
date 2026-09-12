# Metric registry

`MetricRegistry()` discovers installed manyLatents YAML metric and algorithm
configs in memory. It requires the `traces` extra at use time and does not write
into the installed package. Wheels build without manyLatents or a registry file.

```python
from manyagents.adapters.metric_registry import MetricRegistry

registry = MetricRegistry()
print(registry.list_metrics(group="embedding"))
print(registry.get_metric_info("participation_ratio"))
metric_class = registry.get_metric_class("participation_ratio")
```

Entries record the class, default parameters, group, and source package. Installed
manyLatents extensions with config directories are discovered too; extension
entries override core names with a warning.

The CLI prints JSON to stdout by default. `--verbose` adds diagnostic logging.
Only an explicit `--output` creates a persistent cache:

```bash
manyagents-generate-registry --verbose
manyagents-generate-registry --output /tmp/manyagents-registry.json
manyagents-generate-registry --output /tmp/manyagents-registry.json --force
```

An explicit cache may be reused when the manyLatents version is unchanged;
`--force` regenerates it. `MetricRegistry(Path("/tmp/manyagents-registry.json"))`
uses that explicit path (`Path` is from `pathlib`). Cache write failure is logged
and the generated in-memory registry remains usable.

**Trajectory metrics:** manylatents 0.1.7 has no YAML configs for
`trajectory_velocity` or `trajectory_curvature`, so they are absent here. Use
manyLatents' direct metric API and select a layer from 3-D stored tensors first.
See the [runnable trace-to-geometry example](../../README.md#from-traces-to-geometry).
