# manyLatents Key Files Reference Guide

Complete file paths and their roles in the architecture.

## Core Algorithm Infrastructure

### Base Classes
- **`/manylatents/algorithms/latent_module_base.py`**
  - `class LatentModule(ABC)` - Abstract base for all dimensionality reduction algorithms
  - Defines `fit(x)`, `transform(x)`, `fit_transform(x)` interface
  - Statefulness tracking via `_is_fitted` flag

### Latent Algorithms (fit/transform pattern)
- **`/manylatents/algorithms/latent/pca.py`** - PCA wrapper with fit_fraction support
- **`/manylatents/algorithms/latent/phate.py`** - PHATE manifold learning with affinity/kernel matrices
- **`/manylatents/algorithms/latent/umap.py`** - UMAP dimensionality reduction
- **`/manylatents/algorithms/latent/tsne.py`** - t-SNE with early/late exaggeration
- **`/manylatents/algorithms/latent/mds.py`** - Multidimensional scaling
- **`/manylatents/algorithms/latent/diffusionmap.py`** - Diffusion map embedding
- **`/manylatents/algorithms/latent/aa.py`** - Adversarial autoencoder
- **`/manylatents/algorithms/latent/dr_noop.py`** - No-op (identity) algorithm

### Lightning-based Algorithms (neural networks)
- **`/manylatents/algorithms/lightning/reconstruction.py`**
  - `class Reconstruction(LightningModule)` - Autoencoder/reconstruction-based learning
  - Deferred initialization of `input_dim` from first batch
  - Methods: `setup()`, `configure_model()`, `forward()`, `encode()`, `training_step()`, `validation_step()`, `test_step()`, `configure_optimizers()`
  
#### Lightning Networks
- **`/manylatents/algorithms/lightning/networks/autoencoder.py`** - Standard autoencoder with encoder/decoder
- **`/manylatents/algorithms/lightning/networks/aanet.py`** - Adversarial autoencoder variant

#### Lightning Loss Functions
- **`/manylatents/algorithms/lightning/losses/loss.py`** - Base loss class
- **`/manylatents/algorithms/lightning/losses/mse.py`** - Mean squared error loss with component tracking
- **`/manylatents/algorithms/lightning/losses/geometric.py`** - Geometric losses (PR, Anisotropy, TSA, AllGeom)

## Orchestration & Execution

### Core Experiment Engine
- **`/manylatents/experiment.py`** (22KB) - Main orchestration logic
  - `instantiate_datamodule()` - Create data loaders from config
  - `instantiate_algorithm()` - Instantiate algorithm with partial support
  - `instantiate_callbacks()` - Create Lightning callbacks
  - `instantiate_trainer()` - Create Lightning trainer
  - `execute_step()` - Core execution for single algorithm (30+ lines)
  - `evaluate()` - Singledispatch evaluator routing on type
  - `evaluate_embeddings()` - Registered handler for LatentModule outputs
  - `evaluate_lightningmodule()` - Registered handler for LightningModule
  - `run_algorithm()` - Single algorithm execution
  - `run_pipeline()` - Sequential multi-step pipeline with state threading

### Programmatic API
- **`/manylatents/api.py`** (173 lines) - Clean Python interface
  - `run()` - Main entry point with smart routing to `run_algorithm()` or `run_pipeline()`
  - Supports: `input_data`, arbitrary Hydra overrides, pipeline execution
  - Handles: OmegaConf struct mode, numpy array serialization workarounds

## Configuration System

### Base Configuration
- **`/manylatents/configs/config.py`** - Dataclass defining entire config schema
  - `@dataclass Config` with 13 fields (algorithms, pipeline, data, callbacks, trainer, metrics, etc.)
  
- **`/manylatents/configs/__init__.py`** - ConfigStore registration
  - Registers `Config` dataclass as `base_config` for Hydra

### Main Config File
- **`/manylatents/configs/config.yaml`**
  - Hydra defaults composition
  - References: base_config, data, callbacks, algorithms/latent, algorithms/lightning, trainer, metrics, etc.

### Algorithm Configs (Hydra)
- **Latent Algorithms**:
  - `/manylatents/configs/algorithms/latent/pca.yaml` - n_components, random_state
  - `/manylatents/configs/algorithms/latent/phate.yaml` - n_components, knn, t, decay, gamma, etc.
  - `/manylatents/configs/algorithms/latent/tsne.yaml` - All t-SNE hyperparameters
  - `/manylatents/configs/algorithms/latent/umap.yaml` - UMAP parameters
  - `/manylatents/configs/algorithms/latent/mds.yaml` - MDS parameters
  - `/manylatents/configs/algorithms/latent/diffusionmap.yaml` - Diffusion map parameters

- **Lightning Algorithms**:
  - `/manylatents/configs/algorithms/lightning/ae_reconstruction.yaml` - Standard autoencoder
  - `/manylatents/configs/algorithms/lightning/ae_reconstruction_aniso.yaml` - With anisotropy loss
  - `/manylatents/configs/algorithms/lightning/ae_reconstruction_allgeom.yaml` - With all geometric losses
  - `/manylatents/configs/algorithms/lightning/aanet_reconstruction.yaml` - Adversarial AE variant

### Trainer & Infrastructure Configs
- **`/manylatents/configs/trainer/default.yaml`**
  - Lightning Trainer instantiation: accelerator, devices, max_epochs, precision, callbacks, loggers
  - Defaults: auto accelerator, 150 epochs, fp32 precision

- **`/manylatents/configs/trainer/callbacks/`** - Callback configurations
- **`/manylatents/configs/trainer/logger/wandb.yaml`** - WandB logger config

### Metrics Configs
- **`/manylatents/configs/metrics/dataset/`** - Dataset-level metrics (variance, dimensionality)
- **`/manylatents/configs/metrics/embedding/`** - Embedding-level metrics (trustworthiness, continuity)
- **`/manylatents/configs/metrics/module/`** - Module-level metrics (reconstruction error, test loss)

### Experiment Presets
- **`/manylatents/configs/experiment/eval_algorithm.yaml`** - Eval-only mode preset
- **`/manylatents/configs/experiment/pipeline_step_lightning.yaml`** - Pipeline step with LightningModule
- **`/manylatents/configs/experiment/pca_phate_pipeline.yaml`** - Complete PCA→PHATE pipeline example

## Data & Utilities

### Data Module
- **`/manylatents/data/precomputed_datamodule.py`** - Wraps pre-computed numpy arrays as Lightning DataModule
  - Used by `api.run(input_data=...)` for chaining

### Utilities
- **`/manylatents/utils/data.py`** - Data loading, subsampling, dataset utilities
- **`/manylatents/utils/metrics.py`** - Metric flattening/unrolling
- **`/manylatents/utils/utils.py`** - Logging setup, directory management

### Callbacks
- **`/manylatents/callbacks/embedding/base.py`** - Base EmbeddingCallback class
  - Hook: `on_latent_end(dataset, embeddings)` called after embeddings computed

## Testing Infrastructure

- **`/manylatents/algorithms/lightning/losses/test_losses.py`** - Loss function tests
- **`/manylatents/algorithms/lightning/networks/test_networks.py`** - Network architecture tests

---

## Key Design Patterns to Extract

### 1. Config-Driven Instantiation
```
Config YAML → hydra.utils.instantiate() → Algorithm Object
```
All algorithm hyperparameters come from Hydra configs, enabling:
- Zero-code parameter tuning
- Dynamic config generation (perfect for RL agents)

### 2. Singledispatch Evaluation
```python
@functools.singledispatch
def evaluate(algorithm: Any) -> dict
    # Routes on algorithm type

@evaluate.register(dict)  # LatentModule output
def evaluate_embeddings(...)

@evaluate.register(LightningModule)
def evaluate_lightningmodule(...)
```
Same evaluation interface for different algorithm types.

### 3. State Threading in Pipelines
```
Step 1 embeddings → Step 2 input_data
Step 2 embeddings → Step 3 input_data
...
```
In-memory state passing enables sequential workflow composition.

### 4. Deferred Initialization
```python
def setup(self, stage=None):
    if input_dim is None:
        first_batch = next(iter(datamodule.train_dataloader()))
        input_dim = first_batch.shape[1]
        self.instantiate_with(input_dim)
```
Infer dimension from first batch, then build architecture.

### 5. Programmatic API Layer
```python
def run(input_data=None, **overrides):
    # Compose config from overrides
    # Smart route to run_algorithm() or run_pipeline()
    # Handle input_data side-channel
```
Clean Python interface wrapping Hydra complexity.

---

## Critical Entry Points

### For Orchestration (manyAgents Integration)
1. `manylatents.api.run()` - Main entry point
   - Returns: `{'embeddings': ndarray, 'label': ..., 'metadata': {...}, 'scores': {...}}`
   - Supports: single algorithms, pipelines, chaining via input_data

2. `manylatents.experiment.run_algorithm()` - Single algorithm execution
   - Called by `api.run()` internally
   - Handles: datamodule setup, algorithm instantiation, training, evaluation

3. `manylatents.experiment.run_pipeline()` - Sequential pipeline execution
   - Called by `api.run()` internally
   - Handles: state threading between steps, callbacks

### For Metrics & Evaluation
1. `manylatents.experiment.evaluate()` - Singledispatch evaluator
   - Computes metrics from algorithm output
   - Routes on algorithm type automatically

2. `manylatents.utils.metrics.flatten_and_unroll_metrics()` - Metric config processing
   - Converts hierarchical metric configs to flat dict

### For Configuration
1. `manylatents.configs.Config` - Type-safe config schema
2. Hydra compose + initialize for programmatic config building

---

## Total Lines of Code (Approximate)
- Core orchestration: 500 lines (experiment.py)
- API layer: 180 lines (api.py)
- Base classes: 30 lines (latent_module_base.py)
- Algorithm implementations: 1000+ lines (across all latent/ and lightning/)
- Configuration: 100+ YAML files, ~50 lines Python (config.py)
- Utilities: 500+ lines (data, metrics, logging)

**Total: ~3000 lines of core infrastructure**

