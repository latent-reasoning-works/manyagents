# manyLatents Architecture & Training Infrastructure Analysis

## 1. Algorithm Base Class Structure

### LatentModule Base Class
**File**: `/manylatents/algorithms/latent_module_base.py`

```python
class LatentModule(ABC):
    def __init__(self, n_components: int = 2, init_seed: int = 42, **kwargs):
        self.n_components = n_components
        self.init_seed = init_seed
        self.datamodule = kwargs.pop('datamodule', None)
        self._is_fitted = False

    @abstractmethod
    def fit(self, x: Tensor) -> None:
        pass

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        pass

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)
```

**Key Characteristics**:
- **Pattern**: sklearn-style `fit/transform` interface
- **Statefulness**: `_is_fitted` flag tracks fitting status
- **Flexibility**: Accepts arbitrary kwargs for extensibility
- **Data Types**: Uses PyTorch Tensors (supports GPU via `.to(device)`)
- **Device Handling**: Automatically converts to/from CPU via `.detach().cpu().numpy()`

### Concrete Implementations

**PCAModule** (`algorithms/latent/pca.py`):
```python
class PCAModule(LatentModule):
    def __init__(self, n_components: int = 2, random_state: int = 42, 
                 fit_fraction: float = 1.0, **kwargs):
        super().__init__(n_components=n_components, init_seed=random_state, **kwargs)
        self.fit_fraction = fit_fraction
        self.model = PCA(n_components=n_components, random_state=random_state)

    def fit(self, x: Tensor) -> None:
        x_np = x.detach().cpu().numpy()
        n_fit = max(1, int(self.fit_fraction * len(x_np)))
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("Not fitted yet")
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)
```

**Pattern Features**:
- Wraps sklearn algorithms for PyTorch compatibility
- Supports `fit_fraction` for subsampling large datasets
- Preserves tensor device/dtype through numpy intermediary

---

## 2. LightningModule Integration Pattern

**File**: `/manylatents/algorithms/lightning/reconstruction.py`

```python
class Reconstruction(LightningModule):
    def __init__(self, datamodule, network: DictConfig, loss: DictConfig,
                 optimizer: DictConfig, init_seed: int = 42):
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.loss_config = loss
        self.init_seed = init_seed
        self.save_hyperparameters(ignore=["datamodule"])
        self.network: nn.Module | None = None

    def setup(self, stage=None):
        # Infer input_dim from first batch if not provided
        if self.network_config.input_dim is None:
            first_batch = next(iter(self.datamodule.train_dataloader()))["data"]
            feat_dim = first_batch.shape[1]
            self.network_config.input_dim = feat_dim
        self.configure_model()

    def configure_model(self):
        torch.manual_seed(self.init_seed)
        # Instantiate network, loss, optimizer from configs
        self.network = hydra_zen.instantiate(self.network_config)
        self.loss_fn = hydra_zen.instantiate(self.loss_config)
        self._optimizer_partial = self.optimizer_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation from encoder"""
        return self.network.encode(x)

    def training_step(self, batch, batch_idx):
        x = batch["data"]
        outputs = self.network(x)
        loss = self.loss_fn(outputs=outputs, targets=x, latent=self.network.encoder(x), raw=x)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x = batch["data"]
        outputs = self.network(x)
        loss = self.loss_fn(outputs=outputs, targets=x, ...)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x = batch["data"]
        outputs = self.network(x)
        loss = self.loss_fn(outputs=outputs, targets=x, ...)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        return optimizer_partial(self.parameters())
```

**Key Patterns**:
- **Deferred Initialization**: Infers `input_dim` from first batch in `setup()`
- **Config-Driven**: Network, loss, optimizer instantiated from Hydra configs
- **Modular Architecture**: Network has `.encode()` method for latent extraction
- **Loss Flexibility**: Losses can have `.components()` for detailed logging

---

## 3. Training Loop Infrastructure

### Core Execution Engine
**File**: `/manylatents/experiment.py`

#### `execute_step()` Function
```python
def execute_step(algorithm, train_tensor, test_tensor, trainer, cfg, datamodule):
    """Core execution engine for a single algorithm step."""
    latents = None

    if isinstance(algorithm, LatentModule):
        # Fit/Transform pattern
        algorithm.fit(train_tensor)
        latents = algorithm.transform(test_tensor)
    
    elif isinstance(algorithm, LightningModule):
        # LightningModule: training + evaluation
        if cfg.eval_only and cfg.pretrained_ckpt:
            algorithm = LightningModule.load_from_checkpoint(cfg.pretrained_ckpt)
        
        if not cfg.eval_only:
            trainer.fit(algorithm, datamodule=datamodule)
        
        # Evaluate & extract embeddings
        evaluation_result = evaluate(algorithm, cfg=cfg, trainer=trainer, datamodule=datamodule)
        if hasattr(algorithm, "encode"):
            latents = algorithm.encode(test_tensor).detach().cpu().numpy()
    
    return latents
```

#### Dual Dispatch Pattern
```python
@functools.singledispatch
def evaluate(algorithm: Any, /, **kwargs) -> Tuple[str, Optional[float], dict]:
    """Singledispatch evaluator - routes on algorithm type"""
    raise NotImplementedError(...)

@evaluate.register(dict)
def evaluate_embeddings(EmbeddingOutputs: dict, *, cfg, datamodule, **kwargs):
    """Evaluate LatentModule embeddings (dict output)"""
    embeddings = EmbeddingOutputs.get("embeddings")
    # Compute metrics from cfg.metrics configuration
    ...

@evaluate.register(LightningModule)
def evaluate_lightningmodule(algorithm, *, cfg, trainer, datamodule, **kwargs):
    """Evaluate LightningModule using trainer.test()"""
    results = trainer.test(model=algorithm, datamodule=datamodule)
    # Extract metrics from test results
    ...
```

#### Pipeline Orchestration
```python
def run_pipeline(cfg, input_data_holder=None):
    """Sequential multi-step pipeline execution"""
    with wandb.init(project=cfg.project, name=cfg.name, ...):
        initial_datamodule = instantiate_datamodule(cfg, input_data_holder)
        
        # One-time setup
        trainer_cbs, embedding_cbs = instantiate_callbacks(...)
        trainer = instantiate_trainer(cfg, lightning_callbacks=trainer_cbs, ...)
        
        # Sequential execution with state threading
        current_embeddings = None
        for step_cfg in cfg.pipeline:
            # For each step after the first, use embeddings as input_data
            if step_cfg.step_index > 0:
                input_data_holder = {'data': current_embeddings['embeddings']}
            
            # Execute step with dynamically updated datamodule
            step_result = run_algorithm(step_cfg, input_data_holder=input_data_holder)
            current_embeddings = step_result
            
            # Logging and callbacks
            for cb in embedding_cbs:
                cb.on_latent_end(dataset=..., embeddings=current_embeddings)
        
        return current_embeddings
```

---

## 4. Hydra Config → Algorithm Instantiation

### Config Structure

**Base Config** (`configs/config.py`):
```python
@dataclass
class Config:
    algorithms: Optional[Dict[str, Any]] = None  # latent or lightning
    pipeline: List[Any] = field(default_factory=list)
    data: Optional[Any] = None
    callbacks: Optional[Any] = None
    trainer: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[Any] = None
    seed: int = field(default_factory=lambda: random.randint(0, int(1e5)))
    debug: bool = False
    eval_only: bool = False
    pretrained_ckpt: Optional[str] = None
```

### Algorithm Instantiation

**File**: `experiment.py`

```python
def instantiate_algorithm(algorithm_config, datamodule=None):
    """Instantiate algorithm from Hydra config"""
    algo_or_partial = hydra.utils.instantiate(algorithm_config, datamodule=datamodule)
    if isinstance(algo_or_partial, functools.partial):
        return algo_or_partial()
    return algo_or_partial

# Called as:
if hasattr(cfg.algorithms, 'latent') and cfg.algorithms.latent is not None:
    algorithm = instantiate_algorithm(cfg.algorithms.latent, datamodule)
elif hasattr(cfg.algorithms, 'lightning') and cfg.algorithms.lightning is not None:
    algorithm = instantiate_algorithm(cfg.algorithms.lightning, datamodule)
```

### Config Examples

**PCA Algorithm Config** (`configs/algorithms/latent/pca.yaml`):
```yaml
_target_: manylatents.algorithms.latent.pca.PCAModule
n_components: 2
random_state: ${seed}
```

**Autoencoder Config** (`configs/algorithms/lightning/ae_reconstruction.yaml`):
```yaml
_target_: manylatents.algorithms.lightning.reconstruction.Reconstruction
datamodule: ${data}
network:
  _target_: manylatents.algorithms.lightning.networks.autoencoder.Autoencoder
  input_dim: null  # Inferred from first batch
  hidden_dims: [512, 256, 128]
  latent_dim: 50
  activation: relu
optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
loss:
  _target_: manylatents.algorithms.lightning.losses.mse.MSELoss
```

**Trainer Config** (`configs/trainer/default.yaml`):
```yaml
defaults:
  - logger: wandb
  - callbacks: default

_target_: lightning.Trainer
accelerator: auto
strategy: auto
devices: 1
deterministic: true
min_epochs: 5
max_epochs: 150
precision: 32
val_check_interval: 1.0
check_val_every_n_epoch: 1
gradient_clip_val: 1.0
```

---

## 5. Programmatic API

**File**: `/manylatents/api.py`

```python
def run(input_data: Optional[np.ndarray] = None, **overrides) -> Dict[str, Any]:
    """
    Programmatic entry point for manyLatents.
    
    Args:
        input_data: Optional input tensor (chains algorithms)
        **overrides: Hydra config overrides
    
    Returns:
        {'embeddings': ndarray, 'label': ..., 'metadata': {...}, 'scores': {...}}
    """
    with initialize_config_dir(config_dir=..., version_base=None):
        cfg = compose(config_name="config", overrides=[...])
        OmegaConf.set_struct(cfg, False)  # Allow arbitrary fields
        
        # Handle input_data
        input_data_holder = {'data': input_data} if input_data else None
        
        # Smart routing
        is_pipeline = hasattr(cfg, 'pipeline') and cfg.pipeline
        if is_pipeline:
            return run_pipeline(cfg, input_data_holder=input_data_holder)
        else:
            return run_algorithm(cfg, input_data_holder=input_data_holder)

# Usage Examples:
result1 = run(data='swissroll', algorithms={'latent': {'_target_': '...PCA', 'n_components': 50}})
result2 = run(input_data=result1['embeddings'], algorithms={'latent': {'_target_': '...PHATE'}})
result3 = run(pipeline=[
    {'name': 'pca_step', 'overrides': {'algorithms': {'latent': 'pca'}}},
    {'name': 'phate_step', 'overrides': {'algorithms': {'latent': 'phate'}}}
])
```

---

## 6. Metrics & Evaluation Infrastructure

### Metric Computation
```python
@evaluate.register(dict)
def evaluate_embeddings(EmbeddingOutputs, *, cfg, datamodule, **kwargs):
    embeddings = EmbeddingOutputs.get("embeddings")
    
    # Flatten and unroll metrics config
    metric_cfgs = flatten_and_unroll_metrics(cfg.metrics)
    
    results: dict[str, float] = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metric_fn = hydra.utils.instantiate(metric_cfg)
        results[metric_name] = metric_fn(
            embeddings=embeddings,
            dataset=dataset,
            module=module
        )
    return results
```

### Three-Level Metrics System
1. **Dataset-Level**: Global properties (e.g., PCA variance)
2. **Embedding-Level**: Embedding quality (e.g., trustworthiness, continuity)
3. **Module-Level**: Model-specific metrics (e.g., reconstruction error)

---

## 7. Key Design Patterns for Phase 3 (RL/Learning) Implications

### Pattern 1: Modular Algorithm Interface
- **Implication**: New RL agent types can implement `LatentModule` or `LightningModule` interface
- **Advantage**: Direct integration with existing orchestration engine

### Pattern 2: Config-Driven Instantiation
- **Implication**: Hyperparameter tuning can be done via Hydra config overrides
- **Advantage**: No code changes needed for parameter sweeps; config becomes RL state

### Pattern 3: Programmatic API with State Threading
- **Implication**: Perfect for agent decision loops - each decision produces config, runs algorithm, gets result
- **Advantage**: Enables sequential agent decisions with in-memory state passing

### Pattern 4: Dual Dispatch Evaluation
- **Implication**: Same evaluation framework for all algorithm types
- **Advantage**: Reward signals can be computed uniformly across diverse tools

### Pattern 5: Stateful Training Loop
- **Implication**: Algorithms maintain `_is_fitted` state, can be checkpointed
- **Advantage**: Enables experience replay, rollback, and training resumption

---

## 8. Existing Infrastructure Not Yet Used for RL

### Available but Unused:
1. **Callbacks System**: Could track agent actions and reward signals
2. **Metrics Config**: Extensible framework for reward computation
3. **Checkpointing**: Lightning checkpoints can store agent state
4. **Pipeline Logging**: Could record trajectory for learning

### Gaps for Phase 3:
1. **No explicit reward function**: `evaluate()` returns metrics, not normalized rewards
2. **No agent policy**: All algorithms are deterministic, no decision-making
3. **No experience replay**: No mechanism to store/replay trajectories
4. **No hyperparameter adaptation**: Config changes require full re-runs

---

## 9. Proposed RL Integration Points

### Minimal Changes for RL:

1. **Reward Function Wrapper**:
   ```python
   class RewardFunction:
       def compute(self, metrics: dict, metadata: dict) -> float:
           """Normalize manylatents metrics into scalar reward"""
           pass
   ```

2. **Agent Decision Module**:
   ```python
   class AgentPlanner:
       def decide(self, current_state: dict, available_actions: list) -> dict:
           """Return algorithm config to execute next"""
           pass
   ```

3. **Experience Storage**:
   ```python
   class Trajectory:
       def record(self, state, action, reward, next_state):
           """Store (s,a,r,s') transition"""
           pass
   ```

4. **Learning Loop**:
   ```python
   def rl_orchestration_loop(env, agent, episodes=100):
       for episode in range(episodes):
           state = env.reset()
           for step in range(max_steps):
               action = agent.decide(state)
               result = run(algorithm=action['algorithm'], input_data=state['embeddings'])
               reward = compute_reward(result)
               next_state = {'embeddings': result['embeddings'], 'metrics': result['scores']}
               agent.record(state, action, reward, next_state)
               state = next_state
   ```

---

## Summary: Key Takeaways for Phase 3

| Aspect | Current Implementation | Phase 3 Enablement |
|--------|----------------------|-------------------|
| **Algorithm Interface** | Abstract `fit/transform` or `LightningModule` | Extend with `get_action_space()` |
| **Config System** | Hydra-driven, static at runtime | Add dynamic config generation from RL policy |
| **Execution Model** | Deterministic pipeline | Add decision points for agent selection |
| **Evaluation** | Metrics computed per-algorithm | Normalize to rewards with user-defined function |
| **State Management** | In-memory embeddings + metrics | Add experience buffer for replay |
| **Checkpointing** | Per-algorithm checkpoints | Track full trajectory + policy parameters |

