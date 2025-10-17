# Geomancer Architecture: RL Agent for Geometric Workflow Learning

**Date**: 2025-10-16
**Vision**: Implement the Geomancer paper's core idea - RL agent learns to replicate expert workflows by matching geometric signatures

---

## Two Visions, One Framework

### Vision A: Multi-Adapter Orchestration (CURRENT - Phase 3)
**Purpose**: Compare different analysis tools/methods in parallel

**Workflow**:
```
1. Execute Anchor workflow (e.g., manylatents PCA) → G_anchor
2. Execute Comparison workflow (e.g., STELLA) → G_comparison
3. Compute reward = G_comparison - G_anchor
4. Use for method comparison / benchmarking
```

**Implementation**: ✅ Complete
- `phase3_anchor_vs_comparison.yaml`
- `ManyLatentsAdapter`, adapters for external tools
- `MetricComputer` for external embeddings
- `EvaluationPipeline` for comparison

---

### Vision B: Geomancer RL Loop (GOAL)
**Purpose**: Agent learns to replicate expert workflow by matching geometric signature

**Workflow**:
```
1. Expert defines reference workflow → Execute → G_target (the goal)
2. RL agent starts with raw data → G_0
3. Loop (episode):
   - Agent observes G_t (current geometric state)
   - Agent selects action A_t from toolkit (PCA, UMAP, normalize, filter, STELLA, etc.)
   - Execute action → new data state
   - Compute G_{t+1}
   - Reward = -||G_{t+1} - G_target||²
   - Update policy via PPO/GRPO
4. Agent learns sequence of actions that replicate expert geometry
```

**Implementation**: 🚧 To be built

---

## Key Architectural Differences

| Aspect | Vision A: Multi-Adapter | Vision B: Geomancer RL |
|--------|-------------------------|------------------------|
| **Execution** | Parallel workflows | Sequential steps |
| **Control** | Static config (YAML) | RL policy (neural network) |
| **Timing** | One-shot comparison | Episodic learning loop |
| **State** | Final embeddings only | Geometric signature G_t at each step |
| **Reward** | End result difference | Step-by-step deviation from G_target |
| **Goal** | "Which method is better?" | "How to replicate expert workflow?" |
| **Agent Role** | Tool executor | Active learner |
| **Output** | Comparison report | Learned policy |

---

## Common Infrastructure

Both visions share critical components:

### 1. **Geometric Metrics (manylatents.metrics.api)**
- Compute metric vectors G from any data state
- Fast, in-memory, Hydra-free
- Dynamic discovery of all metrics

### 2. **Adapter Abstraction (manyagents.adapters.AgentAdapter)**
- Unified interface for all tools
- Input: config + data state
- Output: transformed data + metadata

### 3. **Metric Computation (manyagents.metrics.MetricComputer)**
- Compute G for non-manylatents outputs
- Used in both visions

### 4. **Reward Calculation**
- Both compute metric vector differences
- Vision A: `reward = G_comparison - G_anchor`
- Vision B: `reward = -||G_{t+1} - G_target||²`

### 5. **RL Infrastructure (manyagents.models)**
- `RLModule` abstract base
- `SB3Module` implementation
- Ready for both use cases

---

## The "Multi-Agent Communication" Element

Both visions need **orchestration of multiple tools**:

### Vision A: Static Multi-Tool Coordination
```python
# Pre-defined workflow config
workflow = [
    {'agent': 'manylatents', 'action': 'pca', 'config': {...}},
    {'agent': 'stella', 'action': 'design_method', 'config': {...}}
]

# Orchestrator executes sequentially or in parallel
orchestrator.execute_workflow(workflow)
```

### Vision B: Dynamic RL-Driven Coordination
```python
# RL agent decides which tool to use at each step
for t in range(T):
    G_t = compute_metrics(current_state)

    # Policy selects from action space
    action_name = policy.predict(G_t)  # e.g., 'run_pca', 'normalize', 'stella_design'

    # Orchestrator routes to correct adapter
    adapter = action_registry[action_name]
    result = adapter.run(config, current_state)

    # Update state and reward
    current_state = result['data']
    G_t1 = compute_metrics(current_state)
    reward = -np.linalg.norm(G_t1 - G_target)**2

    # Learn
    policy.update(G_t, action_name, reward, G_t1)
```

**Common Need**: **Unified Action Space** where each adapter method is a discrete action.

---

## Unified Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                 manyAgents Orchestrator                      │
│                                                               │
│  ┌──────────────────────┐      ┌────────────────────────┐  │
│  │ Mode A:              │      │ Mode B:                │  │
│  │ StaticWorkflow       │      │ GeomancerRL            │  │
│  │ Executor             │      │ Engine                 │  │
│  │ (Phase 3 Current)    │      │ (Geomancer Vision)     │  │
│  └──────────┬───────────┘      └───────────┬────────────┘  │
│             │                               │                │
│             └───────────┬───────────────────┘                │
│                         │                                    │
│              ┌──────────▼──────────────┐                    │
│              │   Action Registry       │                    │
│              │   (Unified Adapter      │                    │
│              │    Interface)           │                    │
│              │                         │                    │
│              │  Actions:               │                    │
│              │  - run_pca              │                    │
│              │  - run_umap             │                    │
│              │  - normalize_data       │                    │
│              │  - filter_hvg           │                    │
│              │  - stella_design        │                    │
│              │  - cellforge_generate   │                    │
│              └──────────┬──────────────┘                    │
│                         │                                    │
│          ┌──────────────┼──────────────┐                    │
│          │              │               │                    │
│     ┌────▼────┐   ┌────▼────┐    ┌────▼────┐              │
│     │manyLA   │   │ STELLA  │    │CellForge│              │
│     │tents    │   │ Adapter │    │ Adapter │              │
│     │Adapter  │   └─────────┘    └─────────┘              │
│     │         │                                             │
│     │ Actions:│                                             │
│     │ - pca   │                                             │
│     │ - umap  │                                             │
│     │ - norm  │                                             │
│     │ - filter│                                             │
│     └────┬────┘                                             │
│          │                                                   │
└──────────┼───────────────────────────────────────────────────┘
           │
    ┌──────▼───────────────────────────┐
    │  Geometric Metrics Computation    │
    │  (manylatents.metrics.api)        │
    │                                   │
    │  G = [Trustworthiness,            │
    │       Continuity,                 │
    │       Persistence,                │
    │       Spectral Entropy, ...]      │
    └───────────────────────────────────┘
```

---

## Geomancer Implementation Plan

### Step 1: Define Expert Workflow and Target Signature
```yaml
# configs/geomancer/expert_workflow.yaml
name: expert_seurat_pipeline

workflow:
  steps:
    - action: filter_cells
      config: {min_genes: 200, max_genes: 2500}

    - action: normalize_data
      config: {method: 'log_normalize', scale_factor: 10000}

    - action: find_hvg
      config: {n_top_genes: 2000}

    - action: scale_data
      config: {}

    - action: run_pca
      config: {n_components: 50}

    - action: leiden_cluster
      config: {resolution: 0.5}

    - action: run_umap
      config: {n_neighbors: 15, min_dist: 0.1}

# Execute this to get G_target
target:
  compute_at_step: -1  # Final step
  metrics: [Trustworthiness, Continuity, H0Persistence, DiffusionCurvature, SpectralEntropy]
```

Execute once:
```bash
python -m manyagents.geomancer.compute_target \
    workflow=expert_seurat_pipeline \
    dataset=embryoid_body
```

Output: `G_target.npy` (the goal vector)

---

### Step 2: Define Action Space
```python
# manyagents/geomancer/action_space.py

from typing import Dict, Callable
from manyagents.adapters import ManyLatentsAdapter, STELLAAdapter, CellForgeAdapter

class ActionSpace:
    """
    Unified action space for Geomancer RL agent.

    Each action is a tuple: (adapter, method, default_config)
    """

    def __init__(self):
        self.actions = {}
        self._register_actions()

    def _register_actions(self):
        # manylatents actions
        self.actions['run_pca'] = {
            'adapter': ManyLatentsAdapter(),
            'method': 'pca',
            'config_template': {'n_components': 50}
        }

        self.actions['run_umap'] = {
            'adapter': ManyLatentsAdapter(),
            'method': 'umap',
            'config_template': {'n_neighbors': 15, 'min_dist': 0.1}
        }

        self.actions['normalize_data'] = {
            'adapter': ManyLatentsAdapter(),
            'method': 'normalize',
            'config_template': {'method': 'log_normalize'}
        }

        # External tool actions
        self.actions['stella_design'] = {
            'adapter': STELLAAdapter(),
            'method': 'design_method',
            'config_template': {'prompt': 'Design DR for this data'}
        }

        self.actions['cellforge_generate'] = {
            'adapter': CellForgeAdapter(),
            'method': 'generate',
            'config_template': {'phase': 'full'}
        }

    def get_action(self, action_name: str) -> Dict:
        return self.actions[action_name]

    def list_actions(self) -> list:
        return list(self.actions.keys())
```

---

### Step 3: Geomancer Environment (Gym API)
```python
# manyagents/geomancer/environment.py

import gym
import numpy as np
from manylatents.metrics.api import compute_metrics
from manyagents.geomancer.action_space import ActionSpace

class GeomancerEnv(gym.Env):
    """
    Geomancer RL environment.

    State: Geometric signature G_t (vector of metric values)
    Action: Discrete choice from action space (tool + method)
    Reward: -||G_{t+1} - G_target||²
    """

    def __init__(
        self,
        dataset: np.ndarray,
        G_target: np.ndarray,
        metric_names: list,
        max_steps: int = 10
    ):
        super().__init__()

        self.dataset_original = dataset  # Keep original for metrics
        self.current_data = dataset.copy()  # Working copy
        self.G_target = G_target
        self.metric_names = metric_names
        self.max_steps = max_steps
        self.current_step = 0

        # Action space
        self.action_space_obj = ActionSpace()
        self.action_list = self.action_space_obj.list_actions()
        self.action_space = gym.spaces.Discrete(len(self.action_list))

        # Observation space (geometric signature)
        n_metrics = len(metric_names)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_metrics,),
            dtype=np.float32
        )

    def reset(self):
        """Reset to initial data state."""
        self.current_data = self.dataset_original.copy()
        self.current_step = 0

        # Compute initial geometric signature
        G_0 = self._compute_current_signature()
        return G_0

    def step(self, action_idx: int):
        """Execute action and return (next_state, reward, done, info)."""
        # Get action details
        action_name = self.action_list[action_idx]
        action_spec = self.action_space_obj.get_action(action_name)

        # Execute action via adapter
        adapter = action_spec['adapter']
        config = action_spec['config_template'].copy()
        config['input_data'] = self.current_data

        result = adapter.run(config, input_files={})

        # Update data state
        if 'embeddings' in result:
            self.current_data = result['embeddings']
        elif 'data' in result:
            self.current_data = result['data']
        else:
            raise ValueError(f"Action {action_name} returned no data/embeddings")

        # Compute new geometric signature
        G_t1 = self._compute_current_signature()

        # Compute reward (negative distance to target)
        distance = np.linalg.norm(G_t1 - self.G_target)
        reward = -distance ** 2

        # Check termination
        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        # Info
        info = {
            'action_name': action_name,
            'distance_to_target': distance,
            'G_current': G_t1,
            'step': self.current_step
        }

        return G_t1, reward, done, info

    def _compute_current_signature(self) -> np.ndarray:
        """Compute geometric metrics on current data state."""
        # Use manylatents.metrics.api
        metrics_dict = compute_metrics(
            x=self.dataset_original,  # Always compare to original
            z=self.current_data,
            metric_names=self.metric_names
        )

        # Convert to vector
        G = np.array([metrics_dict[name] for name in self.metric_names])
        return G.astype(np.float32)
```

---

### Step 4: Training Loop
```python
# manyagents/geomancer/train.py

from stable_baselines3 import PPO
from manyagents.geomancer.environment import GeomancerEnv
import numpy as np

def train_geomancer(
    dataset_path: str,
    G_target_path: str,
    metric_names: list,
    n_timesteps: int = 100000,
    save_path: str = 'models/geomancer_policy.zip'
):
    """Train Geomancer RL agent."""

    # Load data and target
    dataset = np.load(dataset_path)
    G_target = np.load(G_target_path)

    # Create environment
    env = GeomancerEnv(
        dataset=dataset,
        G_target=G_target,
        metric_names=metric_names,
        max_steps=10
    )

    # Create RL agent (PPO)
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    # Train
    model.learn(total_timesteps=n_timesteps)

    # Save
    model.save(save_path)

    return model

# Usage:
if __name__ == '__main__':
    train_geomancer(
        dataset_path='data/embryoid_body.npy',
        G_target_path='outputs/geomancer/G_target.npy',
        metric_names=['Trustworthiness', 'Continuity', 'H0Persistence'],
        n_timesteps=100000
    )
```

---

### Step 5: Evaluation (Agent Replicates Expert)
```python
# manyagents/geomancer/evaluate.py

def evaluate_geomancer(model_path: str, dataset_path: str, G_target_path: str):
    """Evaluate trained Geomancer agent."""

    # Load
    model = PPO.load(model_path)
    dataset = np.load(dataset_path)
    G_target = np.load(G_target_path)

    # Create env
    env = GeomancerEnv(
        dataset=dataset,
        G_target=G_target,
        metric_names=['Trustworthiness', 'Continuity', 'H0Persistence'],
        max_steps=10
    )

    # Run episode
    obs = env.reset()
    done = False
    trajectory = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        trajectory.append({
            'step': info['step'],
            'action': info['action_name'],
            'G_current': info['G_current'],
            'distance': info['distance_to_target'],
            'reward': reward
        })

        print(f"Step {info['step']}: {info['action_name']} | Distance: {info['distance_to_target']:.4f}")

    # Final comparison
    print(f"\nTarget G: {G_target}")
    print(f"Final  G: {obs}")
    print(f"Final Distance: {trajectory[-1]['distance']:.4f}")

    return trajectory
```

---

## How Vision A and Vision B Connect

### Vision A Enables Vision B

**Phase 3 (Vision A)** provides the **building blocks**:
- ✅ Adapter interface for all tools
- ✅ MetricComputer for computing G
- ✅ Reward calculation infrastructure
- ✅ RL model abstractions

**Geomancer (Vision B)** uses those blocks to build:
- 🚧 ActionSpace (adapters as discrete actions)
- 🚧 GeomancerEnv (gym environment)
- 🚧 Training loop (PPO on geometric deviations)
- 🚧 Evaluation pipeline (trajectory analysis)

### Shared Components

Both modes use:
1. **manylatents.metrics.api** - Compute G
2. **manyagents.adapters.*** - Execute tools
3. **manyagents.metrics.MetricComputer** - Metric computation
4. **manyagents.models.RLModule** - RL training

### Different Use Cases

- **Vision A**: "Compare PCA vs STELLA" (static benchmarking)
- **Vision B**: "Learn to replicate expert workflow" (active learning)

---

## Implementation Priority

### Phase 3A: Complete Vision A ✅
- [x] MetricComputer
- [x] EvaluationPipeline
- [x] anchor_vs_comparison config
- [x] Test with manylatents vs manylatents
- [ ] Test with manylatents vs external adapter (sklearn/scanpy)

### Phase 3B: Build Vision B 🚧
- [ ] ActionSpace registry
- [ ] GeomancerEnv (gym environment)
- [ ] Compute expert target workflow
- [ ] Training loop
- [ ] Evaluation tools
- [ ] Config system for Geomancer experiments

---

## Files to Create

### Geomancer Module Structure
```
manyagents/
  geomancer/
    __init__.py              # Export main classes
    action_space.py          # Unified action registry
    environment.py           # GeomancerEnv gym environment
    compute_target.py        # Execute expert workflow → G_target
    train.py                 # Training loop
    evaluate.py              # Evaluation tools

configs/
  geomancer/
    expert_workflow.yaml     # Define expert reference workflow
    train_config.yaml        # Training hyperparameters

docs/
  geomancer_tutorial.md      # How to use Geomancer mode
```

---

## Next Steps

1. **Decision**: Start with Vision A completion (external adapter) OR jump to Vision B (Geomancer)?
2. **If Vision A**: Build ScikitLearnAdapter, test anchor vs comparison
3. **If Vision B**: Build ActionSpace and GeomancerEnv, define expert workflow
4. **Long-term**: Both visions coexist in manyAgents as different execution modes

---

## Key Insight

The "multi-agent communication" element you identified is the **Action Registry**:
- **Vision A**: Static coordination of multiple adapters in parallel
- **Vision B**: Dynamic RL-driven selection of adapters as sequential actions

Both need a **unified interface** where each tool/method is an "action" the system can take.

That's the common architectural element connecting both visions.
