# Phase 3 (RL/Learning Agent) Design Implications from manyLatents Analysis

## Executive Summary

manyLatents provides a **production-ready template** for how Phase 3 should be architected:

1. **Abstract Algorithm Interface**: Standardized `fit/transform` pattern enables pluggable learning policies
2. **Config-Driven Instantiation**: Hydra enables dynamic agent decision-making as config generation
3. **Programmatic State Threading**: In-memory data passing perfect for feedback loops
4. **Unified Evaluation**: Single dispatcher handles diverse algorithm types uniformly
5. **Logging Infrastructure**: WandB integration ready for tracking agent trajectories

---

## 1. RL Environment as Evolved Orchestrator

### Current Phase 2 State
```
Planner (LLM) → Config Generation → execute_workflow_chain() → Results
```

### Phase 3 Target State
```
RLAgent (Policy Network) → Config Generation → execute_workflow_chain() → Reward
                     ↓
            Experience Replay Buffer
                     ↓
            Policy Updates (PPO/A3C)
```

**manyLatents Insight**: The `api.run()` function already supports this pattern:
- Takes config overrides programmatically
- Returns metrics immediately
- Chains inputs to outputs in memory

**Phase 3 Implementation**:
```python
class RLAgentOrchestrator:
    def __init__(self, policy_network, reward_fn, algorithm_space):
        self.policy = policy_network
        self.reward_fn = reward_fn
        self.algorithms = algorithm_space  # All available tools
    
    def step(self, state_embeddings, available_metrics):
        # State = current embeddings + metrics from previous step
        action = self.policy(state_embeddings, available_metrics)  # Outputs algorithm config
        
        # Execute via manyAgents orchestrator (Phase 1)
        result = execute_workflow_chain(
            steps=[action],  # Single action as workflow step
            state=state_embeddings
        )
        
        # Compute reward from manyLatents-style metrics
        reward = self.reward_fn(result['scores'])
        
        return result['embeddings'], reward, result['scores']
```

---

## 2. State Representation: Embeddings as Latent State

### manyLatents Pattern
- **Input**: Raw data tensor
- **Output**: Embeddings + metrics dict
- **State Transition**: x_i → embed_i → metrics_i → x_{i+1}

### Phase 3 Adaptation
```python
class EnvironmentState:
    def __init__(self, embeddings: np.ndarray, metrics: dict, metadata: dict):
        self.embeddings = embeddings  # Current latent representation
        self.metrics = metrics        # Quality signals (trustworthiness, continuity, etc.)
        self.metadata = metadata      # Algorithm history, timestamps
    
    def to_tensor(self):
        """Convert for policy network input"""
        # Embeddings: (n_samples, n_dims)
        # Metrics: normalize and flatten
        metrics_flat = flatten_metrics(self.metrics)  # (n_metrics,)
        # Concatenate or create multi-modal input
        return np.concatenate([
            self.embeddings.mean(axis=0),  # Global embedding statistics
            metrics_flat
        ])  # (n_dims + n_metrics,)
```

**Key Design Decision**: Use manyLatents' three-level metrics as explicit state components:
- **Embedding-level**: Trustworthiness, continuity (quality of current representation)
- **Dataset-level**: Dimensionality, variance (problem hardness)
- **Module-level**: Training loss, convergence (algorithm performance)

---

## 3. Action Space: Algorithm Selection + Hyperparameters

### manyLatents Enables This
The Hydra config system already supports:
```python
# Action as config override
action = {
    'algorithms.latent._target_': 'manylatents.algorithms.latent.phate.PHATEModule',
    'algorithms.latent.n_components': 20,
    'algorithms.latent.knn': 7,
    'algorithms.latent.gamma': 0.5
}

# Execute with state input
result = run(input_data=state.embeddings, **action)
```

### Phase 3 Action Space Architecture
```python
class ActionSpace:
    def __init__(self):
        # Discrete: which algorithm to run
        self.algorithm_choices = [
            'pca', 'phate', 'umap', 'tsne', 'autoencoder', 'biocluster'
        ]
        
        # Continuous: hyperparameters per algorithm
        self.hyperparameter_ranges = {
            'pca': {'n_components': (2, 100)},
            'phate': {
                'n_components': (2, 100),
                'knn': (3, 30),
                'gamma': (0.1, 2.0)
            },
            'autoencoder': {
                'latent_dim': (10, 200),
                'learning_rate': (1e-4, 1e-2),
                'batch_size': (32, 512)
            }
        }
    
    def sample_action(self):
        """Sample random action from space"""
        algo = np.random.choice(self.algorithm_choices)
        hp_ranges = self.hyperparameter_ranges[algo]
        hyperparams = {
            hp: np.random.uniform(low, high)
            for hp, (low, high) in hp_ranges.items()
        }
        return {'algorithm': algo, 'hyperparameters': hyperparams}
    
    def action_to_config(self, action):
        """Convert action to Hydra config override"""
        algo = action['algorithm']
        hp = action['hyperparameters']
        
        # Build config path
        base_config = f'algorithms.latent' if algo != 'autoencoder' else 'algorithms.lightning'
        
        config = {
            f'{base_config}._target_': f'manylatents.algorithms.latent.{algo}.{algo.upper()}Module',
            **{f'{base_config}.{k}': v for k, v in hp.items()}
        }
        return config
```

---

## 4. Reward Function: Normalize manyLatents Metrics

### manyLatents Evaluation Pattern
```python
# Returns dict of metrics
scores = {
    'trustworthiness': 0.87,
    'continuity': 0.92,
    'reconstruction_error': 0.034,
    'variance_explained': 0.95
}
```

### Phase 3 Reward Normalization
```python
class RewardFunction:
    def __init__(self):
        # Define reward composition
        self.metric_weights = {
            'trustworthiness': 0.3,      # Primary quality metric
            'continuity': 0.3,            # Secondary quality metric
            'reconstruction_error': -0.2, # Penalize reconstruction errors
            'variance_explained': 0.2     # Encourage information retention
        }
        
        # Normalization ranges (empirical or domain knowledge)
        self.metric_bounds = {
            'trustworthiness': (0.0, 1.0),
            'continuity': (0.0, 1.0),
            'reconstruction_error': (0.0, 0.5),
            'variance_explained': (0.0, 1.0)
        }
    
    def normalize(self, value, metric_name):
        """Normalize metric to [-1, 1]"""
        low, high = self.metric_bounds[metric_name]
        normalized = 2.0 * (value - low) / (high - low) - 1.0
        return np.clip(normalized, -1.0, 1.0)
    
    def compute(self, metrics_dict: dict, metadata: dict = None) -> float:
        """Compute scalar reward from metrics"""
        reward = 0.0
        for metric_name, weight in self.metric_weights.items():
            if metric_name in metrics_dict:
                value = metrics_dict[metric_name]
                normalized = self.normalize(value, metric_name)
                reward += weight * normalized
        
        # Optional: add exploration bonus or penalty
        if metadata and 'algorithm_diversity' in metadata:
            reward += 0.1 * metadata['algorithm_diversity']
        
        return reward
```

---

## 5. Experience Replay Infrastructure

### Gap in manyLatents
- Algorithms are trained fresh each time
- No mechanism for storing/replaying trajectories
- Checkpointing only saves algorithm state, not decisions

### Phase 3 Addition: Trajectory Buffer
```python
@dataclass
class Transition:
    """Single (s, a, r, s') transition"""
    state: EnvironmentState
    action: dict  # Algorithm config
    reward: float
    next_state: EnvironmentState
    done: bool
    info: dict  # Metadata about execution

class TrajectoryBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = collections.deque(maxlen=capacity)
    
    def record(self, transition: Transition):
        """Store transition"""
        self.buffer.append(transition)
    
    def sample_batch(self, batch_size: int) -> list:
        """Sample batch for learning"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def save_trajectory(self, filepath: str):
        """Persist trajectory for offline analysis"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load_trajectory(self, filepath: str):
        """Load previous trajectory"""
        import pickle
        with open(filepath, 'rb') as f:
            self.buffer.extend(pickle.load(f))
```

### WandB Integration for Trajectory Logging
```python
def log_trajectory_to_wandb(trajectory: list, episode: int):
    """Log entire trajectory as structured table"""
    import wandb
    
    # Create table with columns for each step
    columns = ["Step", "Algorithm", "Hyperparams", "Reward", "Trustworthiness", "Continuity"]
    rows = []
    
    for i, transition in enumerate(trajectory):
        rows.append([
            i,
            transition.action.get('algorithm', 'unknown'),
            str(transition.action.get('hyperparameters', {})),
            transition.reward,
            transition.next_state.metrics.get('trustworthiness', 0),
            transition.next_state.metrics.get('continuity', 0)
        ])
    
    wandb.log({
        f"trajectory/episode_{episode}": wandb.Table(columns=columns, data=rows),
        f"trajectory/cumulative_reward": sum(t.reward for t in trajectory),
        f"trajectory/length": len(trajectory)
    }, step=episode)
```

---

## 6. Policy Architecture: From LLM Planner to Neural Network

### Current Phase 2 (LLM-Based Planning)
```python
class LLMPlanner(AgentAdapter):
    def run(self, goals, scientific_context, **kwargs):
        prompt = build_prompt(goals, context)
        plan = llm.complete(prompt)
        return parse_plan(plan)  # Returns list of steps
```

### Phase 3 Target (RL Policy Network)
```python
class RLPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (action selection)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value estimation for policy gradient)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.feature_net(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

class PPOAgent:
    """Proximal Policy Optimization agent"""
    def __init__(self, policy: RLPolicy, learning_rate: float = 1e-4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    def select_action(self, state_tensor):
        """Sample action from policy"""
        with torch.no_grad():
            action_logits, _ = self.policy(state_tensor)
            dist = torch.distributions.Categorical(logits=action_logits)
            action_idx = dist.sample()
        return action_idx.item()
    
    def compute_loss(self, batch):
        """PPO loss: policy gradient + value function"""
        states, actions, rewards, next_states, dones = batch
        
        # Compute advantages
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            targets = rewards + 0.99 * next_values * (1 - dones)
        
        action_logits, values = self.policy(states)
        advantages = targets - values.detach()
        
        # Actor loss (policy gradient)
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss (value function)
        critic_loss = ((values - targets) ** 2).mean()
        
        return actor_loss + 0.5 * critic_loss
```

---

## 7. Integration with manyAgents Orchestrator

### Minimal Changes to Phase 1 Architecture

**Current**: `execute_workflow_chain()` reads static Hydra config

**Phase 3 Enhancement**:
```python
class AdaptiveOrchestrator:
    def __init__(self, agent, initial_data):
        self.agent = agent
        self.state = {'embeddings': initial_data}
        self.trajectory = []
    
    def execute_adaptive_workflow(self, max_steps: int = 10, goal: dict = None):
        """Execute workflow with agent-driven decisions"""
        
        for step_idx in range(max_steps):
            # Agent observes current state and decides next action
            action = self.agent.select_action(self.state)
            action_config = self.action_space.action_to_config(action)
            
            # Execute via existing orchestrator
            step_result = execute_workflow_chain(
                steps=[{'_target_': 'ManyLatentsAdapter', 'config': action_config}],
                state=self.state
            )
            
            # Record transition
            reward = self.reward_fn(step_result['metrics'])
            next_state = self.state_builder(step_result)
            
            transition = Transition(
                state=self.state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=reward > self.goal_threshold,
                info={'step': step_idx}
            )
            self.trajectory.append(transition)
            self.trajectory_buffer.record(transition)
            
            # Update state for next iteration
            self.state = next_state
            
            # Early termination if goal achieved
            if transition.done:
                break
        
        return self.trajectory
```

---

## 8. Comparison: manyLatents vs. Phase 3 Architecture

| Component | manyLatents | Phase 3 (RL) |
|-----------|------------|------------|
| **Algorithm Input** | Config (Hydra) | Policy network output → Config |
| **Algorithm Output** | Embeddings + Metrics | Embeddings + Metrics + Reward |
| **State Representation** | Implicit in embeddings | Explicit: embeddings + metrics tensor |
| **Decision Making** | None (deterministic) | Policy network (learned) |
| **Learning Signal** | Evaluation metrics | Normalized rewards |
| **Experience Storage** | Outputs only | Full transitions (s,a,r,s') |
| **Training Loop** | Single pass | Episode-based with replay |

---

## 9. Implementation Roadmap for Phase 3

### Phase 3.1: RL Infrastructure (Weeks 1-2)
1. Implement `EnvironmentState` and `Transition` classes
2. Create `TrajectoryBuffer` for experience storage
3. Add `RewardFunction` wrapper for metric normalization
4. Write integration tests with mock algorithms

### Phase 3.2: Policy Network (Weeks 3-4)
1. Implement `RLPolicy` base class (PPO + A3C templates)
2. Create `ActionSpace` with algorithm/hyperparameter mapping
3. Build policy training loop with batch processing
4. Add checkpointing for policy parameters

### Phase 3.3: Agent Integration (Weeks 5-6)
1. Implement `RLAgent` class using policy network
2. Integrate with `AdaptiveOrchestrator`
3. Add trajectory logging to WandB
4. Benchmarking: compare RL vs. LLM Planner performance

### Phase 3.4: Advanced Features (Weeks 7-8)
1. Multi-agent coordination (multiple agents making decisions)
2. Curriculum learning (increase problem difficulty over episodes)
3. Transfer learning (reuse policy across datasets)
4. Sensitivity analysis (identify key hyperparameters)

---

## Key Takeaway

**manyLatents' config-driven, state-threading architecture is the perfect foundation for Phase 3's RL loop.**

The only additions needed are:
1. **RL-specific state representation** (embeddings + metrics)
2. **Policy network** that outputs algorithm configs
3. **Experience replay buffer** for learning from trajectories
4. **Reward normalization** to align metrics with RL objectives

Everything else (execution, logging, evaluation) can reuse existing infrastructure.

