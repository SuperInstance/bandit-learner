# bandit-learner Architecture

**Version**: 1.0.0
**Last Updated**: January 8, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [System Architecture](#system-architecture)
4. [Component Design](#component-design)
5. [Data Flow](#data-flow)
6. [Performance Optimizations](#performance-optimizations)
7. [Integration with frozen-model-rl](#integration-with-frozen-model-rl)
8. [Deployment Patterns](#deployment-patterns)

---

## Overview

bandit-learner is a production-ready contextual bandit library that extends frozen-model-rl with comprehensive implementations, monitoring, and experimentation capabilities. It's designed for real-time decision-making systems that require sub-millisecond latency and robust online learning.

### Key Goals

1. **Performance**: <1ms inference latency, online incremental updates
2. **Production-Ready**: Monitoring, error handling, fallbacks, persistence
3. **Extensibility**: Easy to add new algorithms and integrations
4. **Safety**: Guardrails against reward hacking, exploration risks
5. **Scalability**: Support thousands of concurrent learners

### Scope

bandit-learner focuses on:
- Contextual bandit algorithms (LinUCB, Thompson Sampling, etc.)
- Training and serving infrastructure
- Monitoring and metrics
- A/B testing framework
- Model persistence and state management

It does NOT cover:
- Full RL algorithms (use frozen-model-rl for IRO, KPO)
- Deep RL training (use separate libraries)
- Offline batch training (use frozen-model-rl)

---

## Design Philosophy

### 1. Separate Training and Serving

```python
# Training: Full-featured bandit
training_bandit = LinUCB(config)
training_bandit.update(arm, context, reward)

# Serving: Optimized for inference
serving_bandit = training_bandit.to_serving()
arm = serving_bandit.select_arm(context)  # Faster
```

### 2. Language Layering

- **Rust Core**: Performance-critical path (bandit algorithms, matrix ops)
- **Python API**: User-facing interface, integrations
- **FFI Layer**: PyO3 bindings for zero-copy data transfer

### 3. Observability First

Every component is instrumented:
- Structured logging (JSON)
- Prometheus metrics
- Distributed tracing (OpenTelemetry)
- Per-arm analytics

### 4. Safe Defaults

```python
# SafeBandit wraps any bandit with safety checks
safe_bandit = SafeBandit(
    base_bandit=LinUCB(...),
    fallback_policy=UniformRandom(),
    guardrails=[
        NoArmsWithZeroSelections(),
        MaxExplorationRate(0.3),
        MinRewardVariance(0.01)
    ]
)
```

### 5. Gradual Complexity

```python
# Simple: Just use a bandit
bandit = LinUCB(config)

# Medium: Add monitoring
bandit = MonitoredBandit(bandit, metrics)

# Complex: Add experimentation
bandit = ExperimentBandit(
    bandits={"control": control, "treatment": treatment},
    allocator=ThompsonAllocator()
)
```

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Application Layer                     │
│  (Content Recommendation, Constraint Optimization, etc.)     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Python API
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    bandit-learner Python API                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Bandit    │  │  Experiment  │  │   Monitor    │     │
│  │   Factory    │  │  Framework   │  │   Manager    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                  │              │
└─────────┼─────────────────┼──────────────────┼──────────────┘
          │                 │                  │
          │ PyO3 FFI        │                  │
          │                 │                  │
┌─────────▼─────────────────▼──────────────────▼──────────────┐
│                    bandit-learner Rust Core                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Bandit     │  │   Training   │  │   Serving    │     │
│  │  Algorithms  │  │  Engine      │  │  Optimizer   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                  │              │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐         │
│  │ Monitoring  │  │ Persistence │  │ Validation  │         │
│  │   & Metrics │  │   Layer     │  │   & Safety  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
          │
          │ Extends
          │
┌─────────▼───────────────────────────────────────────────────┐
│                    frozen-model-rl                           │
│  (Core RL concepts: IRO, KPO, Reward Functions)             │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
bandit-learner/
├── python/
│   └── bandit_learner/
│       ├── __init__.py
│       ├── bandit/
│       │   ├── __init__.py
│       │   ├── base.py              # Base bandit interface
│       │   ├── linucb.py            # LinUCB implementation
│       │   ├── thompson.py          # Thompson Sampling
│       │   ├── epsilon_greedy.py    # Epsilon-Greedy
│       │   ├── neural.py            # Neural Bandit
│       │   └── factory.py           # Bandit factory
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py           # Training loop
│       │   ├── updater.py           # Online updater
│       │   └── batch.py             # Batch training
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── server.py            # Serving server
│       │   ├── cache.py             # Caching layer
│       │   └── optimizer.py         # Serving optimizations
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── metrics.py           # Prometheus metrics
│       │   ├── logger.py            # Structured logging
│       │   └── analytics.py         # Per-arm analytics
│       ├── experimentation/
│       │   ├── __init__.py
│       │   ├── ab_test.py           # A/B testing
│       │   ├── multi_armed.py       # Multi-armed bandit tests
│       │   └── analyzer.py          # Statistical analysis
│       └── utils/
│           ├── __init__.py
│           ├── persistence.py       # Save/load
│           ├── validation.py        # Input validation
│           └── safety.py            # Safety guardrails
│
├── rust/
│   └── src/
│       ├── lib.rs                   # FFI entry point
│       ├── bandit/
│       │   ├── mod.rs               # Bandit trait
│       │   ├── linucb.rs            # LinUCB (frozen-model-rl)
│       │   ├── thompson.rs          # Thompson Sampling
│       │   ├── epsilon.rs           # Epsilon-Greedy
│       │   └── neural.rs            # Neural Bandit
│       ├── training/
│       │   ├── mod.rs
│       │   ├── online.rs            # Online updates
│       │   └── batch.rs             # Batch training
│       ├── serving/
│       │   ├── mod.rs
│       │   ├── optimizer.rs         # SIMD optimizations
│       │   └── cache.rs             # LRU cache for contexts
│       ├── monitoring/
│       │   ├── mod.rs
│       │   ├── metrics.rs           # Metrics collector
│       │   └── trace.rs             # Distributed tracing
│       ├── persistence/
│       │   ├── mod.rs
│       │   ├── snapshot.rs          # State snapshots
│       │   └── wal.rs               # Write-ahead log
│       └── error.rs                 # Error types
│
└── docs/
    ├── ARCHITECTURE.md              # This file
    ├── USER_GUIDE.md                # User guide
    ├── ALGORITHMS.md                # Algorithm specs
    └── DEVELOPER_GUIDE.md           # Developer guide
```

---

## Component Design

### 1. Bandit Algorithms

#### Core Trait (Rust)

```rust
pub trait ContextualBandit: Send + Sync {
    /// Select arm based on context (fast path)
    fn select_arm(&self, context: &[f64]) -> Result<usize>;

    /// Update with reward (online learning)
    fn update(&mut self, arm: usize, context: &[f64], reward: f64) -> Result<()>;

    /// Batch update (faster than multiple single updates)
    fn update_batch(&mut self, updates: &[(usize, Vec<f64>, f64)]) -> Result<()>;

    /// Reset to initial state
    fn reset(&mut self);

    /// Get metadata
    fn num_arms(&self) -> usize;
    fn context_dim(&self) -> usize;
    fn arm_counts(&self) -> Vec<usize>;

    /// Create optimized serving version
    fn to_serving(&self) -> Result<Box<dyn ServingBandit>>;
}
```

#### Serving Bandit Trait (Optimized)

```rust
pub trait ServingBandit: Send + Sync {
    /// Select arm (no updates, optimized for speed)
    fn select_arm(&self, context: &[f64]) -> Result<usize>;

    /// Bulk selection (SIMD-optimized)
    fn select_arms(&self, contexts: &[&[f64]]) -> Result<Vec<usize>>;

    /// No update method (serving is read-only)
}
```

#### Python Wrapper

```python
class Bandit(ABC):
    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """Select arm based on context"""
        pass

    @abstractmethod
    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update with observed reward"""
        pass

    def update_batch(self, updates: List[Tuple[int, np.ndarray, float]]) -> None:
        """Batch update (faster)"""
        for arm, context, reward in updates:
            self.update(arm, context, reward)

    @abstractmethod
    def to_serving(self) -> "ServingBandit":
        """Create optimized serving version"""
        pass
```

### 2. Training Infrastructure

#### Online Training

```python
class OnlineTrainer:
    """Train bandit from streaming data"""

    def __init__(self, bandit: Bandit, config: TrainingConfig):
        self.bandit = bandit
        self.config = config
        self.updater = OnlineUpdater(bandit, config)

    async def train_from_stream(self, stream: AsyncIterator[Observation]):
        """Train from async stream of observations"""
        async for obs in stream:
            self.updater.update(obs.arm, obs.context, obs.reward)

            if self.updater.count % self.config.checkpoint_interval == 0:
                self.bandit.save(self.config.checkpoint_path)
```

#### Batch Training

```python
class BatchTrainer:
    """Train bandit from historical data"""

    def train(self, bandit: Bandit, data: Dataset) -> TrainingReport:
        """Train from dataset (offline)"""
        # Shuffle data for better convergence
        data = data.shuffle()

        # Batch updates (faster than sequential)
        for batch in data.batches(self.config.batch_size):
            updates = [(obs.arm, obs.context, obs.reward) for obs in batch]
            bandit.update_batch(updates)

        return self.evaluate(bandit, data.test_set())
```

### 3. Serving Infrastructure

#### Serving Server

```python
class BanditServer:
    """High-performance serving server"""

    def __init__(self, bandit: ServingBandit, config: ServingConfig):
        self.bandit = bandit
        self.cache = LRUCache(maxsize=config.cache_size)
        self.executor = ThreadPoolExecutor(max_workers=config.workers)

    async def select_arm(self, context: np.ndarray) -> int:
        """Select arm with caching and batching"""
        # Check cache
        cache_key = hash(context.tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Select arm
        arm = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.bandit.select_arm,
            context
        )

        # Cache result
        self.cache[cache_key] = arm
        return arm

    async def select_arms_batch(self, contexts: List[np.ndarray]) -> List[int]:
        """Batch selection (SIMD-optimized)"""
        contexts_array = np.array(contexts)
        arms = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.bandit.select_arms,
            contexts_array
        )
        return arms.tolist()
```

#### Serving Optimizations

1. **SIMD Vectorization**: Use `packed_simd` for parallel context processing
2. **Memory Pooling**: Reuse allocations across requests
3. **Lazy Evaluation**: Defer computations until needed
4. **Caching**: Cache recent context→arm mappings
5. **Batching**: Process multiple contexts together

### 4. Monitoring

#### Metrics Collection

```python
class BanditMetrics:
    """Prometheus metrics for bandits"""

    def __init__(self, bandit_name: str):
        self.bandit_name = bandit_name

        # Counters
        self.selections = Counter(
            'bandit_selections_total',
            'Total arm selections',
            ['bandit', 'arm']
        )
        self.updates = Counter(
            'bandit_updates_total',
            'Total updates',
            ['bandit']
        )

        # Histograms
        self.selection_latency = Histogram(
            'bandit_selection_latency_ms',
            'Selection latency',
            ['bandit']
        )
        self.update_latency = Histogram(
            'bandit_update_latency_ms',
            'Update latency',
            ['bandit']
        )
        self.rewards = Histogram(
            'bandit_reward',
            'Reward distribution',
            ['bandit', 'arm']
        )

        # Gauges
        self.avg_reward = Gauge(
            'bandit_avg_reward',
            'Average reward',
            ['bandit']
        )
        self.regret = Gauge(
            'bandit_regret',
            'Cumulative regret',
            ['bandit']
        )
```

#### Per-Arm Analytics

```python
class ArmAnalytics:
    """Detailed analytics for each arm"""

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.stats = [ArmStats() for _ in range(n_arms)]

    def track_selection(self, arm: int, context: np.ndarray):
        """Track arm selection"""
        self.stats[arm].selections += 1
        self.stats[arm].last_selected = time.time()

    def track_reward(self, arm: int, reward: float):
        """Track arm reward"""
        self.stats[arm].rewards.append(reward)
        self.stats[arm].avg_reward = np.mean(self.stats[arm].rewards)
        self.stats[arm].reward_std = np.std(self.stats[arm].rewards)

    def get_report(self) -> Dict[int, ArmStats]:
        """Get analytics report"""
        return {
            arm: stats
            for arm, stats in enumerate(self.stats)
        }
```

### 5. Experimentation Framework

#### A/B Testing

```python
class ABTest:
    """A/B test framework for bandits"""

    def __init__(
        self,
        control: Bandit,
        treatment: Bandit,
        traffic_split: float = 0.5
    ):
        self.control = control
        self.treatment = treatment
        self.traffic_split = traffic_split
        self.metrics_control = BanditMetrics("control")
        self.metrics_treatment = BanditMetrics("treatment")

    def route(self, user_id: str, context: np.ndarray) -> Tuple[int, str]:
        """Route user to control or treatment"""
        # Deterministic routing (consistent per user)
        assignment = "control" if hash(user_id) % 100 < 50 else "treatment"

        bandit = self.control if assignment == "control" else self.treatment
        arm = bandit.select_arm(context)

        return arm, assignment

    def analyze(self, significance_level: float = 0.05) -> ABTestReport:
        """Analyze A/B test results"""
        # Statistical test (t-test)
        t_stat, p_value = stats.ttest_ind(
            self.metrics_control.rewards,
            self.metrics_treatment.rewards
        )

        # Determine winner
        if p_value < significance_level:
            avg_control = np.mean(self.metrics_control.rewards)
            avg_treatment = np.mean(self.metrics_treatment.rewards)
            winner = "treatment" if avg_treatment > avg_control else "control"
            lift = abs(avg_treatment - avg_control) / avg_control * 100
        else:
            winner = "none"
            lift = 0.0

        return ABTestReport(
            winner=winner,
            p_value=p_value,
            lift=lift,
            control_avg=np.mean(self.metrics_control.rewards),
            treatment_avg=np.mean(self.metrics_treatment.rewards)
        )
```

### 6. Persistence

#### State Snapshots

```rust
pub struct BanditSnapshot {
    /// Bandit type (for deserialization)
    pub bandit_type: String,

    /// Serialization version
    pub version: u32,

    /// Number of arms
    pub n_arms: usize,

    /// Context dimensionality
    pub context_dim: usize,

    /// Algorithm-specific state
    pub state: Vec<u8>,  // Serialized state

    /// Metadata (timestamp, etc.)
    pub metadata: SnapshotMetadata,
}

impl BanditSnapshot {
    pub fn save<B: ContextualBandit>(bandit: &B, path: &Path) -> Result<()> {
        let snapshot = Self::serialize(bandit)?;
        let bytes = bincode::serialize(&snapshot)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load<B: ContextualBandit>(path: &Path) -> Result<B> {
        let bytes = std::fs::read(path)?;
        let snapshot: BanditSnapshot = bincode::deserialize(&bytes)?;
        snapshot.deserialize()
    }
}
```

#### Write-Ahead Log

```rust
pub struct WriteAheadLog {
    /// Log file path
    path: PathBuf,

    /// Buffered writer
    writer: BufWriter<File>,

    /// Sync after N writes
    sync_interval: usize,
}

impl WriteAheadLog {
    pub fn append(&mut self, update: BanditUpdate) -> Result<()> {
        // Serialize update
        let bytes = bincode::serialize(&update)?;

        // Write to log
        self.writer.write_all(&bytes)?;

        // Sync periodically
        self.write_count += 1;
        if self.write_count % self.sync_interval == 0 {
            self.writer.flush()?;
            self.file.sync_all()?;
        }

        Ok(())
    }

    pub fn replay<B: ContextualBandit>(&self, bandit: &mut B) -> Result<()> {
        // Read and apply all updates from log
        for update in self.read_all()? {
            bandit.update(update.arm, &update.context, update.reward)?;
        }
        Ok(())
    }
}
```

---

## Data Flow

### Training Flow

```
┌─────────────┐
│   User      │
│ Interaction │
└──────┬──────┘
       │
       │ (context, arm, reward)
       ▼
┌─────────────┐
│  Collector  │  Collect observations
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   OnlineTrainer     │  Async stream processing
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Bandit.update()   │  Update bandit state
└──────┬──────────────┘
       │
       ├──────────────────────────────┐
       │                              │
       ▼                              ▼
┌──────────────┐            ┌──────────────┐
│   Metrics    │            │  Checkpoint  │
│  (Prometheus)│            │  (Periodic)  │
└──────────────┘            └──────────────┘
```

### Serving Flow

```
┌─────────────┐
│   Request   │
│   (context) │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  BanditServer       │  Check cache
└──────┬──────────────┘
       │
       ├─────────┐
       │         │
       ▼         ▼
    ┌──────┐  ┌──────────┐
    │ Hit  │  │  Miss    │
    └──┬───┘  └─────┬────┘
       │            │
       │            ▼
       │     ┌────────────────┐
       │     │ ServingBandit  │  Fast selection
       │     │   (Rust)       │
       │     └───────┬────────┘
       │             │
       └──────┬──────┘
              │
              ▼
     ┌────────────────┐
     │ Return Arm     │
     └────────────────┘
```

---

## Performance Optimizations

### 1. SIMD Vectorization (Rust)

```rust
use packed_simd::{f64x4, f64x8};

fn select_arm_simd(context: &[f64]) -> usize {
    // Process 8 contexts at once
    let chunks: Vec<_> = context.chunks(8).collect();

    for chunk in chunks {
        let simd_ctx = f64x8::from_slice_unaligned(chunk);

        // Compute UCBs in parallel
        let ucbs: Vec<f64x8> = arms.iter().map(|arm| {
            let theta = f64x8::from_slice_unaligned(&arm.theta);
            theta * simd_ctx  // Vectorized multiplication
        }).collect();
    }

    // Find max
    ucbs.iter().argmax()
}
```

### 2. Memory Pooling

```rust
pub struct ContextPool {
    /// Pool of pre-allocated context buffers
    buffers: Vec<Vec<f64>>,
}

impl ContextPool {
    pub fn acquire(&self) -> Vec<f64> {
        self.buffers.pop().unwrap_or_else(|| Vec::with_capacity(128))
    }

    pub fn release(&mut self, buffer: Vec<f64>) {
        buffer.clear();
        self.buffers.push(buffer);
    }
}
```

### 3. Lazy Evaluation

```python
class LazyBandit:
    """Defer computations until needed"""

    def __init__(self, bandit: Bandit):
        self.bandit = bandit
        self._theta_cache = {}  # arm -> theta
        self._dirty = set()  # Arms that need recomputation

    def update(self, arm, context, reward):
        """Mark as dirty, don't recompute yet"""
        self.bandit.update(arm, context, reward)
        self._dirty.add(arm)

    def select_arm(self, context):
        """Recompute if needed"""
        for arm in self._dirty:
            self._theta_cache[arm] = self.bandit.theta(arm)
        self._dirty.clear()

        # Use cached theta
        return self._select_from_cache(context)
```

---

## Integration with frozen-model-rl

### Shared Concepts

1. **ContextualBandit Trait**: Same interface as frozen-model-rl
2. **Reward Functions**: Can use frozen-model-rl's reward functions
3. **Integration**: Both work with equilibrium-tokens

### Extending frozen-model-rl

```rust
// Re-use frozen-model-rl's LinUCB
use frozen_model_rl::bandit::LinUCB as CoreLinUCB;

// Add monitoring
pub struct MonitoredLinUCB {
    core: CoreLinUCB,
    metrics: BanditMetrics,
}

impl ContextualBandit for MonitoredLinUCB {
    fn select_arm(&self, context: &[f64]) -> Result<usize> {
        let start = Instant::now();
        let arm = self.core.select_arm(context)?;
        self.metrics.selection_latency.observe(start.elapsed());

        Ok(arm)
    }
}
```

### Algorithm Compatibility

| Algorithm | frozen-model-rl | bandit-learner |
|-----------|----------------|----------------|
| LinUCB | ✅ Core | ✅ + Monitoring |
| Thompson Sampling | ✅ Core | ✅ + Full impl |
| Epsilon-Greedy | ❌ | ✅ New |
| Neural Bandit | ❌ | ✅ New |

---

## Deployment Patterns

### 1. Single Instance

```python
# Simple: One bandit for all users
bandit = LinUCB(config)
server = BanditServer(bandit)
server.run()
```

### 2. Per-User Bandits

```python
# Personalized: One bandit per user
user_bandits = {}  # user_id -> bandit

def get_bandit(user_id: str) -> Bandit:
    if user_id not in user_bandits:
        user_bandits[user_id] = LinUCB(config)
    return user_bandits[user_id]
```

### 3. Sharded Bandits

```python
# Scalable: Shard users across servers
class ShardedBanditServer:
    def __init__(self, n_shards: int):
        self.shards = [
            BanditServer(LinUCB(config))
            for _ in range(n_shards)
        ]

    def route(self, user_id: str, context: np.ndarray):
        shard_id = hash(user_id) % len(self.shards)
        return self.shards[shard_id].select_arm(context)
```

---

**Document Version**: 1.0
**Author**: Agent 6 (Architecture Designer)
**Status**: ✅ Complete
