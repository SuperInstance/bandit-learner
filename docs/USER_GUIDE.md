# bandit-learner User Guide

**Version**: 1.0.0
**Last Updated**: January 8, 2026

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Algorithm Selection](#algorithm-selection)
4. [Common Use Cases](#common-use-cases)
5. [Training](#training)
6. [Serving](#serving)
7. [Monitoring](#monitoring)
8. [A/B Testing](#ab-testing)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Python Installation

```bash
# Install from PyPI
pip install bandit-learner

# Or install with specific features
pip install bandit-learner[monitoring]  # With monitoring
pip install bandit-learner[experimental]  # With experimental features
```

### Rust Installation

```bash
# Add to Cargo.toml
[dependencies]
bandit-learner = "0.1.0"

# Or install from source
cargo install bandit-learner
```

### Dependencies

**Python:**
- Python 3.8+
- NumPy >= 1.20
- PyO3 (for FFI)

**Rust:**
- Rust 1.70+
- ndarray >= 0.15
- ndarray-linalg >= 0.16

---

## Quick Start

### Basic Usage (Python)

```python
from bandit_learner import LinUCB, BanditConfig

# 1. Configure bandit
config = BanditConfig(
    n_arms=20,           # Number of actions/configurations
    context_dim=12,      # Context feature dimensionality
    alpha=0.5            # Exploration parameter
)

# 2. Create bandit
bandit = LinUCB(config)

# 3. Select action
context = [0.5, 0.3, 0.8, ...]  # 12-dimensional context
arm = bandit.select_arm(context)

# 4. Observe reward
reward = 0.8  # User clicked, liked, etc.

# 5. Update bandit
bandit.update(arm, context, reward)
```

### Basic Usage (Rust)

```rust
use bandit_learner::bandit::LinUCB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create bandit
    let mut bandit = LinUCB::new(20, 12, 0.5)?;

    // 2. Select arm
    let context = vec![0.5; 12];
    let arm = bandit.select_arm(&context)?;

    // 3. Update with reward
    let reward = 0.8;
    bandit.update(arm, &context, reward)?;

    Ok(())
}
```

---

## Algorithm Selection

### LinUCB (Linear Upper Confidence Bound)

**Best for:**
- High-dimensional contexts (d > 10)
- When you need proven theoretical guarantees
- Fast inference is critical (<1ms)
- Linear relationship between context and reward

**Parameters:**
```python
config = BanditConfig(
    n_arms=20,
    context_dim=12,
    alpha=0.5  # Higher = more exploration
)
bandit = LinUCB(config)
```

**Tuning alpha:**
- Start with 0.5-1.0
- Increase if bandit converges too slowly
- Decrease if bandit explores too much
- Can decay over time: `alpha *= 0.99` per 1000 updates

### Thompson Sampling

**Best for:**
- Bayesian uncertainty quantification
- Non-linear models
- Better empirical performance in practice
- When exploration needs to adapt automatically

**Parameters:**
```python
config = ThompsonSamplingConfig(
    n_arms=20,
    context_dim=12,
    sigma=1.0  # Observation noise (higher = more exploration)
)
bandit = ThompsonSampling(config)
```

**Tuning sigma:**
- Start with 1.0 (standard normal)
- Increase if rewards are noisy (sigma = 2.0)
- Decrease if rewards are consistent (sigma = 0.5)

### Epsilon-Greedy

**Best for:**
- Simple baseline comparisons
- When other algorithms are overkill
- Debugging and testing
- Low-dimensional contexts (d < 5)

**Parameters:**
```python
config = EpsilonGreedyConfig(
    n_arms=20,
    context_dim=12,
    epsilon=0.1  # 10% exploration
)
bandit = EpsilonGreedy(config)
```

**Tuning epsilon:**
- Start with 0.1 (10% explore, 90% exploit)
- Decay over time: `epsilon = max(0.01, epsilon * 0.995)`
- Or use schedule: `epsilon = 1.0 / (1.0 + n_updates)`

### Neural Bandit

**Best for:**
- Complex, non-linear patterns
- Large-scale problems (1000+ arms)
- When you have lots of training data
- Willing to trade latency for accuracy

**Parameters:**
```python
config = NeuralBanditConfig(
    n_arms=1000,
    context_dim=50,
    hidden_layers=[64, 32],
    learning_rate=0.01
)
bandit = NeuralBandit(config)
```

### Algorithm Comparison

| Algorithm | Latency | Memory | Sample Efficiency | Use When |
|-----------|---------|--------|-------------------|----------|
| **LinUCB** | <1ms | ~2MB | High | High-dim contexts |
| **Thompson** | <1ms | ~3MB | High | Bayesian needed |
| **Epsilon** | <1ms | ~1MB | Low | Baselines |
| **Neural** | <5ms | ~5MB | Medium | Complex patterns |

---

## Common Use Cases

### 1. Content Recommendation

Select which article/video to show based on user features.

```python
from bandit_learner import LinUCB

# Arms = content items (e.g., top 100 articles)
# Context = user features (age, location, history, etc.)
bandit = LinUCB(n_arms=100, context_dim=20, alpha=0.5)

def recommend(user_id: str) -> int:
    # Extract user features
    context = extract_user_features(user_id)

    # Select content
    content_id = bandit.select_arm(context)

    return content_id

def on_user_click(user_id: str, content_id: int, clicked: bool):
    # Extract context again (or cache it)
    context = extract_user_features(user_id)

    # Reward = 1 if clicked, 0 if not
    reward = 1.0 if clicked else 0.0

    # Update bandit
    bandit.update(content_id, context, reward)
```

### 2. Constraint Weight Optimization (with frozen-model-rl)

Learn optimal constraint weights for equilibrium-tokens.

```python
from bandit_learner import ThompsonSampling
from frozen_model_rl import EquilibriumOrchestrator

# Create bandit (20 weight configurations)
bandit = ThompsonSampling(n_arms=20, context_dim=12, sigma=1.0)

# Pre-defined weight configurations
ARM_WEIGHTS = [
    [1.0, 0.0, 0.0, 0.0],  # Rate-only
    [0.0, 1.0, 0.0, 0.0],  # Context-only
    [0.25, 0.25, 0.25, 0.25],  # Balanced
    # ... 17 more configurations
]

def orchestrate_turn(conversation):
    # Extract context features
    context = extract_conversation_features(conversation)

    # Select weight configuration
    arm = bandit.select_arm(context)
    weights = ARM_WEIGHTS[arm]

    # Apply weights
    orchestrator = EquilibriumOrchestrator()
    result = orchestrator.with_weights(weights).orchestrate(conversation)

    # Calculate reward (conversation quality)
    reward = calculate_quality_score(result)

    # Update bandit
    bandit.update(arm, context, reward)

    return result
```

### 3. Hyperparameter Optimization

Learn optimal hyperparameters for a model.

```python
from bandit_learner import LinUCB

# Arms = hyperparameter configurations
# Context = dataset features (size, dimensionality, etc.)
bandit = LinUCB(n_arms=50, context_dim=10, alpha=0.5)

# Pre-defined hyperparameter configs
ARM_CONFIGS = [
    {"lr": 0.001, "batch_size": 32},
    {"lr": 0.01, "batch_size": 64},
    # ... 48 more configs
]

def select_hyperparams(dataset):
    context = extract_dataset_features(dataset)
    arm = bandit.select_arm(context)
    return ARM_CONFIGS[arm]

def on_training_complete(config, accuracy):
    # Context = dataset features used for selection
    context = extract_dataset_features(dataset)

    # Reward = accuracy (or other metric)
    reward = accuracy

    # Find arm for this config
    arm = ARM_CONFIGS.index(config)

    # Update bandit
    bandit.update(arm, context, reward)
```

### 4. Real-Time Bidding

Select optimal bid in online advertising.

```python
from bandit_learner import LinUCB

# Arms = bid levels ($0.10, $0.20, ..., $5.00)
# Context = auction features (user, ad, time, etc.)
bandit = LinUCB(n_arms=50, context_dim=15, alpha=0.3)

def select_bid(auction):
    context = extract_auction_features(auction)
    bid_level = bandit.select_arm(context)
    return bid_level * 0.10  # Convert to dollars

def on_auction_outcome(auction, bid, won, revenue):
    context = extract_auction_features(auction)

    if won:
        reward = revenue - bid  # Profit
    else:
        reward = 0.0  # Nothing gained

    bid_level = int(bid / 0.10)
    bandit.update(bid_level, context, reward)
```

---

## Training

### Online Training

Train from streaming data (real-time).

```python
from bandit_learner import OnlineTrainer, TrainingConfig

config = TrainingConfig(
    checkpoint_interval=1000,  # Save every 1000 updates
    checkpoint_path="checkpoints/bandit.pkl"
)

trainer = OnlineTrainer(bandit, config)

async def train_from_stream():
    async for observation in observation_stream:
        trainer.update(observation.arm, observation.context, observation.reward)

# Start training
import asyncio
asyncio.run(train_from_stream())
```

### Batch Training

Train from historical data (offline).

```python
from bandit_learner import BatchTrainer
import pandas as pd

# Load historical data
data = pd.read_csv("historical_data.csv")
# Columns: arm, context_1, context_2, ..., context_12, reward

trainer = BatchTrainer()

# Prepare data
observations = [
    (row.arm, row[['context_1', ..., 'context_12']].values, row.reward)
    for _, row in data.iterrows()
]

# Train
report = trainer.train(bandit, observations)
print(f"Training regret: {report.regret}")
print(f"Best arm: {report.best_arm}")
print(f"Average reward: {report.avg_reward}")
```

### Warm Start from Pre-trained Model

```python
# Load pre-trained bandit
bandit = LinUCB.load("models/pretrained_bandit.pkl")

# Continue training with your data
for obs in your_data:
    bandit.update(obs.arm, obs.context, obs.reward)
```

---

## Serving

### Local Serving

```python
from bandit_learner import BanditServer, ServingConfig

# Convert to serving-optimized version
serving_bandit = bandit.to_serving()

# Create server
config = ServingConfig(
    cache_size=10000,  # Cache 10K recent context→arm mappings
    workers=4          # 4 worker threads
)
server = BanditServer(serving_bandit, config)

# Serve
arm = server.select_arm(context)
```

### REST API

```python
from fastapi import FastAPI
from bandit_learner import LinUCB

app = FastAPI()
bandit = LinUCB.load("models/production_bandit.pkl")

@app.post("/select")
async def select_arm(context: List[float]):
    arm = bandit.select_arm(context)
    return {"arm": arm}

@app.post("/update")
async def update_bandit(arm: int, context: List[float], reward: float):
    bandit.update(arm, context, reward)
    return {"status": "ok"}
```

### gRPC Service

```python
import grpc
from bandit_learner_pb2 import SelectRequest, SelectResponse
from bandit_learner_pb2_grpc import BanditServiceServicer

class BanditServicer(BanditServiceServicer):
    def Select(self, request, context):
        arm = bandit.select_arm(request.context)
        return SelectResponse(arm=arm)

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
bandit_pb2_grpc.add_BanditServiceServicer_to_server(BanditServicer(), server)
server.add_insecure_port('[::]:50051')
server.start()
```

---

## Monitoring

### Prometheus Metrics

```python
from bandit_learner import BanditMetrics

# Create metrics
metrics = BanditMetrics("my_bandit")

# Wrap bandit for monitoring
monitored_bandit = MonitoredBandit(bandit, metrics)

# Metrics are automatically tracked
arm = monitored_bandit.select_arm(context)  # Records latency
monitored_bandit.update(arm, context, reward)  # Records reward

# Expose Prometheus endpoint
from prometheus_client import start_http_server
start_http_server(8000)  # Metrics at http://localhost:8000/metrics
```

### Per-Arm Analytics

```python
from bandit_learner import ArmAnalytics

analytics = ArmAnalytics(n_arms=20)

# Track selections and rewards
analytics.track_selection(arm, context)
analytics.track_reward(arm, reward)

# Get report
report = analytics.get_report()
for arm, stats in report.items():
    print(f"Arm {arm}:")
    print(f"  Selections: {stats.selections}")
    print(f"  Avg reward: {stats.avg_reward:.3f}")
    print(f"  Reward std: {stats.reward_std:.3f}")
```

### Logging

```python
from bandit_learner import StructuredLogger

logger = StructuredLogger("my_bandit")

# Log decisions
logger.log_selection(arm=arm, context=context, timestamp=time.time())

# Log updates
logger.log_update(arm=arm, context=context, reward=reward, timestamp=time.time())

# Export logs
logger.export("logs/bandit_decisions.jsonl")
```

---

## A/B Testing

### Simple A/B Test

```python
from bandit_learner import ABTest

# Create two bandits
control = LinUCB(n_arms=20, context_dim=12, alpha=0.5)
treatment = ThompsonSampling(n_arms=20, context_dim=12, sigma=1.0)

# Create A/B test
ab_test = ABTest(
    control=control,
    treatment=treatment,
    traffic_split=0.5  # 50% to each
)

# Route traffic
user_id = "user123"
context = extract_context(user_id)
arm, assignment = ab_test.route(user_id, context)

# Track results
reward = observe_reward(arm, context)
ab_test.track(assignment, arm, context, reward)

# Analyze after collecting data
report = ab_test.analyze(significance_level=0.05)
print(f"Winner: {report.winner}")
print(f"Lift: {report.lift:.2f}%")
print(f"P-value: {report.p_value:.4f}")
```

### Multi-Armed Bandit Test

Test multiple algorithms simultaneously.

```python
from bandit_learner import MultiArmedTest

# Create 4 bandits
bandits = {
    "linucb_0.3": LinUCB(n_arms=20, context_dim=12, alpha=0.3),
    "linucb_0.7": LinUCB(n_arms=20, context_dim=12, alpha=0.7),
    "thompson_0.5": ThompsonSampling(n_arms=20, context_dim=12, sigma=0.5),
    "thompson_1.5": ThompsonSampling(n_arms=20, context_dim=12, sigma=1.5),
}

# Create test
test = MultiArmedTest(bandits)

# Use Thompson Sampling to allocate traffic
assignment = test.allocate_traffic()  # Returns bandit name
arm = bandits[assignment].select_arm(context)

# Track and analyze
test.track(assignment, arm, context, reward)
report = test.analyze()
print(f"Best bandit: {report.best_bandit}")
```

---

## Best Practices

### 1. Context Feature Engineering

```python
# Do: Normalize features to similar scales
context = [
    normalize(user_age, 0, 100),      # 0-1
    normalize(income, 0, 200000),     # 0-1
    one_hot(country, 10),             # 10-dim one-hot
    ...
]

# Don't: Use raw features with vastly different scales
context = [
    25,              # Age: 0-100
    50000,           # Income: 0-200000
    country_id,      # 0-200
    ...
]  # Bandit will be dominated by income
```

### 2. Reward Design

```python
# Do: Use bounded rewards in [-1, +1] or [0, 1]
def calculate_reward(user_action):
    if user_action == "click":
        return 1.0
    elif user_action == "view":
        return 0.1
    else:
        return 0.0

# Don't: Use unbounded rewards
def calculate_reward(user_action):
    if user_action == "purchase":
        return 100.0  # Way too large!
    else:
        return 0.0
```

### 3. Exploration Scheduling

```python
# Do: Decay exploration over time
class DecayScheduler:
    def __init__(self, initial_alpha, decay_rate, min_alpha):
        self.alpha = initial_alpha
        self.decay_rate = decay_rate
        self.min_alpha = min_alpha

    def update(self):
        self.alpha = max(self.min_alpha, self.alpha * self.decay_rate)

# Use with LinUCB
scheduler = DecayScheduler(initial_alpha=1.0, decay_rate=0.9995, min_alpha=0.1)

for iteration in range(10000):
    arm = bandit.select_arm(context)
    bandit.update(arm, context, reward)
    scheduler.update()
    bandit.set_alpha(scheduler.alpha)
```

### 4. Checkpointing

```python
# Do: Save checkpoints regularly
from bandit_learner import CheckpointManager

manager = CheckpointManager(
    bandit=bandit,
    checkpoint_path="checkpoints/",
    interval=1000  # Every 1000 updates
)

manager.start()  # Runs in background

# Don't: Only save at the end (risk of losing data)
```

### 5. Validation

```python
# Do: Validate inputs
from bandit_learner import validate_context, validate_reward

try:
    validate_context(context, expected_dim=12)
    validate_reward(reward, min_val=-1.0, max_val=1.0)
except ValidationError as e:
    logger.error(f"Invalid input: {e}")
    return fallback_action()

# Don't: Assume inputs are always valid
```

---

## Troubleshooting

### Bandit Not Converging

**Symptoms:** Arm selection keeps oscillating, no clear winner.

**Solutions:**
1. Increase exploration (`alpha` or `sigma`)
2. Check if context features are informative
3. Verify reward signal is consistent
4. Try different algorithm

```python
# Example: Increase exploration
bandit.set_alpha(1.0)  # Was 0.5

# Or try Thompson Sampling
bandit = ThompsonSampling(n_arms=20, context_dim=12, sigma=1.0)
```

### Bandit Converging Too Slowly

**Symptoms:** Takes thousands of iterations to find best arm.

**Solutions:**
1. Decrease exploration
2. Use warm start from historical data
3. Reduce context dimensionality (feature selection)
4. Try simpler algorithm (epsilon-greedy)

```python
# Example: Decrease exploration
bandit.set_alpha(0.3)  # Was 0.5

# Or use epsilon-greedy for faster initial convergence
bandit = EpsilonGreedy(n_arms=20, context_dim=12, epsilon=0.1)
```

### Poor Performance on New Data

**Symptoms:** Bandit works well on training data but poorly in production.

**Solutions:**
1. Check for distribution shift (train vs. production data)
2. Add more informative context features
3. Increase exploration to adapt
4. Use separate bandit per segment (e.g., per country)

```python
# Example: Per-segment bandits
bandits = {}
for segment in ["US", "EU", "APAC"]:
    bandits[segment] = LinUCB(n_arms=20, context_dim=12, alpha=0.5)

bandit = bandits[user_segment]
```

### Memory Issues

**Symptoms:** Out of memory errors with large bandits.

**Solutions:**
1. Use serving bandit (smaller memory footprint)
2. Reduce number of arms
3. Use context hashing (for very high-dim contexts)
4. Batch process instead of all-at-once

```python
# Example: Use serving bandit
serving_bandit = bandit.to_serving()  # Smaller, read-only

# Or use smaller bandit
bandit = LinUCB(n_arms=100, context_dim=12, alpha=0.5)  # Was 1000 arms
```

### Latency Issues

**Symptoms:** Selection takes too long (>10ms).

**Solutions:**
1. Use serving bandit (optimized for inference)
2. Enable caching
3. Use Rust API instead of Python
4. Batch multiple selections

```python
# Example: Enable caching
server = BanditServer(serving_bandit, cache_size=10000)

# Or batch selections
arms = server.select_arms_batch([context1, context2, context3])
```

---

**Document Version**: 1.0
**Author**: Agent 6 (Architecture Designer)
**Status**: ✅ Complete
