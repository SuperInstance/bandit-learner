# bandit-learner Developer Guide

**Version**: 1.0.0
**Last Updated**: January 8, 2026

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Adding New Algorithms](#adding-new-algorithms)
4. [Testing](#testing)
5. [Benchmarks](#benchmarks)
6. [Documentation](#documentation)
7. [Release Process](#release-process)
8. [Contributing Guidelines](#contributing-guidelines)

---

## Development Setup

### Prerequisites

**For Python Development:**
```bash
# Python 3.8+
python --version

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

**For Rust Development:**
```bash
# Rust 1.70+
rustc --version

# Install development tools
rustup component add clippy rustfmt

# Install Python bindings
cargo install maturin
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/yourusername/bandit-learner.git
cd bandit-learner

# Build Rust core
cd rust
cargo build --release

# Build Python package
cd ..
pip install -e .

# Run tests
cargo test --manifest-path=rust/Cargo.toml
pytest tests/
```

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit code ...

# 3. Run tests
cargo test
pytest

# 4. Run linters
cargo clippy
flake8 python/
black python/

# 5. Commit
git add .
git commit -m "feat: add my feature"

# 6. Push and create PR
git push origin feature/my-feature
```

---

## Project Structure

```
bandit-learner/
├── python/
│   └── bandit_learner/
│       ├── __init__.py
│       ├── bandit/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract base class
│       │   ├── linucb.py            # LinUCB wrapper
│       │   ├── thompson.py          # Thompson Sampling wrapper
│       │   ├── epsilon_greedy.py    # Epsilon-Greedy wrapper
│       │   ├── neural.py            # Neural Bandit wrapper
│       │   └── factory.py           # Bandit factory
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── updater.py
│       ├── serving/
│       │   ├── __init__.py
│       │   └── server.py
│       ├── monitoring/
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── utils/
│           ├── __init__.py
│           └── validation.py
│
├── rust/
│   └── src/
│       ├── lib.rs                   # PyO3 bindings
│       ├── bandit/
│       │   ├── mod.rs
│       │   ├── linucb.rs
│       │   ├── thompson.rs
│       │   ├── epsilon.rs
│       │   └── neural.rs
│       ├── training/
│       │   └── mod.rs
│       ├── serving/
│       │   └── mod.rs
│       ├── monitoring/
│       │   └── mod.rs
│       └── error.rs
│
├── tests/
│   ├── test_bandit.py
│   ├── test_training.py
│   └── test_serving.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── USER_GUIDE.md
│   ├── ALGORITHMS.md
│   └── DEVELOPER_GUIDE.md          # This file
│
├── Cargo.toml
├── pyproject.toml
├── README.md
└── LICENSE
```

### Key Files

**Python API (`python/bandit_learner/`)**
- `__init__.py`: Public API exports
- `bandit/base.py`: Abstract base class for all bandits
- `bandit/factory.py`: Factory for creating bandits

**Rust Core (`rust/src/`)**
- `lib.rs`: PyO3 bindings and module exports
- `bandit/mod.rs`: Core bandit trait
- `error.rs`: Error types and handling

**Tests (`tests/`)**
- `test_bandit.py`: Bandit algorithm tests
- `test_training.py`: Training infrastructure tests
- `test_serving.py`: Serving optimization tests

---

## Adding New Algorithms

### Step 1: Implement Rust Core

```rust
// rust/src/bandit/my_algorithm.rs

use crate::bandit::ContextualBandit;
use crate::error::{Error, Result};
use ndarray::{Array1, Array2};

pub struct MyBandit {
    n_arms: usize,
    context_dim: usize,
    // Algorithm-specific state
}

impl MyBandit {
    pub fn new(n_arms: usize, context_dim: usize) -> Result<Self> {
        if n_arms == 0 || context_dim == 0 {
            return Err(Error::Bandit("Invalid parameters".into()));
        }

        Ok(Self {
            n_arms,
            context_dim,
            // Initialize state
        })
    }
}

impl ContextualBandit for MyBandit {
    fn select_arm(&self, context: &[f64]) -> Result<usize> {
        if context.len() != self.context_dim {
            return Err(Error::InvalidContext(format!(
                "Expected context of length {}, got {}",
                self.context_dim,
                context.len()
            )));
        }

        // Algorithm-specific selection logic
        let arm = /* ... */;
        Ok(arm)
    }

    fn update(&mut self, arm: usize, context: &[f64], reward: f64) -> Result<()> {
        if arm >= self.n_arms {
            return Err(Error::Bandit(format!("Invalid arm: {}", arm)));
        }

        // Algorithm-specific update logic
        /* ... */

        Ok(())
    }

    fn reset(&mut self) {
        *self = Self::new(self.n_arms, self.context_dim)
            .expect("Failed to reset");
    }

    fn num_arms(&self) -> usize {
        self.n_arms
    }

    fn context_dim(&self) -> usize {
        self.context_dim
    }

    fn arm_counts(&self) -> Vec<usize> {
        // Return selection counts
        vec![0; self.n_arms]
    }
}
```

### Step 2: Add PyO3 Bindings

```rust
// rust/src/lib.rs

use pyo3::prelude::*;

#[pymodule]
fn bandit_learner(_py: Python, m: &PyModule) -> PyResult<()> {
    // Export existing bandits
    m.add_class::<LinUCB>()?;
    m.add_class::<ThompsonSampling>()?;

    // Add new algorithm
    m.add_class::<MyBandit>()?;

    Ok(())
}

// Add PyO3 wrapper for MyBandit
#[pymodule]
mod my_bandit_python {
    use super::*;
    use pyo3::PyResult;

    #[pyclass(name = "MyBandit")]
    pub struct PyMyBandit {
        inner: MyBandit,
    }

    #[pymethods]
    impl PyMyBandit {
        #[new]
        fn new(n_arms: usize, context_dim: usize) -> PyResult<Self> {
            Ok(Self {
                inner: MyBandit::new(n_arms, context_dim)?,
            })
        }

        fn select_arm(&self, context: Vec<f64>) -> PyResult<usize> {
            Ok(self.inner.select_arm(&context)?)
        }

        fn update(&mut self, arm: usize, context: Vec<f64>, reward: f64) -> PyResult<()> {
            Ok(self.inner.update(arm, &context, reward)?)
        }

        fn reset(&mut self) -> PyResult<()> {
            Ok(self.inner.reset())
        }

        fn num_arms(&self) -> usize {
            self.inner.num_arms()
        }

        fn context_dim(&self) -> usize {
            self.inner.context_dim()
        }
    }
}
```

### Step 3: Add Python Wrapper

```python
# python/bandit_learner/bandit/my_algorithm.py

from typing import List
from bandit_learner_bandit import PyMyBandit  # Rust import
from .base import Bandit

class MyBandit(Bandit):
    """My custom bandit algorithm"""

    def __init__(self, n_arms: int, context_dim: int):
        self._inner = PyMyBandit(n_arms, context_dim)

    def select_arm(self, context: List[float]) -> int:
        """Select arm based on context"""
        return self._inner.select_arm(context)

    def update(self, arm: int, context: List[float], reward: float) -> None:
        """Update with observed reward"""
        self._inner.update(arm, context, reward)

    def reset(self) -> None:
        """Reset to initial state"""
        self._inner.reset()

    def num_arms(self) -> int:
        """Get number of arms"""
        return self._inner.num_arms()

    def context_dim(self) -> int:
        """Get context dimensionality"""
        return self._inner.context_dim()

    def to_serving(self) -> "ServingBandit":
        """Create serving-optimized version"""
        # Use Rust's serving optimization
        return ServingMyBandit(self._inner.to_serving())
```

### Step 4: Add to Factory

```python
# python/bandit_learner/bandit/factory.py

from enum import Enum
from typing import Union
from .linucb import LinUCB
from .thompson import ThompsonSampling
from .my_algorithm import MyBandit

class BanditType(Enum):
    LINUCB = "linucb"
    THOMPSON = "thompson"
    MY_ALGORITHM = "my_algorithm"

BanditConfig = Union[LinUCBConfig, ThompsonSamplingConfig, MyBanditConfig]

def create_bandit(config: BanditConfig) -> Bandit:
    """Factory function to create bandits"""
    if isinstance(config, LinUCBConfig):
        return LinUCB(config)
    elif isinstance(config, ThompsonSamplingConfig):
        return ThompsonSampling(config)
    elif isinstance(config, MyBanditConfig):
        return MyBandit(config)
    else:
        raise ValueError(f"Unknown config type: {type(config)}")
```

### Step 5: Add Tests

```python
# tests/test_my_algorithm.py

import pytest
import numpy as np
from bandit_learner import MyBandit

def test_my_bandit_creation():
    """Test bandit creation"""
    bandit = MyBandit(n_arms=5, context_dim=3)
    assert bandit.num_arms() == 5
    assert bandit.context_dim() == 3

def test_my_bandit_select_arm():
    """Test arm selection"""
    bandit = MyBandit(n_arms=5, context_dim=3)
    context = [0.5, 0.3, 0.8]
    arm = bandit.select_arm(context)
    assert 0 <= arm < 5

def test_my_bandit_update():
    """Test bandit update"""
    bandit = MyBandit(n_arms=5, context_dim=3)
    context = [0.5, 0.3, 0.8]
    arm = 2
    reward = 0.8

    # Should not raise
    bandit.update(arm, context, reward)

def test_my_bandit_convergence():
    """Test convergence to optimal arm"""
    bandit = MyBandit(n_arms=5, context_dim=3)
    context = [0.5, 0.3, 0.8]

    # Arm 2 is optimal
    for _ in range(1000):
        arm = bandit.select_arm(context)
        reward = 1.0 if arm == 2 else 0.0
        bandit.update(arm, context, reward)

    # Should select arm 2
    arm = bandit.select_arm(context)
    assert arm == 2
```

### Step 6: Add Documentation

```python
# python/bandit_learner/bandit/my_algorithm.py

class MyBandit(Bandit):
    """My custom bandit algorithm.

    This algorithm implements a novel approach to the contextual bandit problem
    that combines exploration and exploitation in a unique way.

    Args:
        n_arms: Number of arms (actions)
        context_dim: Dimensionality of context features

    Example:
        >>> from bandit_learner import MyBandit
        >>> bandit = MyBandit(n_arms=20, context_dim=12)
        >>> arm = bandit.select_arm([0.5] * 12)
        >>> bandit.update(arm, [0.5] * 12, 0.8)

    References:
        - Smith et al. (2025). "My Novel Bandit Algorithm". ICML.
    """
```

---

## Testing

### Test Structure

```
tests/
├── test_bandit.py              # Bandit algorithm tests
│   ├── test_linucb.py
│   ├── test_thompson.py
│   ├── test_epsilon_greedy.py
│   └── test_neural.py
├── test_training.py            # Training tests
├── test_serving.py             # Serving tests
└── test_integration.py         # Integration tests
```

### Unit Tests

```python
# tests/test_bandit/test_linucb.py

import pytest
import numpy as np
from bandit_learner import LinUCB

class TestLinUCB:
    def test_creation(self):
        """Test bandit creation"""
        bandit = LinUCB(n_arms=5, context_dim=3)
        assert bandit.num_arms() == 5
        assert bandit.context_dim() == 3

    def test_invalid_params(self):
        """Test invalid parameters"""
        with pytest.raises(ValueError):
            LinUCB(n_arms=0, context_dim=3)

        with pytest.raises(ValueError):
            LinUCB(n_arms=5, context_dim=0)

    def test_select_arm(self):
        """Test arm selection"""
        bandit = LinUCB(n_arms=5, context_dim=3)
        context = np.array([0.5, 0.3, 0.8])
        arm = bandit.select_arm(context)
        assert 0 <= arm < 5

    def test_update(self):
        """Test bandit update"""
        bandit = LinUCB(n_arms=5, context_dim=3)
        context = np.array([0.5, 0.3, 0.8])
        bandit.update(2, context, 0.8)  # Should not raise

    def test_convergence(self):
        """Test convergence to optimal arm"""
        bandit = LinUCB(n_arms=5, context_dim=3, alpha=0.5)
        context = np.array([0.5, 0.3, 0.8])

        # Arm 2 is optimal
        for _ in range(1000):
            arm = bandit.select_arm(context)
            reward = 1.0 if arm == 2 else 0.0
            bandit.update(arm, context, reward)

        # Should select arm 2
        arm = bandit.select_arm(context)
        assert arm == 2
```

### Integration Tests

```python
# tests/test_integration.py

import pytest
from bandit_learner import LinUCB, OnlineTrainer

def test_online_training():
    """Test online training loop"""
    bandit = LinUCB(n_arms=20, context_dim=12)
    trainer = OnlineTrainer(bandit)

    # Simulate streaming data
    for i in range(1000):
        context = np.random.randn(12)
        arm = bandit.select_arm(context)
        reward = np.random.rand()  # Random reward
        trainer.update(arm, context, reward)

    # Bandit should have learned something
    assert bandit.arm_counts().sum() == 1000

def test_end_to_end():
    """Test full pipeline"""
    # 1. Create bandit
    bandit = LinUCB(n_arms=20, context_dim=12)

    # 2. Train
    for _ in range(1000):
        context = np.random.randn(12)
        arm = bandit.select_arm(context)
        reward = 1.0 if arm == 5 else 0.0  # Arm 5 is optimal
        bandit.update(arm, context, reward)

    # 3. Convert to serving
    serving_bandit = bandit.to_serving()

    # 4. Serve
    context = np.random.randn(12)
    arm = serving_bandit.select_arm(context)

    # 5. Verify
    assert arm == 5  # Should select optimal arm
```

### Property-Based Tests

```python
# tests/test_properties.py

import pytest
from hypothesis import given, strategies as st
from bandit_learner import LinUCB

@given(
    n_arms=st.integers(min_value=2, max_value=100),
    context_dim=st.integers(min_value=1, max_value=50),
    alpha=st.floats(min_value=0.1, max_value=2.0)
)
def test_linucb_properties(n_arms, context_dim, alpha):
    """Test LinUCB properties"""
    bandit = LinUCB(n_arms=n_arms, context_dim=context_dim, alpha=alpha)

    # Property: select_arm always returns valid arm
    context = [0.5] * context_dim
    arm = bandit.select_arm(context)
    assert 0 <= arm < n_arms

    # Property: update doesn't raise
    bandit.update(arm, context, 0.5)

    # Property: counts increment
    assert bandit.arm_counts()[arm] == 1

@given(
    context=st.lists(st.floats(min_value=-1, max_value=1), min_size=12, max_size=12)
)
def test_context_validation(context):
    """Test context validation"""
    bandit = LinUCB(n_arms=5, context_dim=12)

    # Valid context
    arm = bandit.select_arm(context)
    assert 0 <= arm < 5

    # Invalid context (wrong length)
    with pytest.raises(ValueError):
        bandit.select_arm([0.5] * 10)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bandit/test_linucb.py

# Run with coverage
pytest --cov=bandit_learner --cov-report=html

# Run Rust tests
cargo test --manifest-path=rust/Cargo.toml
```

---

## Benchmarks

### Benchmark Setup

```python
# benchmarks/benchmark_bandits.py

import time
import numpy as np
from bandit_learner import LinUCB, ThompsonSampling, EpsilonGreedy

def benchmark_algorithm(algorithm, n_arms, context_dim, n_iterations):
    """Benchmark bandit algorithm"""
    # Create bandit
    if algorithm == "linucb":
        bandit = LinUCB(n_arms=n_arms, context_dim=context_dim, alpha=0.5)
    elif algorithm == "thompson":
        bandit = ThompsonSampling(n_arms=n_arms, context_dim=context_dim, sigma=1.0)
    elif algorithm == "epsilon":
        bandit = EpsilonGreedy(n_arms=n_arms, context_dim=context_dim, epsilon=0.1)

    # Benchmark selection
    selection_times = []
    for _ in range(n_iterations):
        context = np.random.randn(context_dim)

        start = time.perf_counter()
        arm = bandit.select_arm(context)
        end = time.perf_counter()

        selection_times.append(end - start)

        # Update
        reward = np.random.rand()
        bandit.update(arm, context, reward)

    # Compute statistics
    avg_time = np.mean(selection_times)
    p95_time = np.percentile(selection_times, 95)
    p99_time = np.percentile(selection_times, 99)

    return {
        "avg": avg_time * 1000,  # Convert to ms
        "p95": p95_time * 1000,
        "p99": p99_time * 1000,
    }

if __name__ == "__main__":
    algorithms = ["linucb", "thompson", "epsilon"]
    n_arms = 20
    context_dim = 12
    n_iterations = 10000

    for algorithm in algorithms:
        stats = benchmark_algorithm(algorithm, n_arms, context_dim, n_iterations)
        print(f"{algorithm}:")
        print(f"  Avg: {stats['avg']:.3f}ms")
        print(f"  P95: {stats['p95']:.3f}ms")
        print(f"  P99: {stats['p99']:.3f}ms")
```

### Running Benchmarks

```bash
# Python benchmarks
python benchmarks/benchmark_bandits.py

# Rust benchmarks
cargo bench --manifest-path=rust/Cargo.toml
```

### Performance Targets

| Algorithm | Avg Latency | P95 Latency | P99 Latency |
|-----------|-------------|-------------|-------------|
| LinUCB | <0.5ms | <0.8ms | <1.0ms |
| Thompson Sampling | <0.5ms | <0.8ms | <1.0ms |
| Epsilon-Greedy | <0.1ms | <0.2ms | <0.3ms |
| Neural Bandit | <3.0ms | <5.0ms | <8.0ms |

---

## Documentation

### Documentation Standards

All public APIs must have:
1. Docstrings (Google style)
2. Type hints
3. Examples
4. References (for algorithms)

### Example Docstring

```python
def select_arm(self, context: np.ndarray) -> int:
    """Select arm based on context features.

    This method implements the LinUCB algorithm, which computes an upper
    confidence bound for each arm and selects the arm with the highest bound.

    Args:
        context: Feature vector of shape (context_dim,). Features should be
            normalized to [-1, +1] or [0, 1] for numerical stability.

    Returns:
        The index of the selected arm (0 to n_arms-1).

    Raises:
        ValueError: If context dimensionality doesn't match context_dim.

    Example:
        >>> bandit = LinUCB(n_arms=20, context_dim=12)
        >>> context = np.array([0.5, 0.3, 0.8, ...])
        >>> arm = bandit.select_arm(context)
        >>> print(f"Selected arm: {arm}")
        Selected arm: 7

    References:
        - Chu et al. (2011). "A Contextual-Bandit Approach to Personalized
          News Article Recommendation". WWW.
    """
```

### Building Documentation

```bash
# Build API docs
sphinx-build -b html docs/ docs/_build/html

# View docs
open docs/_build/html/index.html
```

---

## Release Process

### Versioning

Use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. **Pre-release**
   - [ ] All tests passing
   - [ ] Benchmarks meet performance targets
   - [ ] Documentation updated
   - [ ] Changelog updated

2. **Testing**
   - [ ] Run full test suite
   - [ ] Test on multiple platforms (Linux, macOS, Windows)
   - [ ] Test with Python 3.8, 3.9, 3.10, 3.11

3. **Build**
   - [ ] Build Python wheel
   - [ ] Build Rust binaries
   - [ ] Test installation from wheel

4. **Release**
   - [ ] Create Git tag
   - [ ] Push to PyPI
   - [ ] Create GitHub release
   - [ ] Update website/documentation

### Example Release

```bash
# 1. Update version in files
# - Cargo.toml
# - pyproject.toml
# - python/bandit_learner/__init__.py

# 2. Update changelog
vim CHANGELOG.md

# 3. Commit changes
git add .
git commit -m "Release v1.0.0"

# 4. Create tag
git tag -a v1.0.0 -m "Release v1.0.0"

# 5. Push
git push origin main --tags

# 6. Build and publish to PyPI
cd python
python -m build
twine upload dist/bandit_learner-1.0.0.tar.gz
twine upload dist/bandit_learner-1.0.0-*.whl

# 7. Create GitHub release
gh release create v1.0.0 --notes "See CHANGELOG.md"
```

---

## Contributing Guidelines

### Code Style

**Python (PEP 8 + Black):**
```bash
# Format code
black python/

# Check style
flake8 python/
```

**Rust:**
```bash
# Format code
cargo fmt

# Check style
cargo clippy
```

### Commit Messages

Use conventional commits:
```
feat: add new bandit algorithm
fix: correct Sherman-Morrison update
docs: update algorithm documentation
test: add convergence tests
refactor: simplify bandit trait
perf: optimize matrix operations
```

### Pull Request Process

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Update documentation
6. Submit PR
7. Address review comments
8. Merge

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Benchmarks run (if performance-critical)
- [ ] No breaking changes (or documented)
- [ ] Changelog updated

### Getting Help

- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: Questions, ideas
- Discord/Slack: Real-time chat

---

**Document Version**: 1.0
**Author**: Agent 6 (Architecture Designer)
**Status**: ✅ Complete
