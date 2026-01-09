# bandit-learner Documentation Overview

**Version**: 1.0.0  
**Date**: January 8, 2026  

---

## Documentation Map

```
bandit-learner/
│
├── README.md (Start here!)
│   ├── Quick start
│   ├── Installation
│   └── Algorithm comparison
│
└── docs/
    ├── ARCHITECTURE.md (System design)
    │   ├── Design philosophy
    │   ├── System architecture
    │   ├── Component design
    │   └── Deployment patterns
    │
    ├── USER_GUIDE.md (How to use)
    │   ├── Installation
    │   ├── Algorithm selection
    │   ├── Use cases
    │   ├── Training & serving
    │   ├── Monitoring
    │   └── Troubleshooting
    │
    ├── ALGORITHMS.md (Deep dive)
    │   ├── LinUCB
    │   ├── Thompson Sampling
    │   ├── Epsilon-Greedy
    │   ├── Neural Bandit
    │   └── Theoretical guarantees
    │
    └── DEVELOPER_GUIDE.md (Contributing)
        ├── Development setup
        ├── Adding new algorithms
        ├── Testing
        ├── Benchmarks
        └── Release process
```

---

## Quick Navigation

### I'm a... **User**

Want to use bandit-learner in your project?

1. **Start**: [README.md](../README.md) - Overview and quick start
2. **Learn**: [USER_GUIDE.md](USER_GUIDE.md) - How to use bandit-learner
3. **Choose**: [ALGORITHMS.md](ALGORITHMS.md) - Which algorithm to use

### I'm a... **Developer**

Want to contribute or extend bandit-learner?

1. **Understand**: [ARCHITECTURE.md](ARCHITECTURE.md) - System design
2. **Setup**: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Development environment
3. **Implement**: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#adding-new-algorithms) - Add new algorithms

### I'm a... **Researcher**

Want to understand the algorithms?

1. **Algorithms**: [ALGORITHMS.md](ALGORITHMS.md) - Detailed specifications
2. **Theory**: [ALGORITHMS.md](ALGORITHMS.md#theoretical-guarantees) - Regret bounds
3. **Implementation**: [ARCHITECTURE.md](ARCHITECTURE.md#component-design) - Architecture details

### I'm a... **Production Engineer**

Want to deploy bandit-learner?

1. **Deploy**: [ARCHITECTURE.md](ARCHITECTURE.md#deployment-patterns) - Deployment options
2. **Monitor**: [USER_GUIDE.md](USER_GUIDE.md#monitoring) - Monitoring and metrics
3. **Operate**: [USER_GUIDE.md](USER_GUIDE.md#troubleshooting) - Troubleshooting guide

---

## Document Summary

| Document | Lines | Purpose | Audience |
|----------|-------|---------|----------|
| **README.md** | 243 | Project overview, quick start | Users |
| **ARCHITECTURE.md** | 850 | System design, components | Developers, Architects |
| **USER_GUIDE.md** | 802 | How to use, examples | Users |
| **ALGORITHMS.md** | 790 | Algorithm specifications | Researchers, Developers |
| **DEVELOPER_GUIDE.md** | 909 | Contributing guide | Developers |
| **Total** | 3,594 | Complete documentation | All |

---

## Key Concepts

### What is bandit-learner?

bandit-learner is a **production-ready contextual bandit library** for real-time decision-making.

**Key Features:**
- 4 bandit algorithms (LinUCB, Thompson Sampling, Epsilon-Greedy, Neural Bandit)
- <1ms inference latency
- Online learning from streaming data
- Production-grade monitoring and experimentation
- Python and Rust APIs

### When to use bandit-learner?

**Use cases:**
- Content recommendation (articles, videos, products)
- Constraint weight optimization (with frozen-model-rl)
- Hyperparameter optimization
- Real-time bidding (ad auctions)
- Any sequential decision-making problem

**When NOT to use:**
- Full RL problems (use frozen-model-rl)
- Offline batch training (use other libraries)
- Simple A/B testing (use simpler tools)

### How does it integrate with frozen-model-rl?

bandit-learner **extends** frozen-model-rl:

```
frozen-model-rl (Round 1)
├── Core bandit concepts
├── IRO and KPO optimizers
└── Reward functions

bandit-learner (Round 3)
├── More bandit algorithms (4 vs 2)
├── Production infrastructure
├── Monitoring and experimentation
└── A/B testing framework
```

Both work with **equilibrium-tokens** for adaptive conversation orchestration.

---

## Common Workflows

### Workflow 1: Get Started in 5 Minutes

```bash
# Install
pip install bandit-learner

# Use (Python)
from bandit_learner import LinUCB

bandit = LinUCB(n_arms=20, context_dim=12)
arm = bandit.select_arm([0.5] * 12)
bandit.update(arm, [0.5] * 12, 0.8)
```

See [README.md](../README.md#quick-start) for details.

### Workflow 2: Choose the Right Algorithm

1. Read [ALGORITHMS.md](ALGORITHMS.md#algorithm-overview)
2. Use the selection guide (flowchart)
3. Compare performance in [ALGORITHMS.md](ALGORITHMS.md#algorithm-comparison)
4. Try examples in [USER_GUIDE.md](USER_GUIDE.md#algorithm-selection)

### Workflow 3: Deploy to Production

1. Build serving infrastructure: [ARCHITECTURE.md](ARCHITECTURE.md#serving-infrastructure)
2. Set up monitoring: [USER_GUIDE.md](USER_GUIDE.md#monitoring)
3. Run A/B test: [USER_GUIDE.md](USER_GUIDE.md#ab-testing)
4. Troubleshoot: [USER_GUIDE.md](USER_GUIDE.md#troubleshooting)

### Workflow 4: Contribute a New Algorithm

1. Read architecture: [ARCHITECTURE.md](ARCHITECTURE.md#component-design)
2. Follow guide: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#adding-new-algorithms)
3. Write tests: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#testing)
4. Run benchmarks: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#benchmarks)

---

## Performance at a Glance

| Metric | Target | Algorithms |
|--------|--------|------------|
| **Inference** | <1ms | LinUCB, Thompson, Epsilon |
| **Inference** | <5ms | Neural Bandit |
| **Training** | <100µs | All (online update) |
| **Memory** | <10MB | LinUCB (2MB), Thompson (3MB), Neural (5MB) |
| **Throughput** | 1000+ req/s | Single instance |

---

## Algorithm Quick Reference

| Algorithm | Best For | Latency | Sample Efficiency |
|-----------|----------|---------|-------------------|
| **LinUCB** | High-dim contexts | <1ms | High |
| **Thompson** | Bayesian uncertainty | <1ms | High |
| **Epsilon** | Simple baselines | <1ms | Low |
| **Neural** | Complex patterns | <5ms | Medium |

See [ALGORITHMS.md](ALGORITHMS.md) for details.

---

## Getting Help

- **Documentation**: Start with relevant doc above
- **Examples**: [USER_GUIDE.md](USER_GUIDE.md#common-use-cases)
- **Issues**: GitHub Issues (bug reports, feature requests)
- **Discussions**: GitHub Discussions (questions, ideas)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Jan 8, 2026 | Initial release design |

---

**Next**: Read [README.md](../README.md) to get started!

