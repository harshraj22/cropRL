# 🧠 LLM Inference Scheduling RL Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment for training agents to schedule heterogeneous LLM inference requests across a simulated GPU cluster.

The agent learns to make routing, batching, caching, and autoscaling decisions that jointly minimise latency, maximise fairness, preserve response quality, and control GPU cost — a multi-objective optimisation problem that no single heuristic can solve.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Agent (external)                     │
│         Receives observations, emits SchedulingAction       │
└───────────┬─────────────────────────────────┬───────────────┘
            │  action                         │  observation + reward
            ▼                                 │
┌───────────────────────────────────────────────────────────────┐
│                  LLMInferenceEnvironment                      │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│   │  Workload     │  │  Cluster     │  │  Reward Model      │ │
│   │  Generator    │──│  Simulator   │──│  (multi-objective)  │ │
│   │  (traffic     │  │  (replicas,  │  │                    │ │
│   │   patterns)   │  │   KV cache,  │  │  Latency · SLO ·   │ │
│   │              │  │   batching)  │  │  Fairness · Quality │ │
│   └──────────────┘  └──────────────┘  │  Cost · Throughput  │ │
│                                       └────────────────────┘ │
│   ┌──────────────────────────────────────────────────────┐   │
│   │              Observation Builder                      │   │
│   │   Per-request (14-d) · Per-replica (16-d) · Global    │   │
│   │   (20-d) features  +  Action masks                   │   │
│   └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **Heterogeneous workloads** | Requests vary by prompt length (50–128K tokens), type (chat, code, RAG, reasoning, vision), SLO tier (P0–P3), and streaming mode |
| **Realistic GPU cluster** | Models H100, A100, and L4 GPUs with accurate compute/memory profiles and cost |
| **Continuous batching** | Iteration-level scheduling with chunked prefill (Sarathi-Serve style) |
| **KV cache simulation** | PagedAttention-style paged allocation, prefix caching via a prefix trie, LRU/LFU eviction |
| **Speculative decoding** | Draft-target model pipeline with configurable acceptance rates |
| **Autoscaling** | Agent-controlled scale-up/down with startup delays and cooldown periods |
| **Multi-objective reward** | 7 reward components: latency, SLO violation, fairness (Jain's index), quality, cost, throughput, starvation |
| **Action masking** | Invalid actions are masked to guide exploration |
| **Traffic patterns** | Poisson, bursty (MMPP), diurnal (sinusoidal), correlated prefix, and combined patterns |
| **OpenEnv compatible** | Client-server architecture with FastAPI / WebSocket, Pydantic-typed actions & observations |

---

## Quick Start

### Installation

```bash
# Clone and install in editable mode
cd llm_inference_rl_env
pip install -e .

# Or with dev dependencies (for testing)
pip install -e ".[dev]"
```

### Standalone Usage (No Server)

The environment can be used directly without a server for fast local iteration:

```python
from llm_inference_env import LLMInferenceEnv, SchedulingAction

# Create environment with default (dev) configuration
with LLMInferenceEnv() as env:
    result = env.reset(seed=42)
    print(f"Initial reward: {result.reward}")

    # Run a few steps with simple actions
    for step in range(100):
        action = SchedulingAction(
            request_index=0,        # schedule the first pending request
            replica_index=0,        # route to the first replica
            batch_admission=1,      # admit to batch
            chunk_size_level=1,     # 512-token prefill chunks
            use_prefix_cache=1,     # use prefix caching
        )
        result = env.step(action)

        if result.done:
            print(f"Episode ended at step {step}")
            break

    print(f"Final reward: {result.reward}")
```

### Server Mode (OpenEnv Protocol)

```bash
# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then connect using the OpenEnv client:

```python
from llm_inference_env import LLMInferenceEnv, SchedulingAction

# Connect to running server (async)
async with LLMInferenceEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(seed=42)
    action = SchedulingAction(request_index=0, replica_index=0)
    result = await env.step(action)

# Or synchronous
with LLMInferenceEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42)
    result = env.step(SchedulingAction(request_index=0, replica_index=0))
```

---

## Configuration

### Presets

Two built-in configuration presets are provided:

```python
from llm_inference_env import dev_config, prod_config, LLMInferenceEnv, EnvironmentConfig

# Small-scale development (fast iteration)
#   - 4x A100 GPUs, 2x 7B replicas + 1x 70B replica
#   - Poisson traffic at 5 req/s
#   - 5-minute episodes
env = LLMInferenceEnv(config=dev_config())

# Production-scale evaluation
#   - 32x H100 + 16x A100 + 8x L4 GPUs
#   - Mixed model sizes (7B, 70B, 405B) with autoscaling
#   - Diurnal + bursty traffic at 50 req/s
#   - 1-hour episodes
env = LLMInferenceEnv(config=prod_config())
```

### Custom Configuration

```python
from llm_inference_env import (
    EnvironmentConfig, WorkloadConfig, ClusterConfig,
    EpisodeConfig, RewardWeights, TrafficPattern
)

config = EnvironmentConfig(
    workload=WorkloadConfig(
        pattern=TrafficPattern.BURSTY,
        base_arrival_rate=10.0,
        burst_multiplier=4.0,
        streaming_fraction=0.8,
    ),
    cluster=ClusterConfig(
        gpu_inventory={"H100_SXM": 16, "A100_80GB": 8},
        initial_replicas=[
            {"model": "llama-3-7b", "gpu_type": "A100_80GB", "num_gpus": 1, "count": 4},
            {"model": "llama-3-70b", "gpu_type": "H100_SXM", "num_gpus": 4, "count": 2},
        ],
    ),
    episode=EpisodeConfig(duration_s=600.0, warmup_s=30.0),
    reward_weights=RewardWeights(w_slo_violation=10.0, w_cost=0.05),
)

env = LLMInferenceEnv(config=config)
```

---

## Action Space

The agent outputs a `SchedulingAction` with **high-frequency** (every step) and **low-frequency** (every N steps) decisions:

### High-Frequency (every step)

| Field | Type | Range | Description |
|---|---|---|---|
| `request_index` | `int` | `[0, 128)` | Which pending request to schedule |
| `replica_index` | `int` | `[0, 16)` | Which replica to route it to |
| `batch_admission` | `int` | `{0, 1}` | 0 = defer, 1 = admit to batch |
| `chunk_size_level` | `int` | `{0, 1, 2, 3}` | Prefill chunk: 256 / 512 / 1024 / 2048 tokens |
| `speculative_decoding` | `int` | `{0, 1}` | Enable speculative decoding |
| `use_prefix_cache` | `int` | `{0, 1}` | Use prefix cache matching |

### Low-Frequency (every N steps)

| Field | Type | Range | Description |
|---|---|---|---|
| `autoscale_small_model` | `int` | `{0..4}` | Scale small models: −2, −1, 0, +1, +2 |
| `autoscale_large_model` | `int` | `{0..4}` | Scale large models: −2, −1, 0, +1, +2 |
| `new_replica_gpu_type` | `int` | `{0, 1, 2}` | GPU for new replicas: H100, A100, L4 |
| `cache_eviction_policy` | `int` | `{0, 1, 2}` | Eviction: LRU, LFU, urgency-weighted |

Action masks are provided in each observation to prevent invalid actions.

---

## Observation Space

The observation is a structured set of normalised feature vectors:

| Component | Shape | Description |
|---|---|---|
| `pending_request_features` | `(128, 14)` | Per-request: tokens, type, SLO tier, wait time, urgency |
| `pending_request_mask` | `(128,)` | 1 for valid entries, 0 for padding |
| `replica_features` | `(16, 16)` | Per-replica: model size, GPU, batch util, KV cache, latency |
| `replica_mask` | `(16,)` | 1 for active replicas |
| `global_features` | `(20,)` | Cluster-wide: arrival rates, utilization, goodput, cost |
| `action_mask` | `Dict[str, List]` | Validity masks for each action dimension |

All features are normalised to `[0, 1]`.

---

## Reward Model

The reward is a weighted sum of seven components:

| Component | Signal | Weight | Reference |
|---|---|---|---|
| **Latency** | `−(TTFT/budget + TPOT/budget)` | `w_ttft=1.0`, `w_tpot=0.5` | Per-completion |
| **SLO Violation** | `−5.0` per violation | `w_slo=5.0` | Per-completion |
| **Quality** | `±quality` based on model-request match | `w_quality=2.0` | Per-completion |
| **Throughput** | `+0.2` per completed request | `w_throughput=0.2` | Per-step |
| **Fairness** | `−(1 − J)` using Jain's index per tier | `w_fairness=0.3` | Per-step |
| **Starvation** | `−10.0` per starved request (> 5× budget) | `w_starvation=10.0` | Per-step |
| **Cost** | `−(GPU cost × dt)` − autoscale churn | `w_cost=0.1` | Per-step |

All weights are configurable via `RewardWeights`.

---

## Project Structure

```
llm_inference_rl_env/
├── LLM_INFERENCE_ENVIRONMENT.md    # Design document
├── README.md                       # This file
├── pyproject.toml                  # Package configuration
├── openenv.yaml                    # OpenEnv manifest
├── __init__.py                     # Package exports
├── models.py                       # Pydantic data models, enums, configs
├── observation.py                  # Observation builder (feature encoding)
├── reward.py                       # Multi-objective reward computation
├── client.py                       # OpenEnv client (WebSocket + standalone)
├── simulation/                     # Discrete-event simulation engine
│   ├── engine.py                   #   Top-level orchestrator
│   ├── workload.py                 #   Traffic pattern generator
│   ├── cluster.py                  #   GPU cluster + autoscaling
│   ├── replica.py                  #   Model replica (batching + timing)
│   └── kv_cache.py                 #   Paged KV cache + prefix trie
├── server/                         # OpenEnv FastAPI server
│   ├── llm_inference_environment.py#   Environment implementation
│   ├── app.py                      #   FastAPI application
│   ├── Dockerfile                  #   Container image
│   └── requirements.txt            #   Server dependencies
└── tests/                          # Test suite
    ├── test_models.py
    ├── test_simulation.py
    ├── test_environment.py
    └── test_reward.py
```

---

## Training Integration

This environment is designed to work with any RL framework that supports the OpenEnv protocol:

- **[torchforge](https://github.com/meta-pytorch/torchforge)** — PyTorch's agentic RL framework
- **[TRL](https://huggingface.co/docs/trl/openenv)** — Hugging Face transformer RL
- **[Stable Baselines3](https://stable-baselines3.readthedocs.io/)** — via a Gymnasium wrapper (SB3-Contrib Masked PPO)
- **[CleanRL](https://github.com/vwxyzjn/cleanrl)** — single-file RL implementations

The recommended algorithm is **Masked PPO** with a set-encoder network architecture (per-request and per-replica MLPs → masked mean-pool → shared policy heads).

---

## Evaluation Metrics

| Metric | Definition | Target |
|---|---|---|
| **Goodput** | % of requests meeting both TTFT and TPOT SLOs | > 95% |
| **P50 / P99 TTFT** | Median and tail time-to-first-token | Minimise |
| **P50 / P99 TPOT** | Median and tail time-per-output-token | Minimise |
| **Jain's Fairness Index** | Wait time fairness within each SLO tier | > 0.9 |
| **Quality Match Rate** | % of quality-sensitive requests on appropriate model | > 90% |
| **Cost Efficiency** | Requests served per GPU-hour | Maximise |
| **Throughput** | Requests completed per second | Maximise |

---

## References

This environment is based on concepts from:

- **vLLM** — PagedAttention, continuous batching
- **Sarathi-Serve** — Chunked prefill, stall-free scheduling
- **Splitwise / DistServe** — Disaggregated prefill/decode (future work)
- **llm-d** — Prefix-cache-aware distributed scheduling
- **Vidur** — High-fidelity LLM inference simulation framework

See the full [Design Document](./LLM_INFERENCE_ENVIRONMENT.md) for detailed references.

---

## License

BSD 3-Clause License
