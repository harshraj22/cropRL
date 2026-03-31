"""
LLM Inference Scheduling RL Environment — Data Models.

All Pydantic data models, enums, and configuration structures used by both
the simulation engine (server-side) and the client.

References design document sections: §2.2, §3.2-3.6, §4.1-4.4, §5.2, §6.1, §7.1
"""

from __future__ import annotations

import hashlib
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Try OpenEnv imports; fall back to standalone Pydantic if not available
try:
    from openenv.core.env_server.types import (
        Action as _OpenEnvAction,
        Observation as _OpenEnvObservation,
    )
except ImportError:
    # If openenv-core is not installed, provide compatible base classes
    class _OpenEnvAction(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class _OpenEnvObservation(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
        done: bool = Field(default=False)
        reward: Optional[float] = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Enums (§2.1, §2.2, §2.4, §3.4, §5.2, §7.1)
# =============================================================================

class RequestType(str, Enum):
    """Types of LLM inference requests."""
    CHAT = "CHAT"
    CODE = "CODE"
    RAG = "RAG"
    TOOL_USE = "TOOL_USE"
    REASONING = "REASONING"
    VISION = "VISION"


class ReasoningDepth(str, Enum):
    """Depth of reasoning required by a request."""
    DIRECT = "DIRECT"
    SHALLOW_COT = "SHALLOW_COT"
    DEEP_COT = "DEEP_COT"
    MULTI_STEP = "MULTI_STEP"


class SLOTier(str, Enum):
    """SLO priority tiers, P0 (highest) to P3 (lowest)."""
    P0 = "P0"  # Realtime
    P1 = "P1"  # Interactive
    P2 = "P2"  # Batch
    P3 = "P3"  # Best-effort


class ModelAffinity(str, Enum):
    """Which model sizes a request can be routed to."""
    SMALL_ONLY = "SMALL_ONLY"
    LARGE_ONLY = "LARGE_ONLY"
    ANY = "ANY"


class TrafficPattern(str, Enum):
    """Workload traffic arrival patterns."""
    POISSON = "POISSON"
    BURSTY = "BURSTY"
    DIURNAL = "DIURNAL"
    CORRELATED = "CORRELATED"
    DIURNAL_WITH_BURSTS = "DIURNAL_WITH_BURSTS"


class StepMode(str, Enum):
    """How the simulation advances between agent decisions."""
    EVENT_DRIVEN = "EVENT_DRIVEN"
    FIXED_TIMESTEP = "FIXED_TIMESTEP"


class EvictionPolicy(str, Enum):
    """KV cache eviction strategies."""
    LRU = "LRU"
    LFU = "LFU"
    URGENCY_WEIGHTED = "URGENCY_WEIGHTED"


class GPUType(str, Enum):
    """Supported GPU types."""
    H100_SXM = "H100_SXM"
    A100_80GB = "A100_80GB"
    L4_24GB = "L4_24GB"


# =============================================================================
# Infrastructure Models (§3.2 - §3.6)
# =============================================================================

class GPUProfile(BaseModel):
    """Hardware profile for a GPU type. Reference: §3.2."""
    model_config = ConfigDict(frozen=True)

    gpu_type: GPUType
    memory_gb: float
    compute_tflops_fp16: float
    memory_bandwidth_gbps: float
    interconnect_bandwidth_gbps: float
    cost_per_hour: float

    @property
    def cost_per_second(self) -> float:
        return self.cost_per_hour / 3600.0


# Pre-defined GPU profiles from the design doc table
GPU_PROFILES: Dict[GPUType, GPUProfile] = {
    GPUType.H100_SXM: GPUProfile(
        gpu_type=GPUType.H100_SXM,
        memory_gb=80.0,
        compute_tflops_fp16=989.0,
        memory_bandwidth_gbps=3350.0,
        interconnect_bandwidth_gbps=900.0,
        cost_per_hour=3.50,
    ),
    GPUType.A100_80GB: GPUProfile(
        gpu_type=GPUType.A100_80GB,
        memory_gb=80.0,
        compute_tflops_fp16=312.0,
        memory_bandwidth_gbps=2039.0,
        interconnect_bandwidth_gbps=600.0,
        cost_per_hour=2.00,
    ),
    GPUType.L4_24GB: GPUProfile(
        gpu_type=GPUType.L4_24GB,
        memory_gb=24.0,
        compute_tflops_fp16=121.0,
        memory_bandwidth_gbps=300.0,
        interconnect_bandwidth_gbps=32.0,
        cost_per_hour=0.40,
    ),
}


class ParallelismConfig(BaseModel):
    """Tensor and pipeline parallelism configuration."""
    model_config = ConfigDict(frozen=True)

    tensor_parallel: int = 1
    pipeline_parallel: int = 1


class SpecDecConfig(BaseModel):
    """Speculative decoding configuration. Reference: §3.6."""
    model_config = ConfigDict(frozen=True)

    draft_model: str = "llama-3-3b"
    draft_model_latency_ms: float = 2.0
    num_speculative_tokens: int = 5
    acceptance_rate: float = 0.8


class ModelSpec(BaseModel):
    """Specification for an LLM model architecture (used to compute timings)."""
    model_config = ConfigDict(frozen=True)

    name: str
    params_billions: float
    num_layers: int
    d_model: int
    n_heads: int
    d_head: int = 128
    dtype_bytes: int = 2  # FP16
    max_context_length: int = 131072
    capabilities: List[RequestType] = Field(default_factory=lambda: list(RequestType))


# Pre-defined model specs
MODEL_SPECS: Dict[str, ModelSpec] = {
    "llama-3-7b": ModelSpec(
        name="llama-3-7b",
        params_billions=7.0,
        num_layers=32,
        d_model=4096,
        n_heads=32,
        max_context_length=131072,
        capabilities=list(RequestType),
    ),
    "llama-3-70b": ModelSpec(
        name="llama-3-70b",
        params_billions=70.0,
        num_layers=80,
        d_model=8192,
        n_heads=64,
        max_context_length=131072,
        capabilities=list(RequestType),
    ),
    "llama-3-405b": ModelSpec(
        name="llama-3-405b",
        params_billions=405.0,
        num_layers=126,
        d_model=16384,
        n_heads=128,
        max_context_length=131072,
        capabilities=list(RequestType),
    ),
}


# =============================================================================
# Request Model (§2.2)
# =============================================================================

class InferenceRequest(BaseModel):
    """
    A single LLM inference request entering the scheduling queue.

    Reference: §2.2 Request Data Structure.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    arrival_time: float = 0.0
    prompt_tokens: int = 512
    estimated_output_tokens: int = 128
    request_type: RequestType = RequestType.CHAT
    reasoning_depth: ReasoningDepth = ReasoningDepth.DIRECT
    slo_tier: SLOTier = SLOTier.P1
    ttft_budget_ms: float = 500.0
    tpot_budget_ms: float = 50.0
    streaming: bool = False
    session_id: Optional[str] = None
    prefix_hash: Optional[str] = None
    model_affinity: ModelAffinity = ModelAffinity.ANY
    quality_sensitive: bool = False

    # Runtime tracking (set during simulation)
    actual_ttft_ms: Optional[float] = None
    actual_tpot_ms: Optional[float] = None
    assigned_replica_id: Optional[str] = None
    completion_time: Optional[float] = None
    tokens_generated: int = 0
    prefill_start_time: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.estimated_output_tokens

    @property
    def current_wait_time_ms(self) -> float:
        """Must be set externally based on current sim time."""
        return 0.0  # Computed dynamically in sim

    def compute_prefix_hash(self, prefix_tokens: int = 64) -> str:
        """Generate a hash for the first prefix_tokens of the prompt."""
        content = f"{self.request_type}:{self.prompt_tokens}:{prefix_tokens}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


# =============================================================================
# Replica Configuration (§3.3)
# =============================================================================

class ReplicaConfig(BaseModel):
    """Static configuration for a model replica. Reference: §3.3."""
    model_config = ConfigDict(frozen=True)

    replica_id: str = Field(default_factory=lambda: f"replica-{str(uuid.uuid4())[:6]}")
    model_name: str = "llama-3-7b"
    gpu_type: GPUType = GPUType.A100_80GB
    num_gpus: int = 1
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    max_batch_size: int = 64
    supports_streaming: bool = True
    capabilities: List[RequestType] = Field(default_factory=lambda: list(RequestType))
    speculative_decoding: Optional[SpecDecConfig] = None
    chunk_size: int = 512  # Chunked prefill chunk size


# =============================================================================
# Configuration Models (§2.4, §7.1, §6.1)
# =============================================================================

class AutoscaleConfig(BaseModel):
    """Autoscaling parameters for the cluster."""
    enabled: bool = False
    min_replicas_per_model: int = 1
    max_replicas_total: int = 12
    cooldown_s: float = 60.0
    startup_delay_s: float = 30.0


class WorkloadConfig(BaseModel):
    """Workload generator configuration. Reference: §2.4."""

    pattern: TrafficPattern = TrafficPattern.POISSON
    base_arrival_rate: float = 5.0  # requests per second
    burst_multiplier: float = 3.0
    burst_duration_s: float = 10.0
    burst_interval_s: float = 60.0

    # Distributions (parameterised as log-normal mu, sigma)
    prompt_length_mu: float = 6.5     # ln-space mean (~665 tokens)
    prompt_length_sigma: float = 1.2
    output_length_mu: float = 5.0     # ln-space mean (~148 tokens)
    output_length_sigma: float = 1.5

    # Request type weights (must sum to ~1.0)
    request_type_weights: Dict[str, float] = Field(default_factory=lambda: {
        "CHAT": 0.5, "CODE": 0.2, "RAG": 0.2, "REASONING": 0.1,
    })

    # SLO tier weights
    slo_tier_weights: Dict[str, float] = Field(default_factory=lambda: {
        "P0": 0.1, "P1": 0.5, "P2": 0.3, "P3": 0.1,
    })

    streaming_fraction: float = 0.5
    session_fraction: float = 0.2
    prefix_overlap_prob: float = 0.1

    # Diurnal parameters
    diurnal_amplitude: float = 0.5  # fraction of base rate
    diurnal_period_s: float = 3600.0  # period in sim seconds
    diurnal_phase: float = 0.0


class ClusterConfig(BaseModel):
    """GPU cluster configuration."""

    gpu_inventory: Dict[str, int] = Field(
        default_factory=lambda: {"A100_80GB": 4},
        description="Available GPUs by type",
    )

    initial_replicas: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"model": "llama-3-7b", "gpu_type": "A100_80GB", "num_gpus": 1, "count": 2},
            {"model": "llama-3-70b", "gpu_type": "A100_80GB", "num_gpus": 2, "count": 1},
        ],
    )

    autoscaling: AutoscaleConfig = Field(default_factory=AutoscaleConfig)
    kv_cache_block_size: int = 16  # tokens per block


class EpisodeConfig(BaseModel):
    """Episode structure configuration. Reference: §7.1."""

    duration_s: float = 300.0          # simulated time
    warmup_s: float = 10.0
    step_mode: StepMode = StepMode.EVENT_DRIVEN
    fixed_timestep_ms: float = 10.0
    low_freq_interval: int = 100       # autoscale every N steps
    max_steps: int = 100_000
    budget_limit: Optional[float] = None


class RewardWeights(BaseModel):
    """Multi-objective reward weights. Reference: §6.1."""

    w_ttft: float = 1.0
    w_tpot: float = 0.5
    w_slo_violation: float = 5.0
    w_fairness: float = 0.3
    w_quality: float = 2.0
    w_cost: float = 0.1
    w_throughput: float = 0.2
    w_starvation: float = 10.0
    w_autoscale_churn: float = 1.0


# =============================================================================
# Environment Configuration (top-level)
# =============================================================================

class EnvironmentConfig(BaseModel):
    """Complete environment configuration combining all sub-configs."""

    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)


# =============================================================================
# Configuration Presets (Appendix C)
# =============================================================================

def dev_config() -> EnvironmentConfig:
    """Small-scale development configuration. Reference: Appendix C.1."""
    return EnvironmentConfig(
        workload=WorkloadConfig(
            pattern=TrafficPattern.POISSON,
            base_arrival_rate=5.0,
            request_type_weights={
                "CHAT": 0.5, "CODE": 0.2, "RAG": 0.2, "REASONING": 0.1,
            },
        ),
        cluster=ClusterConfig(
            gpu_inventory={"A100_80GB": 4},
            initial_replicas=[
                {"model": "llama-3-7b", "gpu_type": "A100_80GB", "num_gpus": 1, "count": 2},
                {"model": "llama-3-70b", "gpu_type": "A100_80GB", "num_gpus": 2, "count": 1},
            ],
        ),
        episode=EpisodeConfig(duration_s=300.0, warmup_s=10.0),
    )


def prod_config() -> EnvironmentConfig:
    """Production-scale evaluation configuration. Reference: Appendix C.2."""
    return EnvironmentConfig(
        workload=WorkloadConfig(
            pattern=TrafficPattern.DIURNAL_WITH_BURSTS,
            base_arrival_rate=50.0,
            burst_multiplier=5.0,
            request_type_weights={
                "CHAT": 0.35, "CODE": 0.15, "RAG": 0.15,
                "REASONING": 0.15, "TOOL_USE": 0.1, "VISION": 0.1,
            },
            streaming_fraction=0.7,
            session_fraction=0.4,
            prefix_overlap_prob=0.2,
        ),
        cluster=ClusterConfig(
            gpu_inventory={
                "H100_SXM": 32,
                "A100_80GB": 16,
                "L4_24GB": 8,
            },
            initial_replicas=[
                {"model": "llama-3-7b", "gpu_type": "L4_24GB", "num_gpus": 1, "count": 8},
                {"model": "llama-3-70b", "gpu_type": "A100_80GB", "num_gpus": 4, "count": 4},
                {"model": "llama-3-405b", "gpu_type": "H100_SXM", "num_gpus": 8, "count": 2},
            ],
            autoscaling=AutoscaleConfig(
                enabled=True,
                min_replicas_per_model=1,
                max_replicas_total=12,
                cooldown_s=60.0,
            ),
        ),
        episode=EpisodeConfig(duration_s=3600.0, warmup_s=60.0),
        reward_weights=RewardWeights(
            w_ttft=1.0, w_tpot=0.5, w_slo_violation=5.0,
            w_fairness=0.3, w_quality=2.0, w_cost=0.1,
            w_throughput=0.2, w_starvation=10.0, w_autoscale_churn=1.0,
        ),
    )


# =============================================================================
# OpenEnv Action / Observation Types (§5.2, §4.1)
# =============================================================================

# Constants for observation sizes
MAX_PENDING_REQUESTS = 128
MAX_REPLICAS = 16
PER_REQUEST_FEATURES = 14
PER_REPLICA_FEATURES = 16
GLOBAL_FEATURES = 20

# Chunk size mapping for chunk_size_level action
CHUNK_SIZE_MAP = {0: 256, 1: 512, 2: 1024, 3: 2048}

# GPU type mapping for new_replica_gpu_type action
GPU_TYPE_MAP = {0: GPUType.H100_SXM, 1: GPUType.A100_80GB, 2: GPUType.L4_24GB}


class SchedulingAction(_OpenEnvAction):
    """
    Action for the LLM inference scheduling agent. Reference: §5.2.

    High-frequency actions (every step):
      - request_index: Which pending request to schedule
      - replica_index: Which replica to route it to
      - batch_admission: 0=defer, 1=admit
      - chunk_size_level: 0=256, 1=512, 2=1024, 3=2048
      - speculative_decoding: 0=off, 1=on
      - use_prefix_cache: 0=bypass, 1=use

    Low-frequency actions (every N steps):
      - autoscale_small_model: 0=scale-down-2..4=scale-up-2
      - autoscale_large_model: same
      - new_replica_gpu_type: 0=H100, 1=A100, 2=L4
      - cache_eviction_policy: 0=LRU, 1=LFU, 2=urgency-weighted
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # High-frequency
    request_index: int = Field(default=0, ge=0, lt=MAX_PENDING_REQUESTS)
    replica_index: int = Field(default=0, ge=0, lt=MAX_REPLICAS)
    batch_admission: int = Field(default=1, ge=0, le=1)
    chunk_size_level: int = Field(default=1, ge=0, le=3)
    speculative_decoding: int = Field(default=0, ge=0, le=1)
    use_prefix_cache: int = Field(default=1, ge=0, le=1)

    # Low-frequency
    autoscale_small_model: int = Field(default=2, ge=0, le=4)  # 2=no-op
    autoscale_large_model: int = Field(default=2, ge=0, le=4)  # 2=no-op
    new_replica_gpu_type: int = Field(default=1, ge=0, le=2)
    cache_eviction_policy: int = Field(default=0, ge=0, le=2)


class SchedulingObservation(_OpenEnvObservation):
    """
    Observation returned to the agent. Reference: §4.1-4.4.

    All feature arrays are flattened lists for JSON serialization.
    The agent should reshape them using the known dimensions.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    # Per-request features: shape (MAX_PENDING_REQUESTS, PER_REQUEST_FEATURES) flattened
    pending_request_features: List[float] = Field(default_factory=list)
    pending_request_mask: List[int] = Field(default_factory=list)

    # Per-replica features: shape (MAX_REPLICAS, PER_REPLICA_FEATURES) flattened
    replica_features: List[float] = Field(default_factory=list)
    replica_mask: List[int] = Field(default_factory=list)

    # Global features: shape (GLOBAL_FEATURES,)
    global_features: List[float] = Field(default_factory=list)

    # Action masks for each action dimension
    action_mask: Dict[str, List[int]] = Field(default_factory=dict)

    # Info dict with human-readable metrics
    info: Dict[str, Any] = Field(default_factory=dict)
