"""
LLM Inference Scheduling RL Environment.

An OpenEnv-compatible environment for training RL agents to schedule
heterogeneous LLM inference requests across a simulated GPU cluster.

Example:
    >>> from llm_inference_env import LLMInferenceEnv, SchedulingAction
    >>>
    >>> # Standalone mode (no server needed)
    >>> with LLMInferenceEnv() as env:
    ...     result = env.reset()
    ...     action = SchedulingAction(request_index=0, replica_index=0)
    ...     result = env.step(action)
    ...     print(f"Reward: {result.reward}, Done: {result.done}")
"""

from .models import (
    # Enums
    RequestType,
    ReasoningDepth,
    SLOTier,
    ModelAffinity,
    TrafficPattern,
    StepMode,
    EvictionPolicy,
    GPUType,
    # Data models
    InferenceRequest,
    GPUProfile,
    ReplicaConfig,
    SpecDecConfig,
    # Configuration
    WorkloadConfig,
    ClusterConfig,
    EpisodeConfig,
    RewardWeights,
    EnvironmentConfig,
    # Presets
    dev_config,
    prod_config,
    # OpenEnv types
    SchedulingAction,
    SchedulingObservation,
    # Constants
    MAX_PENDING_REQUESTS,
    MAX_REPLICAS,
)

from .client import LLMInferenceEnv

__all__ = [
    # Client
    "LLMInferenceEnv",
    # Action/Observation
    "SchedulingAction",
    "SchedulingObservation",
    # Enums
    "RequestType",
    "ReasoningDepth",
    "SLOTier",
    "ModelAffinity",
    "TrafficPattern",
    "StepMode",
    "EvictionPolicy",
    "GPUType",
    # Data models
    "InferenceRequest",
    "GPUProfile",
    "ReplicaConfig",
    "SpecDecConfig",
    # Configuration
    "WorkloadConfig",
    "ClusterConfig",
    "EpisodeConfig",
    "RewardWeights",
    "EnvironmentConfig",
    # Presets
    "dev_config",
    "prod_config",
    # Constants
    "MAX_PENDING_REQUESTS",
    "MAX_REPLICAS",
]
