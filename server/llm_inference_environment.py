"""
LLM Inference Scheduling Environment — OpenEnv Server Implementation.

This is the core Environment class that implements the OpenEnv
MCPEnvironment interface. It wraps the simulation engine, reward model,
and observation builder to provide a step/reset API for RL training.

Reference: Design document §7, Appendix A.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

    _HAS_OPENENV = True
except ImportError:
    # Standalone mode: define compatible base classes
    from pydantic import BaseModel, Field, ConfigDict

    class Action(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="forbid")
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(extra="allow")
        episode_id: Optional[str] = None
        step_count: int = 0

    class MCPEnvironment:  # type: ignore[no-redef]
        """Minimal base when openenv-core is not installed."""
        def step(self, action, **kwargs):
            return self._step_impl(action, **kwargs)

    _HAS_OPENENV = False

try:
    from fastmcp import FastMCP
    _HAS_FASTMCP = True
except ImportError:
    _HAS_FASTMCP = False

from ..models import (
    EnvironmentConfig,
    SchedulingAction,
    SchedulingObservation,
    dev_config,
    prod_config,
)
from ..observation import ObservationBuilder
from ..reward import RewardModel
from ..simulation.engine import SimulationEngine


class LLMInferenceEnvironment(MCPEnvironment):
    """
    OpenEnv-compatible environment for LLM inference request scheduling.

    The agent controls request routing, batching, caching, and autoscaling
    decisions for a simulated GPU cluster serving heterogeneous LLM inference
    requests.

    MCP Tools (when FastMCP is available):
    - get_cluster_status: Current replica states and utilization
    - get_queue_status: Pending request summary
    - get_metrics: Goodput, latency percentiles, cost metrics
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialize the environment.

        Args:
            config: Environment configuration. Defaults to dev_config().
        """
        self._config = config or dev_config()

        # MCP tools (optional)
        if _HAS_OPENENV and _HAS_FASTMCP:
            mcp = FastMCP("llm_inference_env")
            self._register_mcp_tools(mcp)
            super().__init__(mcp)
        elif _HAS_OPENENV:
            super().__init__()

        # State tracking
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sim_engine: Optional[SimulationEngine] = None
        self._reward_model = RewardModel(self._config.reward_weights)
        self._obs_builder = ObservationBuilder()
        self._last_reward: float = 0.0
        self._is_reset = False

    def _register_mcp_tools(self, mcp) -> None:
        """Register MCP tools for environment introspection."""

        @mcp.tool
        def get_cluster_status() -> dict:
            """Get current cluster status including replica states."""
            if self._sim_engine is None:
                return {"error": "Environment not reset"}

            replicas = []
            for r in self._sim_engine.cluster.active_replicas:
                replicas.append({
                    "replica_id": r.replica_id,
                    "model": r.model_spec.name,
                    "gpu_type": r.config.gpu_type.value if hasattr(r.config.gpu_type, 'value') else str(r.config.gpu_type),
                    "num_gpus": r.config.num_gpus,
                    "batch_size": r.current_batch_size,
                    "max_batch_size": r.config.max_batch_size,
                    "kv_cache_util": round(r.kv_cache_utilization, 3),
                    "avg_ttft_ms": round(r.avg_ttft, 2),
                    "avg_tpot_ms": round(r.avg_tpot, 2),
                })

            return {
                "active_replicas": len(replicas),
                "total_cost": round(self._sim_engine.cluster.total_cost, 4),
                "replicas": replicas,
            }

        @mcp.tool
        def get_queue_status() -> dict:
            """Get current request queue status."""
            if self._sim_engine is None:
                return {"error": "Environment not reset"}

            pending = self._sim_engine.pending_requests
            by_tier = {}
            for r in pending:
                tier = r.slo_tier.value
                by_tier[tier] = by_tier.get(tier, 0) + 1

            return {
                "pending_count": len(pending),
                "by_tier": by_tier,
                "total_generated": self._sim_engine.workload_gen.total_generated,
            }

        @mcp.tool
        def get_metrics() -> dict:
            """Get environment performance metrics."""
            if self._sim_engine is None:
                return {"error": "Environment not reset"}

            return {
                "simulation_time": round(self._sim_engine.sim_time, 3),
                "step_count": self._sim_engine.step_count,
                "goodput": round(self._sim_engine.goodput, 4),
                "p50_ttft_ms": round(self._sim_engine.percentile_ttft(50), 2),
                "p99_ttft_ms": round(self._sim_engine.percentile_ttft(99), 2),
                "p50_tpot_ms": round(self._sim_engine.percentile_tpot(50), 2),
                "p99_tpot_ms": round(self._sim_engine.percentile_tpot(99), 2),
                "total_completed": self._sim_engine.total_completed,
                "total_cost": round(self._sim_engine.cluster.total_cost, 4),
                "slo_violations": self._sim_engine.slo_violations_total,
            }

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        config_preset: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Custom episode identifier
            config_preset: "dev" or "prod" to switch configs
            **kwargs: Additional reset parameters

        Returns:
            Initial Observation
        """
        # Allow config preset switching
        if config_preset == "prod":
            self._config = prod_config()
        elif config_preset == "dev":
            self._config = dev_config()

        # Create fresh state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Create simulation engine
        self._sim_engine = SimulationEngine(
            workload_config=self._config.workload,
            cluster_config=self._config.cluster,
            episode_config=self._config.episode,
            seed=seed,
        )
        self._sim_engine.reset(seed=seed)

        # Reset reward model
        self._reward_model = RewardModel(self._config.reward_weights)
        self._last_reward = 0.0
        self._is_reset = True

        # Build initial observation
        obs = self._build_scheduling_observation()

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": "LLM Inference Scheduling Environment ready",
                "observation": obs.model_dump(),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute one scheduling step.

        Parses the action, runs the simulation forward, computes reward,
        and returns the new observation.
        """
        if self._sim_engine is None:
            return Observation(
                done=True,
                reward=0.0,
                metadata={"error": "Environment not reset. Call reset() first."},
            )

        # Parse action
        scheduling_action = self._parse_action(action)

        # Run simulation step
        completed, dt = self._sim_engine.step(scheduling_action)

        # Compute reward
        autoscale_events = self._sim_engine.cluster.get_autoscale_events_this_step()
        reward = self._reward_model.compute(
            completed_requests=completed,
            pending_requests=self._sim_engine.pending_requests,
            active_replicas=self._sim_engine.cluster.active_replicas,
            dt=dt,
            autoscale_events=autoscale_events,
            sim_time=self._sim_engine.sim_time,
        )
        self._last_reward = reward

        # Check termination
        terminated = self._sim_engine.is_terminated
        truncated = self._sim_engine.is_truncated
        done = terminated or truncated

        # Update state
        self._state.step_count += 1

        # Build observation
        obs = self._build_scheduling_observation()

        # Build info dict
        info = self._build_info(completed, done, truncated)
        info["observation"] = obs.model_dump()

        return Observation(
            done=done,
            reward=reward,
            metadata=info,
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step. Delegates to base class for MCP routing."""
        self._state.step_count += 1

        if _HAS_OPENENV:
            return super().step(action, timeout_s=timeout_s, **kwargs)
        else:
            return self._step_impl(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step for WebSocket handler."""
        self._state.step_count += 1
        if _HAS_OPENENV:
            return await super().step_async(action, timeout_s=timeout_s, **kwargs)
        else:
            return self._step_impl(action, timeout_s=timeout_s, **kwargs)

    def _parse_action(self, action: Action) -> SchedulingAction:
        """Parse an OpenEnv Action into a SchedulingAction."""
        if isinstance(action, SchedulingAction):
            return action

        # Try to extract from action metadata or dict
        data = {}
        if hasattr(action, 'metadata') and action.metadata:
            data = action.metadata
        elif hasattr(action, 'model_dump'):
            data = action.model_dump()
        elif isinstance(action, dict):
            data = action

        # Filter to only valid SchedulingAction fields
        valid_fields = set(SchedulingAction.model_fields.keys()) - {'metadata'}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        return SchedulingAction(**filtered)

    def _build_scheduling_observation(self) -> SchedulingObservation:
        """Build the full observation from current simulation state."""
        if self._sim_engine is None:
            return SchedulingObservation()

        engine = self._sim_engine
        return self._obs_builder.build(
            workload_gen=engine.workload_gen,
            cluster=engine.cluster,
            sim_time=engine.sim_time,
            step_count=engine.step_count,
            episode_duration=self._config.episode.duration_s,
            goodput_1min=engine.goodput_window(60.0),
            goodput_5min=engine.goodput_window(300.0),
            p99_ttft=engine.percentile_ttft(99),
            p99_tpot=engine.percentile_tpot(99),
            cache_hit_rate=engine.cluster.avg_gpu_memory_utilization,
            slo_violation_rate=1.0 - engine.goodput,
        )

    def _build_info(
        self,
        completed: list,
        done: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        """Build the info dict with evaluation metrics."""
        engine = self._sim_engine
        if engine is None:
            return {}

        return {
            "simulation_time": round(engine.sim_time, 4),
            "step_count": engine.step_count,
            "requests_completed_total": engine.total_completed,
            "requests_dropped_total": engine.total_dropped,
            "goodput": round(engine.goodput, 4),
            "p50_ttft": round(engine.percentile_ttft(50), 2),
            "p99_ttft": round(engine.percentile_ttft(99), 2),
            "p50_tpot": round(engine.percentile_tpot(50), 2),
            "p99_tpot": round(engine.percentile_tpot(99), 2),
            "total_cost": round(engine.cluster.total_cost, 4),
            "active_replicas": engine.cluster.active_replica_count,
            "active_replicas_by_size": engine.cluster.get_replicas_by_model_size(),
            "cache_hit_rate": round(engine.cluster.avg_gpu_memory_utilization, 4),
            "slo_violations": engine.slo_violations_total,
            "pending_requests": engine.workload_gen.pending_count,
            "terminated": done and not truncated,
            "truncated": truncated,
        }

    @property
    def state(self) -> State:
        """Current environment state."""
        return self._state
