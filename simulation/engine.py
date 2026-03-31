"""
Simulation Engine.

The top-level engine orchestrating the discrete-event simulation. It coordinates
the workload generator, cluster simulator, and the scheduling decision loop.

Rather than using SimPy directly (to avoid a complex co-routine based event loop
that would be hard to integrate with the step-by-step RL API), this engine
implements a simpler time-stepping simulation that advances the simulation clock
at each agent step.

Reference: Design document §3.8, §7.2.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..models import (
    CHUNK_SIZE_MAP,
    ClusterConfig,
    EpisodeConfig,
    EvictionPolicy,
    GPU_TYPE_MAP,
    GPUType,
    InferenceRequest,
    SchedulingAction,
    StepMode,
    WorkloadConfig,
)
from .cluster import ClusterSimulator
from .workload import WorkloadGenerator


class SimulationEngine:
    """
    Top-level simulation engine for the LLM inference scheduling environment.

    Manages the simulation clock, coordinates workload generation and cluster
    iteration processing, and provides the step/reset interface used by the
    OpenEnv environment server.
    """

    def __init__(
        self,
        workload_config: WorkloadConfig,
        cluster_config: ClusterConfig,
        episode_config: EpisodeConfig,
        seed: Optional[int] = None,
    ):
        self.workload_config = workload_config
        self.cluster_config = cluster_config
        self.episode_config = episode_config

        # Random state
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Components
        self.workload_gen = WorkloadGenerator(workload_config, seed=seed)
        self.cluster = ClusterSimulator(cluster_config, sim_time=0.0)

        # Simulation clock
        self.sim_time: float = 0.0
        self.step_count: int = 0

        # Completed requests tracking
        self._completed_this_step: List[InferenceRequest] = []
        self._all_completed: List[InferenceRequest] = []
        self._all_dropped: List[InferenceRequest] = []

        # Rolling metrics windows
        self._recent_completed: Deque[InferenceRequest] = deque(maxlen=500)
        self._slo_violations_recent: Deque[bool] = deque(maxlen=500)

        # Pre-generate initial arrivals
        self._next_arrival_time: float = 0.0
        self._warmup_complete = False

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the simulation to initial state."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        self.workload_gen = WorkloadGenerator(self.workload_config, seed=self._seed)
        self.cluster = ClusterSimulator(self.cluster_config, sim_time=0.0)

        self.sim_time = 0.0
        self.step_count = 0

        self._completed_this_step.clear()
        self._all_completed.clear()
        self._all_dropped.clear()
        self._recent_completed.clear()
        self._slo_violations_recent.clear()

        self._warmup_complete = False

        # Run warmup
        self._run_warmup()

    def _run_warmup(self) -> None:
        """
        Run warmup period with heuristic round-robin scheduling.

        During warmup, requests arrive and are scheduled using a simple
        round-robin policy. No reward is tracked.
        """
        warmup_end = self.episode_config.warmup_s
        replica_idx = 0

        while self.sim_time < warmup_end:
            # Generate arrivals for a small time window
            self.workload_gen.generate_arrivals(self.sim_time, 0.1)

            # Schedule pending requests with round-robin
            replicas = self.cluster.active_replicas
            if replicas:
                while self.workload_gen.pending_queue:
                    request = self.workload_gen.pending_queue[0]
                    target = replicas[replica_idx % len(replicas)]
                    if target.can_admit(request):
                        self.workload_gen.pending_queue.popleft()
                        target.admit_request(request, self.sim_time)
                    else:
                        break
                    replica_idx += 1

            # Run one iteration
            iter_time, completed = self.cluster.run_all_iterations(self.sim_time)
            self.cluster.accumulate_cost(iter_time)
            self.sim_time += max(iter_time, 0.001)

        self._warmup_complete = True

    def step(self, action: SchedulingAction) -> Tuple[List[InferenceRequest], float]:
        """
        Execute one scheduling step.

        1. Generate new arrivals up to current time
        2. Apply the agent's scheduling action
        3. Advance simulation (run cluster iterations)
        4. Collect completed requests

        Args:
            action: The agent's scheduling decision

        Returns:
            (completed_requests_this_step, dt)
        """
        self.step_count += 1
        self._completed_this_step.clear()

        # 1. Generate new arrivals
        if self.episode_config.step_mode == StepMode.FIXED_TIMESTEP:
            dt_arrivals = self.episode_config.fixed_timestep_ms / 1000.0
        else:
            # Event-driven: generate next few arrivals
            dt_arrivals = 0.05  # 50ms lookahead

        new_arrivals = self.workload_gen.generate_arrivals(self.sim_time, dt_arrivals)

        # 2. Apply high-frequency action: route a request
        if action.batch_admission == 1:
            self._apply_routing_action(action)

        # 3. Apply low-frequency actions (autoscaling, cache policy)
        autoscale_events = 0
        if self.step_count % self.episode_config.low_freq_interval == 0:
            autoscale_events = self._apply_low_freq_actions(action)

        # 4. Advance simulation — run iterations
        iter_time, completed = self.cluster.run_all_iterations(self.sim_time)
        dt = max(iter_time, 0.001)

        # 5. Accumulate cost
        self.cluster.accumulate_cost(dt)

        # 6. Track completed requests
        for req in completed:
            self._completed_this_step.append(req)
            self._all_completed.append(req)
            self._recent_completed.append(req)

            # Track SLO violations
            violated = False
            if req.actual_ttft_ms and req.actual_ttft_ms > req.ttft_budget_ms:
                violated = True
            if req.actual_tpot_ms and req.actual_tpot_ms > req.tpot_budget_ms:
                violated = True
            self._slo_violations_recent.append(violated)

        # 7. Advance clock
        self.sim_time += dt

        return self._completed_this_step, dt

    def _apply_routing_action(self, action: SchedulingAction) -> bool:
        """Apply the request routing / batching action."""
        # Get the selected request
        request = self.workload_gen.get_request_by_index(action.request_index)
        if request is None:
            return False

        # Get the selected replica
        replica = self.cluster.get_replica_by_index(action.replica_index)
        if replica is None:
            return False

        # Check if replica can admit
        if not replica.can_admit(request):
            return False

        # Determine chunk size
        chunk_size = CHUNK_SIZE_MAP.get(action.chunk_size_level, 512)

        # Admit the request
        success = replica.admit_request(
            request,
            sim_time=self.sim_time,
            use_prefix_cache=action.use_prefix_cache == 1,
            chunk_size_override=chunk_size,
        )

        if success:
            # Remove from pending queue
            self.workload_gen.remove_request(request.request_id)

        return success

    def _apply_low_freq_actions(self, action: SchedulingAction) -> int:
        """Apply autoscaling and cache policy actions."""
        # Cache eviction policy
        eviction_policies = [EvictionPolicy.LRU, EvictionPolicy.LFU, EvictionPolicy.URGENCY_WEIGHTED]
        policy = eviction_policies[min(action.cache_eviction_policy, 2)]
        self.cluster.set_cache_eviction_policy(policy)

        # Autoscaling
        new_gpu_type = GPU_TYPE_MAP.get(action.new_replica_gpu_type, GPUType.A100_80GB)
        events = self.cluster.apply_autoscale(
            scale_small=action.autoscale_small_model,
            scale_large=action.autoscale_large_model,
            new_gpu_type=new_gpu_type,
            sim_time=self.sim_time,
        )

        return events

    # --- Termination Checks ---

    @property
    def is_terminated(self) -> bool:
        """Check if episode should terminate (natural end)."""
        # Time limit
        if self.sim_time >= self.episode_config.duration_s:
            return True

        # Budget exhaustion
        if (self.episode_config.budget_limit is not None
                and self.cluster.total_cost >= self.episode_config.budget_limit):
            return True

        # All replicas gone
        if self.cluster.active_replica_count == 0 and self._warmup_complete:
            return True

        return False

    @property
    def is_truncated(self) -> bool:
        """Check if episode should be truncated (safety limit)."""
        return self.step_count >= self.episode_config.max_steps

    # --- Metrics ---

    @property
    def goodput(self) -> float:
        """Fraction of completed requests meeting SLO."""
        if not self._slo_violations_recent:
            return 1.0
        violations = sum(1 for v in self._slo_violations_recent if v)
        return 1.0 - (violations / len(self._slo_violations_recent))

    def goodput_window(self, window_s: float) -> float:
        """Goodput over a time window."""
        cutoff = self.sim_time - window_s
        recent = [r for r in self._all_completed if r.completion_time and r.completion_time >= cutoff]
        if not recent:
            return 1.0
        violations = sum(1 for r in recent if (
            (r.actual_ttft_ms and r.actual_ttft_ms > r.ttft_budget_ms)
            or (r.actual_tpot_ms and r.actual_tpot_ms > r.tpot_budget_ms)
        ))
        return 1.0 - (violations / len(recent))

    def percentile_ttft(self, p: float) -> float:
        """p-th percentile TTFT of recent completions (ms)."""
        ttfts = [r.actual_ttft_ms for r in self._recent_completed if r.actual_ttft_ms is not None]
        if not ttfts:
            return 0.0
        return float(np.percentile(ttfts, p))

    def percentile_tpot(self, p: float) -> float:
        """p-th percentile TPOT of recent completions (ms)."""
        tpots = [r.actual_tpot_ms for r in self._recent_completed if r.actual_tpot_ms is not None]
        if not tpots:
            return 0.0
        return float(np.percentile(tpots, p))

    @property
    def total_completed(self) -> int:
        return len(self._all_completed)

    @property
    def total_dropped(self) -> int:
        return len(self._all_dropped)

    @property
    def slo_violations_total(self) -> int:
        return sum(1 for v in self._slo_violations_recent if v)

    @property
    def completed_this_step(self) -> List[InferenceRequest]:
        return list(self._completed_this_step)

    @property
    def pending_requests(self) -> List[InferenceRequest]:
        return list(self.workload_gen.pending_queue)
