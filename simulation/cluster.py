"""
GPU Cluster Simulator.

Manages a collection of model replicas, GPU inventory, and autoscaling logic.

Reference: Design document §3.1, §3.2 (cluster architecture and GPU fleet).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    GPU_PROFILES,
    MODEL_SPECS,
    AutoscaleConfig,
    ClusterConfig,
    EvictionPolicy,
    GPUType,
    InferenceRequest,
    ReplicaConfig,
    RequestType,
)
from .replica import ReplicaSimulator


class ClusterSimulator:
    """
    Simulates a GPU cluster serving multiple LLM model replicas.

    Manages replica lifecycle (creation, teardown), GPU inventory allocation,
    autoscaling decisions, and aggregate cluster metrics.
    """

    def __init__(self, config: ClusterConfig, sim_time: float = 0.0):
        self.config = config
        self._sim_time = sim_time

        # GPU inventory: gpu_type -> available count
        self._gpu_inventory: Dict[str, int] = dict(config.gpu_inventory)
        self._gpu_allocated: Dict[str, int] = defaultdict(int)

        # Replicas
        self.replicas: Dict[str, ReplicaSimulator] = {}

        # Autoscaling state
        self.autoscale_config = config.autoscaling
        self._autoscale_cooldown_until = 0.0
        self._pending_scale_ups: List[Tuple[float, ReplicaConfig]] = []  # (ready_time, config)

        # Metrics
        self._autoscale_events_this_step = 0
        self._total_autoscale_events = 0
        self._cost_accumulator = 0.0

        # Spawn initial replicas
        self._spawn_initial_replicas()

    def _spawn_initial_replicas(self) -> None:
        """Create replicas from the initial configuration."""
        for replica_def in self.config.initial_replicas:
            model_name = replica_def["model"]
            gpu_type_str = replica_def["gpu_type"]
            num_gpus = replica_def.get("num_gpus", 1)
            count = replica_def.get("count", 1)

            gpu_type = GPUType(gpu_type_str)
            model_spec = MODEL_SPECS.get(model_name)
            capabilities = list(model_spec.capabilities) if model_spec else list(RequestType)

            for i in range(count):
                replica_id = f"{model_name}-{gpu_type_str}-{i}"
                config = ReplicaConfig(
                    replica_id=replica_id,
                    model_name=model_name,
                    gpu_type=gpu_type,
                    num_gpus=num_gpus,
                    capabilities=capabilities,
                    max_batch_size=self._estimate_max_batch_size(model_name, gpu_type, num_gpus),
                )

                # Allocate GPUs
                if self._allocate_gpus(gpu_type_str, num_gpus):
                    replica = ReplicaSimulator(config, sim_time=self._sim_time)
                    self.replicas[replica_id] = replica

    def _estimate_max_batch_size(
        self, model_name: str, gpu_type: GPUType, num_gpus: int,
    ) -> int:
        """Estimate max batch size based on GPU memory and model size."""
        gpu_profile = GPU_PROFILES[gpu_type]
        total_memory_gb = gpu_profile.memory_gb * num_gpus

        model_spec = MODEL_SPECS.get(model_name)
        if model_spec:
            model_size_gb = model_spec.params_billions * 2 / 1e0  # FP16 bytes -> GB rough
        else:
            model_size_gb = 14.0  # default

        available_gb = total_memory_gb - model_size_gb
        # Rough: ~0.5 GB per batch slot for average context
        max_batch = max(1, int(available_gb / 0.5))
        return min(max_batch, 256)

    def _allocate_gpus(self, gpu_type: str, count: int) -> bool:
        """Try to allocate GPUs from inventory. Returns True if successful."""
        available = self._gpu_inventory.get(gpu_type, 0) - self._gpu_allocated.get(gpu_type, 0)
        if available >= count:
            self._gpu_allocated[gpu_type] += count
            return True
        return False

    def _free_gpus(self, gpu_type: str, count: int) -> None:
        """Return GPUs to the available pool."""
        self._gpu_allocated[gpu_type] = max(0, self._gpu_allocated.get(gpu_type, 0) - count)

    @property
    def active_replicas(self) -> List[ReplicaSimulator]:
        """All currently active replicas."""
        return [r for r in self.replicas.values() if r.is_active]

    @property
    def active_replica_count(self) -> int:
        return len(self.active_replicas)

    @property
    def replica_list(self) -> List[ReplicaSimulator]:
        """All replicas (active and inactive) in deterministic order."""
        return list(self.replicas.values())

    def get_replica_by_index(self, index: int) -> Optional[ReplicaSimulator]:
        """Get a replica by its position index."""
        replicas = self.replica_list
        if 0 <= index < len(replicas):
            return replicas[index]
        return None

    def get_replicas_by_model_size(self) -> Dict[str, int]:
        """Count active replicas grouped by model size category."""
        counts = {"small": 0, "medium": 0, "large": 0}
        for replica in self.active_replicas:
            size = replica.model_spec.params_billions
            if size <= 10:
                counts["small"] += 1
            elif size <= 100:
                counts["medium"] += 1
            else:
                counts["large"] += 1
        return counts

    def run_all_iterations(self, sim_time: float) -> Tuple[float, List[InferenceRequest]]:
        """
        Run one iteration on all active replicas.

        Returns:
            (max_iteration_time, all_completed_requests)
        """
        self._sim_time = sim_time
        all_completed = []
        max_iter_time = 0.0

        for replica in self.active_replicas:
            iter_time, completed = replica.run_iteration(sim_time)
            max_iter_time = max(max_iter_time, iter_time)
            all_completed.extend(completed)

        # Process pending scale-ups
        self._process_pending_scaleups(sim_time)

        return max_iter_time, all_completed

    def apply_autoscale(
        self,
        scale_small: int,  # -2..+2 (offset from 2 = no-op)
        scale_large: int,
        new_gpu_type: GPUType,
        sim_time: float,
    ) -> int:
        """
        Apply autoscaling decisions.

        Args:
            scale_small: 0=scale-down-2, 1=scale-down-1, 2=no-op, 3=scale-up-1, 4=scale-up-2
            scale_large: same encoding
            new_gpu_type: GPU type for new replicas
            sim_time: current simulation time

        Returns:
            Number of autoscale events that occurred.
        """
        if not self.autoscale_config.enabled:
            return 0

        if sim_time < self._autoscale_cooldown_until:
            return 0

        events = 0
        delta_small = scale_small - 2
        delta_large = scale_large - 2

        if delta_small != 0:
            events += self._apply_scale_delta("small", delta_small, new_gpu_type, sim_time)

        if delta_large != 0:
            events += self._apply_scale_delta("large", delta_large, new_gpu_type, sim_time)

        if events > 0:
            self._autoscale_cooldown_until = sim_time + self.autoscale_config.cooldown_s
            self._autoscale_events_this_step += events
            self._total_autoscale_events += events

        return events

    def _apply_scale_delta(
        self, size_class: str, delta: int, new_gpu_type: GPUType, sim_time: float,
    ) -> int:
        """Apply scaling delta to a model size class."""
        events = 0

        if delta > 0:
            # Scale up
            for _ in range(abs(delta)):
                if self.active_replica_count >= self.autoscale_config.max_replicas_total:
                    break
                if self._schedule_scale_up(size_class, new_gpu_type, sim_time):
                    events += 1

        elif delta < 0:
            # Scale down
            for _ in range(abs(delta)):
                candidates = self._get_scale_down_candidates(size_class)
                if len(candidates) <= self.autoscale_config.min_replicas_per_model:
                    break
                if candidates:
                    self._remove_replica(candidates[-1], sim_time)
                    events += 1

        return events

    def _schedule_scale_up(
        self, size_class: str, gpu_type: GPUType, sim_time: float,
    ) -> bool:
        """Schedule a new replica to come online after startup delay."""
        # Determine model name for size class
        if size_class == "small":
            model_name = "llama-3-7b"
            num_gpus = 1
        elif size_class == "large":
            model_name = "llama-3-70b"
            num_gpus = 4 if gpu_type == GPUType.A100_80GB else (8 if gpu_type == GPUType.H100_SXM else 4)
        else:
            model_name = "llama-3-70b"
            num_gpus = 2

        gpu_type_str = gpu_type.value
        if not self._allocate_gpus(gpu_type_str, num_gpus):
            return False

        replica_id = f"{model_name}-{gpu_type_str}-scale-{str(uuid.uuid4())[:4]}"
        model_spec = MODEL_SPECS.get(model_name)
        config = ReplicaConfig(
            replica_id=replica_id,
            model_name=model_name,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            capabilities=list(model_spec.capabilities) if model_spec else list(RequestType),
            max_batch_size=self._estimate_max_batch_size(model_name, gpu_type, num_gpus),
        )

        ready_time = sim_time + self.autoscale_config.startup_delay_s
        self._pending_scale_ups.append((ready_time, config))
        return True

    def _process_pending_scaleups(self, sim_time: float) -> None:
        """Activate replicas that have finished their startup delay."""
        still_pending = []
        for ready_time, config in self._pending_scale_ups:
            if sim_time >= ready_time:
                replica = ReplicaSimulator(config, sim_time=sim_time)
                self.replicas[config.replica_id] = replica
            else:
                still_pending.append((ready_time, config))
        self._pending_scale_ups = still_pending

    def _get_scale_down_candidates(self, size_class: str) -> List[str]:
        """Get replica IDs eligible for scale-down, sorted by utilization (lowest first)."""
        candidates = []
        for rid, replica in self.replicas.items():
            if not replica.is_active:
                continue
            size = replica.model_spec.params_billions
            if size_class == "small" and size <= 10:
                candidates.append((replica.batch_utilization, rid))
            elif size_class == "large" and size > 10:
                candidates.append((replica.batch_utilization, rid))

        candidates.sort(key=lambda x: x[0])
        return [rid for _, rid in candidates]

    def _remove_replica(self, replica_id: str, sim_time: float) -> None:
        """Deactivate and remove a replica, freeing its GPUs."""
        replica = self.replicas.get(replica_id)
        if replica:
            replica.set_active(False)
            gpu_type_str = replica.config.gpu_type.value if isinstance(
                replica.config.gpu_type, GPUType
            ) else replica.config.gpu_type
            self._free_gpus(gpu_type_str, replica.config.num_gpus)
            del self.replicas[replica_id]

    def accumulate_cost(self, dt: float) -> float:
        """Accumulate GPU cost for the time interval dt."""
        cost = 0.0
        for replica in self.active_replicas:
            cost += replica.cost_per_second * replica.config.num_gpus * dt
        self._cost_accumulator += cost
        return cost

    @property
    def total_cost(self) -> float:
        return self._cost_accumulator

    @property
    def autoscale_cooldown_remaining(self) -> float:
        return max(0.0, self._autoscale_cooldown_until - self._sim_time)

    def get_autoscale_events_this_step(self) -> int:
        """Get and reset autoscale events counter for this step."""
        events = self._autoscale_events_this_step
        self._autoscale_events_this_step = 0
        return events

    @property
    def avg_gpu_memory_utilization(self) -> float:
        if not self.active_replicas:
            return 0.0
        return sum(r.kv_cache_utilization for r in self.active_replicas) / len(self.active_replicas)

    @property
    def avg_compute_utilization(self) -> float:
        if not self.active_replicas:
            return 0.0
        return sum(r.batch_utilization for r in self.active_replicas) / len(self.active_replicas)

    def set_cache_eviction_policy(self, policy: EvictionPolicy) -> None:
        """Set eviction policy on all replica KV caches."""
        for replica in self.active_replicas:
            replica.kv_cache.set_eviction_policy(policy)
