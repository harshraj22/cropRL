"""
Observation Builder.

Constructs the structured observation (feature tensors + masks) from raw
simulation state, ready for the RL agent to consume.

Reference: Design document §4.1 - §4.5.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .models import (
    MAX_PENDING_REQUESTS,
    MAX_REPLICAS,
    PER_REPLICA_FEATURES,
    PER_REQUEST_FEATURES,
    GLOBAL_FEATURES,
    GPUType,
    InferenceRequest,
    ModelAffinity,
    RequestType,
    SchedulingObservation,
    SLOTier,
)
from .simulation.cluster import ClusterSimulator
from .simulation.replica import ReplicaSimulator
from .simulation.workload import WorkloadGenerator


# Normalisation constants
MAX_CONTEXT_LENGTH = 131072
MAX_OUTPUT_LENGTH = 8192
MAX_WAIT_TIME_MS = 60000.0    # 60 seconds
TTFT_NORM = 5000.0            # 5 seconds
TPOT_NORM = 500.0             # 500 ms
SLOT_TIME_NORM = 10.0         # 10 seconds
MAX_MODEL_SIZE_B = 500.0      # 500 billion params
NUM_GPU_TYPES = 3
MAX_GPUS_PER_REPLICA = 8
MAX_QUEUE_DEPTH = 256
NUM_TIERS = 4
MAX_ARRIVAL_RATE = 200.0      # normalisation for arrival rate


class ObservationBuilder:
    """
    Builds the RL observation from simulation state.

    The observation is a structured set of feature tensors:
    - Per-request features: (MAX_PENDING, 14) with mask
    - Per-replica features: (MAX_REPLICAS, 16) with mask
    - Global features: (20,)
    - Action masks for each action dimension
    """

    def build(
        self,
        workload_gen: WorkloadGenerator,
        cluster: ClusterSimulator,
        sim_time: float,
        step_count: int,
        episode_duration: float,
        goodput_1min: float = 1.0,
        goodput_5min: float = 1.0,
        p99_ttft: float = 0.0,
        p99_tpot: float = 0.0,
        cache_hit_rate: float = 0.0,
        slo_violation_rate: float = 0.0,
    ) -> SchedulingObservation:
        """Build the complete observation."""
        pending = list(workload_gen.pending_queue)
        replicas = cluster.replica_list

        # Per-request features
        req_features, req_mask = self._encode_requests(pending, sim_time)

        # Per-replica features
        rep_features, rep_mask = self._encode_replicas(replicas)

        # Global features
        global_feats = self._encode_global(
            workload_gen=workload_gen,
            cluster=cluster,
            sim_time=sim_time,
            step_count=step_count,
            episode_duration=episode_duration,
            goodput_1min=goodput_1min,
            goodput_5min=goodput_5min,
            p99_ttft=p99_ttft,
            p99_tpot=p99_tpot,
            cache_hit_rate=cache_hit_rate,
            slo_violation_rate=slo_violation_rate,
        )

        # Action masks
        action_mask = self._build_action_masks(pending, replicas)

        return SchedulingObservation(
            done=False,
            reward=None,
            pending_request_features=req_features,
            pending_request_mask=req_mask,
            replica_features=rep_features,
            replica_mask=rep_mask,
            global_features=global_feats,
            action_mask=action_mask,
        )

    def _encode_requests(
        self, pending: List[InferenceRequest], sim_time: float,
    ) -> tuple[List[float], List[int]]:
        """
        Encode pending requests as a flattened feature matrix.

        Each request -> 14-dim normalized vector (§4.2).
        """
        features = []
        mask = []

        for i in range(MAX_PENDING_REQUESTS):
            if i < len(pending):
                req = pending[i]
                feat = self._encode_single_request(req, sim_time)
                features.extend(feat)
                mask.append(1)
            else:
                features.extend([0.0] * PER_REQUEST_FEATURES)
                mask.append(0)

        return features, mask

    def _encode_single_request(
        self, req: InferenceRequest, sim_time: float,
    ) -> List[float]:
        """Encode a single request as a 14-dim normalized vector."""
        # Feature 0: prompt_tokens / MAX_CONTEXT_LENGTH
        f0 = min(req.prompt_tokens / MAX_CONTEXT_LENGTH, 1.0)

        # Feature 1: estimated_output_tokens / MAX_OUTPUT_LENGTH
        f1 = min(req.estimated_output_tokens / MAX_OUTPUT_LENGTH, 1.0)

        # Features 2-7: request_type one-hot (6 types)
        type_onehot = [0.0] * 6
        type_idx = list(RequestType).index(req.request_type) if req.request_type in RequestType else 0
        if type_idx < 6:
            type_onehot[type_idx] = 1.0

        # Feature 8: slo_tier (ordinal)
        tier_map = {SLOTier.P0: 0, SLOTier.P1: 1, SLOTier.P2: 2, SLOTier.P3: 3}
        f8 = tier_map.get(req.slo_tier, 1) / NUM_TIERS

        # Feature 9: wait_time / MAX_WAIT_TIME
        wait_ms = max(0.0, (sim_time - req.arrival_time) * 1000.0)
        f9 = min(wait_ms / MAX_WAIT_TIME_MS, 1.0)

        # Feature 10: streaming flag
        f10 = 1.0 if req.streaming else 0.0

        # Feature 11: session_bound flag
        f11 = 1.0 if req.session_id is not None else 0.0

        # Feature 12: quality_sensitive flag
        f12 = 1.0 if req.quality_sensitive else 0.0

        # Feature 13: ttft_slack = (budget - elapsed) / budget
        if req.ttft_budget_ms > 0:
            f13 = max(0.0, min(1.0, (req.ttft_budget_ms - wait_ms) / req.ttft_budget_ms))
        else:
            f13 = 0.0

        return [f0, f1] + type_onehot[:5] + [f8, f9, f10, f11, f12, f13]

    def _encode_replicas(
        self, replicas: List[ReplicaSimulator],
    ) -> tuple[List[float], List[int]]:
        """
        Encode replicas as a flattened feature matrix.

        Each replica -> 16-dim normalized vector (§4.3).
        """
        features = []
        mask = []

        for i in range(MAX_REPLICAS):
            if i < len(replicas) and replicas[i].is_active:
                feat = self._encode_single_replica(replicas[i])
                features.extend(feat)
                mask.append(1)
            else:
                features.extend([0.0] * PER_REPLICA_FEATURES)
                mask.append(0)

        return features, mask

    def _encode_single_replica(self, replica: ReplicaSimulator) -> List[float]:
        """Encode a single replica as a 16-dim normalized vector."""
        spec = replica.model_spec
        config = replica.config

        # Feature 0: model_size / MAX_MODEL_SIZE_B
        f0 = min(spec.params_billions / MAX_MODEL_SIZE_B, 1.0)

        # Feature 1: gpu_type ordinal
        gpu_type_map = {GPUType.H100_SXM: 0, GPUType.A100_80GB: 1, GPUType.L4_24GB: 2}
        gpu_type = config.gpu_type if isinstance(config.gpu_type, GPUType) else GPUType(config.gpu_type)
        f1 = gpu_type_map.get(gpu_type, 1) / NUM_GPU_TYPES

        # Feature 2: num_gpus
        f2 = min(config.num_gpus / MAX_GPUS_PER_REPLICA, 1.0)

        # Feature 3: current_batch_size / max_batch_size
        f3 = replica.batch_utilization

        # Feature 4: kv_cache_utilization
        f4 = replica.kv_cache_utilization

        # Feature 5: queue_depth
        f5 = min(replica.queue_depth / MAX_QUEUE_DEPTH, 1.0)

        # Feature 6: avg_ttft_recent
        f6 = min(replica.avg_ttft / TTFT_NORM, 1.0)

        # Feature 7: avg_tpot_recent
        f7 = min(replica.avg_tpot / TPOT_NORM, 1.0)

        # Feature 8: prefill_ratio
        f8 = replica.prefill_ratio

        # Feature 9: speculative_decoding_enabled
        f9 = 1.0 if replica.spec_dec_enabled else 0.0

        # Feature 10: estimated_next_slot_time
        f10 = min(replica.estimate_next_slot_time() / SLOT_TIME_NORM, 1.0)

        # Feature 11: prefix_cache_hit_rate
        f11 = replica.cache_hit_rate

        # Feature 12: supports_streaming
        f12 = 1.0 if config.supports_streaming else 0.0

        # Features 13-15: capability flags (code, rag, reasoning)
        f13 = 1.0 if RequestType.CODE in config.capabilities else 0.0
        f14 = 1.0 if RequestType.RAG in config.capabilities else 0.0
        f15 = 1.0 if RequestType.REASONING in config.capabilities else 0.0

        return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]

    def _encode_global(
        self,
        workload_gen: WorkloadGenerator,
        cluster: ClusterSimulator,
        sim_time: float,
        step_count: int,
        episode_duration: float,
        goodput_1min: float,
        goodput_5min: float,
        p99_ttft: float,
        p99_tpot: float,
        cache_hit_rate: float,
        slo_violation_rate: float,
    ) -> List[float]:
        """
        Encode global cluster state as a 20-dim vector (§4.4).
        """
        pending_count = workload_gen.pending_count

        # Feature 0: total pending / MAX_PENDING
        f0 = min(pending_count / MAX_PENDING_REQUESTS, 1.0)

        # Features 1-3: arrival rates (1s, 10s, 60s windows)
        f1 = min(workload_gen.arrival_rate(1.0, sim_time) / MAX_ARRIVAL_RATE, 1.0)
        f2 = min(workload_gen.arrival_rate(10.0, sim_time) / MAX_ARRIVAL_RATE, 1.0)
        f3 = min(workload_gen.arrival_rate(60.0, sim_time) / MAX_ARRIVAL_RATE, 1.0)

        # Features 4-5: cluster utilization
        f4 = cluster.avg_gpu_memory_utilization
        f5 = cluster.avg_compute_utilization

        # Features 6-8: active replicas by model size
        by_size = cluster.get_replicas_by_model_size()
        max_replicas = max(cluster.active_replica_count, 1)
        f6 = by_size.get("small", 0) / max_replicas
        f7 = by_size.get("medium", 0) / max_replicas
        f8 = by_size.get("large", 0) / max_replicas

        # Feature 9: autoscaler cooldown remaining
        if cluster.autoscale_config.cooldown_s > 0:
            f9 = cluster.autoscale_cooldown_remaining / cluster.autoscale_config.cooldown_s
        else:
            f9 = 0.0

        # Features 10-11: goodput
        f10 = goodput_1min
        f11 = goodput_5min

        # Feature 12: cost accumulator (normalized by episode budget)
        budget = episode_duration * 10.0  # rough budget estimate
        f12 = min(cluster.total_cost / max(budget, 1.0), 1.0)

        # Feature 13: avg queue wait time
        pending = list(workload_gen.pending_queue)
        if pending:
            avg_wait = sum((sim_time - r.arrival_time) * 1000.0 for r in pending) / len(pending)
            f13 = min(avg_wait / MAX_WAIT_TIME_MS, 1.0)
        else:
            f13 = 0.0

        # Features 14-15: p99 latencies
        f14 = min(p99_ttft / TTFT_NORM, 1.0)
        f15 = min(p99_tpot / TPOT_NORM, 1.0)

        # Feature 16: simulation time progress
        f16 = min(sim_time / max(episode_duration, 1.0), 1.0)

        # Feature 17: global cache hit rate
        f17 = cache_hit_rate

        # Feature 18: streaming request fraction in queue
        if pending:
            f18 = sum(1 for r in pending if r.streaming) / len(pending)
        else:
            f18 = 0.0

        # Feature 19: SLO violation rate
        f19 = slo_violation_rate

        return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
                f10, f11, f12, f13, f14, f15, f16, f17, f18, f19]

    def _build_action_masks(
        self,
        pending: List[InferenceRequest],
        replicas: List[ReplicaSimulator],
    ) -> Dict[str, List[int]]:
        """
        Build validity masks for each action dimension.

        Reference: §5.3.
        """
        # Request index mask: 1 if a valid request exists at that index
        request_mask = [0] * MAX_PENDING_REQUESTS
        for i in range(min(len(pending), MAX_PENDING_REQUESTS)):
            request_mask[i] = 1

        # Replica index mask: 1 if an active replica exists at that index
        replica_mask = [0] * MAX_REPLICAS
        for i in range(min(len(replicas), MAX_REPLICAS)):
            if replicas[i].is_active:
                replica_mask[i] = 1

        # Batch admission: 1 if there are pending requests and active replicas
        batch_admission_mask = [1, 1]  # defer is always valid
        if not pending or not any(r.is_active for r in replicas):
            batch_admission_mask[1] = 0  # can't admit if nothing to admit

        # Speculative decoding: depends on selected replica (permissive mask)
        spec_dec_mask = [1, 1]  # both valid; validated at step time

        # Autoscale masks (permissive; cluster validates constraints)
        autoscale_small = [1, 1, 1, 1, 1]  # all options
        autoscale_large = [1, 1, 1, 1, 1]

        return {
            "request_index": request_mask,
            "replica_index": replica_mask,
            "batch_admission": batch_admission_mask,
            "chunk_size_level": [1, 1, 1, 1],
            "speculative_decoding": spec_dec_mask,
            "use_prefix_cache": [1, 1],
            "autoscale_small_model": autoscale_small,
            "autoscale_large_model": autoscale_large,
            "new_replica_gpu_type": [1, 1, 1],
            "cache_eviction_policy": [1, 1, 1],
        }
