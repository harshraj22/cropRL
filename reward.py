"""
Multi-Objective Reward Model.

Computes the composite reward signal from multiple objectives: latency,
SLO compliance, fairness, quality, cost, throughput, and starvation prevention.

Reference: Design document §6.1 - §6.8.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .models import (
    InferenceRequest,
    ModelAffinity,
    RequestType,
    RewardWeights,
    SLOTier,
)
from .simulation.replica import ReplicaSimulator


def model_capability_score(replica: ReplicaSimulator, request: InferenceRequest) -> float:
    """
    Compute how well a replica's model matches the request's requirements.

    Returns a score in [0, 1] where 1 = perfect match.

    Reference: §6.4.
    """
    model_size = replica.model_spec.params_billions

    # Capability check
    if request.request_type not in replica.config.capabilities:
        return 0.0

    # Model size appropriateness
    if request.model_affinity == ModelAffinity.LARGE_ONLY:
        if model_size >= 70:
            return 1.0
        elif model_size >= 30:
            return 0.6
        else:
            return 0.2

    elif request.model_affinity == ModelAffinity.SMALL_ONLY:
        if model_size <= 10:
            return 1.0
        elif model_size <= 30:
            return 0.7
        else:
            return 0.3

    else:  # ANY
        # For quality-sensitive requests, prefer larger models
        if request.quality_sensitive:
            if model_size >= 70:
                return 1.0
            elif model_size >= 30:
                return 0.7
            else:
                return 0.5
        else:
            # For non-quality-sensitive, smaller is fine (cheaper)
            return 0.8

    return 0.5


class RewardModel:
    """
    Multi-objective reward computation.

    The reward is a weighted sum of per-completion and per-step components.
    All components are designed to be roughly in [-1, 1] before weighting.

    Reference: §6.1 - §6.7.
    """

    def __init__(self, weights: RewardWeights):
        self.weights = weights

        # Running statistics for normalization
        self._reward_sum = 0.0
        self._reward_sq_sum = 0.0
        self._reward_count = 0

    def compute(
        self,
        completed_requests: List[InferenceRequest],
        pending_requests: List[InferenceRequest],
        active_replicas: List[ReplicaSimulator],
        dt: float,
        autoscale_events: int,
        sim_time: float,
    ) -> float:
        """
        Compute the composite reward for one environment step.

        Args:
            completed_requests: Requests that completed this step
            pending_requests: Requests still in the queue
            active_replicas: Currently active replicas
            dt: Time elapsed in this step (seconds)
            autoscale_events: Number of autoscale events this step
            sim_time: Current simulation time

        Returns:
            The composite reward (float)
        """
        reward = 0.0

        # --- Per-completion rewards ---
        reward += self._latency_reward(completed_requests)
        reward += self._slo_violation_reward(completed_requests)
        reward += self._quality_reward(completed_requests, active_replicas)

        # --- Per-step rewards ---
        reward += self._throughput_reward(completed_requests)
        reward += self._fairness_reward(pending_requests, sim_time)
        reward += self._starvation_reward(pending_requests, sim_time)
        reward += self._cost_reward(active_replicas, dt, autoscale_events)

        # Update running stats
        self._reward_sum += reward
        self._reward_sq_sum += reward * reward
        self._reward_count += 1

        return reward

    def _latency_reward(self, completed: List[InferenceRequest]) -> float:
        """
        Latency penalty normalized by SLO budgets.

        R_latency = -w_ttft * (actual_ttft / budget) - w_tpot * (actual_tpot / budget)

        Reference: §6.2.
        """
        reward = 0.0
        for r in completed:
            if r.actual_ttft_ms is not None and r.ttft_budget_ms > 0:
                ttft_ratio = min(r.actual_ttft_ms / r.ttft_budget_ms, 3.0)
                reward -= self.weights.w_ttft * ttft_ratio

            if r.actual_tpot_ms is not None and r.tpot_budget_ms > 0:
                tpot_ratio = min(r.actual_tpot_ms / r.tpot_budget_ms, 3.0)
                reward -= self.weights.w_tpot * tpot_ratio

        return reward

    def _slo_violation_reward(self, completed: List[InferenceRequest]) -> float:
        """
        Step-function penalty for SLO violations.

        R_slo = -w_slo_violation if TTFT > budget OR TPOT > budget

        Reference: §6.2.
        """
        reward = 0.0
        for r in completed:
            violated = False
            if r.actual_ttft_ms is not None and r.actual_ttft_ms > r.ttft_budget_ms:
                violated = True
            if r.actual_tpot_ms is not None and r.actual_tpot_ms > r.tpot_budget_ms:
                violated = True
            if violated:
                reward -= self.weights.w_slo_violation

        return reward

    def _quality_reward(
        self,
        completed: List[InferenceRequest],
        replicas: List[ReplicaSimulator],
    ) -> float:
        """
        Quality match reward: how well was the request routed?

        R_quality = w_quality * (capability_score - 0.5) * 2

        Reference: §6.4.
        """
        reward = 0.0
        replica_map = {r.replica_id: r for r in replicas}

        for req in completed:
            if not req.quality_sensitive:
                continue

            replica = replica_map.get(req.assigned_replica_id)
            if replica is None:
                continue

            score = model_capability_score(replica, req)
            reward += self.weights.w_quality * (score - 0.5) * 2.0

        return reward

    def _throughput_reward(self, completed: List[InferenceRequest]) -> float:
        """
        Throughput bonus: reward for each completed request.

        R_throughput = w_throughput * num_completed

        Reference: §6.6.
        """
        return self.weights.w_throughput * len(completed)

    def _fairness_reward(
        self, pending: List[InferenceRequest], sim_time: float,
    ) -> float:
        """
        Fairness penalty using Jain's fairness index per SLO tier.

        J(tier) = (sum(waits))^2 / (n * sum(waits^2))
        R_fairness = -w_fairness * sum(1 - J(tier))

        Reference: §6.3.
        """
        reward = 0.0

        # Group by SLO tier
        tier_waits: Dict[SLOTier, List[float]] = defaultdict(list)
        for r in pending:
            wait = max(0.0, (sim_time - r.arrival_time) * 1000.0)  # ms
            tier_waits[r.slo_tier].append(wait)

        for tier, waits in tier_waits.items():
            n = len(waits)
            if n <= 1:
                continue

            sum_waits = sum(waits)
            sum_sq = sum(w * w for w in waits)

            if sum_sq > 0:
                jains = (sum_waits ** 2) / (n * sum_sq)
            else:
                jains = 1.0

            reward -= self.weights.w_fairness * (1.0 - jains)

        return reward

    def _starvation_reward(
        self, pending: List[InferenceRequest], sim_time: float,
    ) -> float:
        """
        Hard penalty for starved requests (waiting > 5x TTFT budget).

        Reference: §6.3.
        """
        starved = 0
        for r in pending:
            wait_ms = max(0.0, (sim_time - r.arrival_time) * 1000.0)
            if wait_ms > 5.0 * r.ttft_budget_ms:
                starved += 1

        return -self.weights.w_starvation * starved

    def _cost_reward(
        self,
        replicas: List[ReplicaSimulator],
        dt: float,
        autoscale_events: int,
    ) -> float:
        """
        GPU cost penalty + autoscale churn penalty.

        R_cost = -w_cost * sum(cost/s * num_gpus * dt) - w_churn * events

        Reference: §6.5.
        """
        cost = 0.0
        for replica in replicas:
            cost += replica.cost_per_second * replica.config.num_gpus * dt

        reward = -self.weights.w_cost * cost
        reward -= self.weights.w_autoscale_churn * autoscale_events

        return reward

    @property
    def reward_mean(self) -> float:
        if self._reward_count == 0:
            return 0.0
        return self._reward_sum / self._reward_count

    @property
    def reward_std(self) -> float:
        if self._reward_count < 2:
            return 1.0
        mean = self.reward_mean
        var = (self._reward_sq_sum / self._reward_count) - (mean * mean)
        return max(var ** 0.5, 1e-8)
