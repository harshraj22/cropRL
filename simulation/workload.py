"""
Workload Generator.

Generates synthetic LLM inference requests following configurable traffic
patterns: Poisson, Bursty (MMPP), Diurnal, and Correlated prefix bursts.

Reference: Design document §2.3, §2.4.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..models import (
    InferenceRequest,
    ModelAffinity,
    ReasoningDepth,
    RequestType,
    SLOTier,
    TrafficPattern,
    WorkloadConfig,
)


# SLO budgets by tier (ms)
DEFAULT_SLO_BUDGETS: Dict[SLOTier, Tuple[float, float]] = {
    SLOTier.P0: (200.0, 20.0),    # TTFT, TPOT
    SLOTier.P1: (500.0, 50.0),
    SLOTier.P2: (2000.0, 100.0),
    SLOTier.P3: (10000.0, 200.0),
}


class WorkloadGenerator:
    """
    Generates LLM inference requests with configurable traffic patterns.

    The generator produces InferenceRequest objects with attributes sampled
    from configurable distributions, arriving at rates determined by the
    selected traffic pattern.
    """

    def __init__(self, config: WorkloadConfig, seed: Optional[int] = None):
        self.config = config
        self._rng = np.random.default_rng(seed)

        # Pending request queue
        self.pending_queue: Deque[InferenceRequest] = deque()

        # Traffic pattern state
        self._burst_active = False
        self._burst_end_time = 0.0
        self._last_burst_time = 0.0

        # Correlated prefix state
        self._active_prefix_hash: Optional[str] = None
        self._prefix_burst_remaining = 0

        # Session tracking
        self._active_sessions: Dict[str, int] = {}  # session_id -> request_count

        # Metrics
        self._total_generated = 0
        self._arrival_times: Deque[float] = deque(maxlen=1000)

    def generate_arrivals(self, sim_time: float, duration: float) -> List[InferenceRequest]:
        """
        Generate all requests that arrive in the interval [sim_time, sim_time + duration).

        Returns list of new requests sorted by arrival time.
        """
        new_requests = []
        t = sim_time

        while t < sim_time + duration:
            # Get current arrival rate
            rate = self._get_arrival_rate(t)
            if rate <= 0:
                t += 0.1  # skip forward
                continue

            # Sample inter-arrival time
            dt = self._rng.exponential(1.0 / rate)
            t += dt

            if t >= sim_time + duration:
                break

            # Generate the request
            request = self._sample_request(t)
            new_requests.append(request)
            self.pending_queue.append(request)
            self._total_generated += 1
            self._arrival_times.append(t)

        return new_requests

    def generate_single(self, sim_time: float) -> Tuple[Optional[InferenceRequest], float]:
        """
        Generate the next single request arrival.

        Returns:
            (request_or_None, next_arrival_time)
        """
        rate = self._get_arrival_rate(sim_time)
        if rate <= 0:
            return None, sim_time + 1.0

        dt = self._rng.exponential(1.0 / rate)
        arrival_time = sim_time + dt

        request = self._sample_request(arrival_time)
        self.pending_queue.append(request)
        self._total_generated += 1
        self._arrival_times.append(arrival_time)

        return request, arrival_time

    def _get_arrival_rate(self, t: float) -> float:
        """Compute the instantaneous arrival rate at time t."""
        config = self.config
        base_rate = config.base_arrival_rate

        if config.pattern == TrafficPattern.POISSON:
            return base_rate

        elif config.pattern == TrafficPattern.BURSTY:
            return self._bursty_rate(t, base_rate)

        elif config.pattern == TrafficPattern.DIURNAL:
            return self._diurnal_rate(t, base_rate)

        elif config.pattern == TrafficPattern.CORRELATED:
            # Same as Poisson but with prefix correlation
            return base_rate

        elif config.pattern == TrafficPattern.DIURNAL_WITH_BURSTS:
            diurnal = self._diurnal_rate(t, base_rate)
            if self._check_burst(t):
                return diurnal * config.burst_multiplier
            return diurnal

        return base_rate

    def _bursty_rate(self, t: float, base_rate: float) -> float:
        """MMPP bursty traffic model."""
        if self._check_burst(t):
            return base_rate * self.config.burst_multiplier
        return base_rate

    def _diurnal_rate(self, t: float, base_rate: float) -> float:
        """Sinusoidal diurnal traffic pattern."""
        config = self.config
        amplitude = config.diurnal_amplitude * base_rate
        rate = base_rate + amplitude * math.sin(
            2.0 * math.pi * t / config.diurnal_period_s + config.diurnal_phase
        )
        return max(0.1, rate)  # floor at 0.1 rps

    def _check_burst(self, t: float) -> bool:
        """Check/update burst state for MMPP model."""
        config = self.config

        if self._burst_active:
            if t >= self._burst_end_time:
                self._burst_active = False
                self._last_burst_time = t
            return True
        else:
            # Check if new burst should start
            time_since_last = t - self._last_burst_time
            if time_since_last > config.burst_interval_s:
                # Probability of burst onset
                if self._rng.random() < 0.3:  # ~30% chance per check
                    self._burst_active = True
                    self._burst_end_time = t + config.burst_duration_s
                    return True
            return False

    def _sample_request(self, arrival_time: float) -> InferenceRequest:
        """Sample a complete request with all attributes."""
        # Request type
        request_type = self._sample_enum(
            RequestType, self.config.request_type_weights,
        )

        # SLO tier
        slo_tier = self._sample_enum(
            SLOTier, self.config.slo_tier_weights,
        )

        # Token lengths (log-normal)
        prompt_tokens = max(16, int(self._rng.lognormal(
            self.config.prompt_length_mu, self.config.prompt_length_sigma,
        )))
        prompt_tokens = min(prompt_tokens, 131072)

        output_tokens = max(8, int(self._rng.lognormal(
            self.config.output_length_mu, self.config.output_length_sigma,
        )))
        output_tokens = min(output_tokens, 8192)

        # Reasoning depth (correlated with request type)
        reasoning_depth = self._sample_reasoning_depth(request_type)

        # SLO budgets
        ttft_budget, tpot_budget = DEFAULT_SLO_BUDGETS[slo_tier]

        # Streaming
        streaming = self._rng.random() < self.config.streaming_fraction

        # Session
        session_id = None
        if self._rng.random() < self.config.session_fraction:
            if self._active_sessions and self._rng.random() < 0.7:
                session_id = self._rng.choice(list(self._active_sessions.keys()))
                self._active_sessions[session_id] += 1
            else:
                session_id = f"session-{str(uuid.uuid4())[:6]}"
                self._active_sessions[session_id] = 1

        # Prefix hash (for correlated/cached requests)
        prefix_hash = None
        if self.config.pattern == TrafficPattern.CORRELATED:
            if self._prefix_burst_remaining > 0:
                prefix_hash = self._active_prefix_hash
                self._prefix_burst_remaining -= 1
            elif self._rng.random() < self.config.prefix_overlap_prob:
                prefix_hash = hashlib.md5(
                    f"prefix-{self._rng.integers(100)}".encode()
                ).hexdigest()[:8]
                self._active_prefix_hash = prefix_hash
                self._prefix_burst_remaining = self._rng.integers(5, 20)
        elif self._rng.random() < self.config.prefix_overlap_prob:
            prefix_hash = hashlib.md5(
                f"prefix-{self._rng.integers(50)}".encode()
            ).hexdigest()[:8]

        # Model affinity
        if request_type == RequestType.REASONING and reasoning_depth in (
            ReasoningDepth.DEEP_COT, ReasoningDepth.MULTI_STEP,
        ):
            model_affinity = ModelAffinity.LARGE_ONLY
            quality_sensitive = True
        elif request_type == RequestType.VISION:
            model_affinity = ModelAffinity.LARGE_ONLY
            quality_sensitive = True
        elif prompt_tokens < 100 and request_type == RequestType.CHAT:
            model_affinity = ModelAffinity.ANY
            quality_sensitive = False
        else:
            model_affinity = ModelAffinity.ANY
            quality_sensitive = self._rng.random() < 0.3

        return InferenceRequest(
            request_id=f"req-{self._total_generated:06d}",
            arrival_time=arrival_time,
            prompt_tokens=prompt_tokens,
            estimated_output_tokens=output_tokens,
            request_type=request_type,
            reasoning_depth=reasoning_depth,
            slo_tier=slo_tier,
            ttft_budget_ms=ttft_budget,
            tpot_budget_ms=tpot_budget,
            streaming=streaming,
            session_id=session_id,
            prefix_hash=prefix_hash,
            model_affinity=model_affinity,
            quality_sensitive=quality_sensitive,
        )

    def _sample_enum(self, enum_cls, weights: Dict[str, float]):
        """Sample from an enum using provided weights."""
        members = [m for m in enum_cls if m.value in weights]
        probs = [weights.get(m.value, 0.0) for m in members]

        if not members or sum(probs) == 0:
            return list(enum_cls)[0]

        total = sum(probs)
        probs = [p / total for p in probs]

        idx = self._rng.choice(len(members), p=probs)
        return members[idx]

    def _sample_reasoning_depth(self, req_type: RequestType) -> ReasoningDepth:
        """Sample reasoning depth correlated with request type."""
        if req_type == RequestType.REASONING:
            choices = [ReasoningDepth.SHALLOW_COT, ReasoningDepth.DEEP_COT, ReasoningDepth.MULTI_STEP]
            probs = [0.3, 0.5, 0.2]
        elif req_type == RequestType.CODE:
            choices = [ReasoningDepth.DIRECT, ReasoningDepth.SHALLOW_COT, ReasoningDepth.DEEP_COT]
            probs = [0.4, 0.4, 0.2]
        else:
            choices = [ReasoningDepth.DIRECT, ReasoningDepth.SHALLOW_COT]
            probs = [0.8, 0.2]

        idx = self._rng.choice(len(choices), p=probs)
        return choices[idx]

    def remove_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Remove a specific request from the pending queue."""
        for i, req in enumerate(self.pending_queue):
            if req.request_id == request_id:
                del self.pending_queue[i]
                return req
        return None

    def get_request_by_index(self, index: int) -> Optional[InferenceRequest]:
        """Get a request from the pending queue by index."""
        if 0 <= index < len(self.pending_queue):
            return self.pending_queue[index]
        return None

    # --- Metrics ---

    def arrival_rate(self, window_s: float, current_time: float) -> float:
        """Compute arrival rate over a sliding window."""
        cutoff = current_time - window_s
        count = sum(1 for t in self._arrival_times if t >= cutoff)
        return count / max(window_s, 0.001)

    @property
    def total_generated(self) -> int:
        return self._total_generated

    @property
    def pending_count(self) -> int:
        return len(self.pending_queue)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the generator state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pending_queue.clear()
        self._burst_active = False
        self._burst_end_time = 0.0
        self._last_burst_time = 0.0
        self._active_prefix_hash = None
        self._prefix_burst_remaining = 0
        self._active_sessions.clear()
        self._total_generated = 0
        self._arrival_times.clear()
