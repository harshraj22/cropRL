"""
Model Replica Simulator.

Simulates a single LLM model replica running on one or more GPUs, including:
- Continuous batching with iteration-level scheduling
- Chunked prefill (Sarathi-Serve style)
- Speculative decoding
- KV cache management
- Latency estimation based on GPU hardware profiles

Reference: Design document §3.3 - §3.6.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from ..models import (
    GPU_PROFILES,
    MODEL_SPECS,
    CHUNK_SIZE_MAP,
    EvictionPolicy,
    GPUType,
    InferenceRequest,
    ModelSpec,
    ReplicaConfig,
    RequestType,
    SpecDecConfig,
)
from .kv_cache import KVCacheSimulator


@dataclass
class BatchSlot:
    """A request currently in the active batch."""
    request: InferenceRequest
    phase: str = "prefill"   # "prefill" or "decode"
    prefill_chunks_remaining: int = 0
    tokens_decoded: int = 0
    prefill_start_time: float = 0.0
    first_token_time: Optional[float] = None


class ReplicaSimulator:
    """
    Simulates a single model replica with continuous batching.

    The replica runs a loop of batch iterations. Each iteration processes:
    1. Prefill chunks for new/ongoing prefill requests
    2. One decode token for each active decode request

    The iteration time is modeled as max(compute_time, memory_time)
    following the roofline model from §3.5.
    """

    def __init__(self, config: ReplicaConfig, sim_time: float = 0.0):
        self.config = config
        self.replica_id = config.replica_id

        # GPU profile
        gpu_type = GPUType(config.gpu_type) if isinstance(config.gpu_type, str) else config.gpu_type
        self.gpu_profile = GPU_PROFILES[gpu_type]

        # Model spec
        self.model_spec = MODEL_SPECS.get(config.model_name)
        if self.model_spec is None:
            # Create a generic spec
            self.model_spec = ModelSpec(
                name=config.model_name,
                params_billions=7.0,
                num_layers=32,
                d_model=4096,
                n_heads=32,
            )

        # KV Cache
        kv_memory_bytes = self._compute_kv_cache_memory()
        bytes_per_token = (
            2  # K and V
            * self.model_spec.num_layers
            * self.model_spec.d_head
            * self.model_spec.n_heads
            * self.model_spec.dtype_bytes
        )
        self.kv_cache = KVCacheSimulator(
            total_memory_bytes=kv_memory_bytes,
            block_size=16,
            bytes_per_token=bytes_per_token,
        )

        # Active batch
        self.active_batch: Dict[str, BatchSlot] = {}
        self.local_queue: Deque[InferenceRequest] = deque()

        # Speculative decoding
        self.spec_dec_enabled = config.speculative_decoding is not None
        self.spec_dec_config = config.speculative_decoding

        # Chunk size for chunked prefill
        self.chunk_size = config.chunk_size

        # Metrics (rolling windows)
        self._recent_ttft: Deque[float] = deque(maxlen=100)
        self._recent_tpot: Deque[float] = deque(maxlen=100)
        self._recent_cache_hits: Deque[bool] = deque(maxlen=100)
        self._completed_requests: List[InferenceRequest] = []

        # State
        self._sim_time = sim_time
        self._total_iterations = 0
        self._is_active = True

    def _compute_kv_cache_memory(self) -> int:
        """
        Compute available memory for KV cache after model weights and activations.

        KV_cache_memory = GPU_memory - model_weights - activation_memory - overhead
        """
        total_gpu_memory = self.gpu_profile.memory_gb * 1e9 * self.config.num_gpus

        # Model weights: params * dtype_bytes
        model_weight_bytes = self.model_spec.params_billions * 1e9 * self.model_spec.dtype_bytes

        # Activation memory estimate: ~10% of model weights
        activation_bytes = model_weight_bytes * 0.1

        # Overhead: ~5% of total
        overhead_bytes = total_gpu_memory * 0.05

        kv_memory = total_gpu_memory - model_weight_bytes - activation_bytes - overhead_bytes
        return max(int(kv_memory), 1024 * 1024)  # At least 1 MB

    @property
    def current_batch_size(self) -> int:
        return len(self.active_batch)

    @property
    def batch_utilization(self) -> float:
        return self.current_batch_size / max(1, self.config.max_batch_size)

    @property
    def kv_cache_utilization(self) -> float:
        return self.kv_cache.utilization

    @property
    def queue_depth(self) -> int:
        return len(self.local_queue)

    @property
    def prefill_ratio(self) -> float:
        """Fraction of active batch in prefill phase."""
        if not self.active_batch:
            return 0.0
        prefill_count = sum(1 for s in self.active_batch.values() if s.phase == "prefill")
        return prefill_count / len(self.active_batch)

    @property
    def avg_ttft(self) -> float:
        if not self._recent_ttft:
            return 0.0
        return sum(self._recent_ttft) / len(self._recent_ttft)

    @property
    def avg_tpot(self) -> float:
        if not self._recent_tpot:
            return 0.0
        return sum(self._recent_tpot) / len(self._recent_tpot)

    @property
    def cache_hit_rate(self) -> float:
        if not self._recent_cache_hits:
            return 0.0
        return sum(1 for h in self._recent_cache_hits if h) / len(self._recent_cache_hits)

    @property
    def cost_per_second(self) -> float:
        return self.gpu_profile.cost_per_second

    def can_admit(self, request: InferenceRequest) -> bool:
        """Check if this replica can admit a request into the batch."""
        if not self._is_active:
            return False
        if self.current_batch_size >= self.config.max_batch_size:
            return False

        # Check capability
        if request.request_type not in self.config.capabilities:
            return False

        # Check model affinity
        from ..models import ModelAffinity
        model_size = self.model_spec.params_billions
        if request.model_affinity == ModelAffinity.SMALL_ONLY and model_size > 20:
            return False
        if request.model_affinity == ModelAffinity.LARGE_ONLY and model_size < 20:
            return False

        # Check KV cache space
        if not self.kv_cache.can_allocate(request.total_tokens, request.prefix_hash):
            return False

        return True

    def admit_request(
        self,
        request: InferenceRequest,
        sim_time: float,
        use_prefix_cache: bool = True,
        chunk_size_override: Optional[int] = None,
    ) -> bool:
        """
        Admit a request into the active batch for processing.

        Returns True if admitted successfully.
        """
        if not self.can_admit(request):
            return False

        self._sim_time = sim_time

        # Allocate KV cache
        prefix_hash = request.prefix_hash if use_prefix_cache else None
        allocated_blocks, cached_tokens = self.kv_cache.allocate(
            request.request_id, request.total_tokens, prefix_hash, sim_time,
        )
        self._recent_cache_hits.append(cached_tokens > 0)

        # Determine prefill chunks
        chunk_size = chunk_size_override or self.chunk_size
        effective_prefill_tokens = max(0, request.prompt_tokens - cached_tokens)
        prefill_chunks = max(1, math.ceil(effective_prefill_tokens / chunk_size))

        # Create batch slot
        slot = BatchSlot(
            request=request,
            phase="prefill",
            prefill_chunks_remaining=prefill_chunks,
            prefill_start_time=sim_time,
        )
        self.active_batch[request.request_id] = slot

        request.assigned_replica_id = self.replica_id
        request.prefill_start_time = sim_time

        return True

    def run_iteration(self, sim_time: float) -> Tuple[float, List[InferenceRequest]]:
        """
        Execute one batch iteration.

        Returns:
            (iteration_duration_s, completed_requests)
        """
        if not self.active_batch:
            return 0.001, []  # 1ms idle tick

        self._sim_time = sim_time
        self._total_iterations += 1

        completed = []
        prefill_tokens_this_iter = 0
        decode_tokens_this_iter = 0

        # Process each slot in the batch
        for req_id, slot in list(self.active_batch.items()):
            if slot.phase == "prefill":
                # Process one prefill chunk
                chunk_tokens = min(self.chunk_size, slot.request.prompt_tokens)
                prefill_tokens_this_iter += chunk_tokens
                slot.prefill_chunks_remaining -= 1

                if slot.prefill_chunks_remaining <= 0:
                    # Prefill complete -> switch to decode
                    slot.phase = "decode"
                    ttft_ms = (sim_time - slot.prefill_start_time) * 1000
                    # Add iteration time estimate for TTFT
                    ttft_ms += self._estimate_iteration_time(chunk_tokens, 0) * 1000
                    slot.request.actual_ttft_ms = ttft_ms
                    slot.first_token_time = sim_time
                    self._recent_ttft.append(ttft_ms)

            elif slot.phase == "decode":
                # Generate one token (or k tokens with spec dec)
                tokens_per_step = 1
                if self.spec_dec_enabled and self.spec_dec_config:
                    # Speculative decoding: effective tokens per iteration
                    k = self.spec_dec_config.num_speculative_tokens
                    accept_rate = self.spec_dec_config.acceptance_rate
                    tokens_per_step = max(1, int(1 + k * accept_rate))

                slot.tokens_decoded += tokens_per_step
                decode_tokens_this_iter += 1  # one decode step per slot

                # Check completion
                if slot.tokens_decoded >= slot.request.estimated_output_tokens:
                    # Request complete
                    slot.request.tokens_generated = slot.tokens_decoded
                    slot.request.completion_time = sim_time

                    # Calculate TPOT
                    if slot.first_token_time and slot.tokens_decoded > 1:
                        decode_duration = (sim_time - slot.first_token_time) * 1000
                        tpot_ms = decode_duration / max(1, slot.tokens_decoded - 1)
                    else:
                        tpot_ms = 0.0
                    slot.request.actual_tpot_ms = tpot_ms
                    self._recent_tpot.append(tpot_ms)

                    completed.append(slot.request)
                    # Free KV cache
                    self.kv_cache.free(req_id)
                    del self.active_batch[req_id]

        # Compute iteration time
        iteration_time = self._estimate_iteration_time(
            prefill_tokens_this_iter, decode_tokens_this_iter,
        )

        # Update TPOT for decode requests based on actual iteration time
        for slot in self.active_batch.values():
            if slot.phase == "decode" and slot.first_token_time:
                # TPOT approximation from iteration time
                pass  # Already tracked at completion

        self._completed_requests.extend(completed)
        return iteration_time, completed

    def _estimate_iteration_time(
        self, prefill_tokens: int, decode_tokens: int,
    ) -> float:
        """
        Estimate iteration duration using the roofline model.

        t_iteration = max(t_compute, t_memory)

        Reference: §3.5 Iteration Timing Model.
        """
        if prefill_tokens == 0 and decode_tokens == 0:
            return 0.001  # 1ms idle

        spec = self.model_spec
        gpu = self.gpu_profile

        # Compute bound: prefill tokens
        # t_compute = (prefill_tokens * 2 * model_params) / GPU_TFLOPS
        if prefill_tokens > 0:
            flops = prefill_tokens * 2 * spec.params_billions * 1e9
            tflops = gpu.compute_tflops_fp16 * 1e12 * self.config.num_gpus
            t_compute = flops / max(1, tflops)
        else:
            t_compute = 0.0

        # Memory bound: decode tokens
        # t_memory = (decode_tokens * 2 * n_layers * d_model * sizeof(dtype)) / mem_bandwidth
        if decode_tokens > 0:
            bytes_per_decode = (
                decode_tokens
                * 2  # K and V
                * spec.num_layers
                * spec.d_model
                * spec.dtype_bytes
            )
            bandwidth = gpu.memory_bandwidth_gbps * 1e9 * self.config.num_gpus
            t_memory = bytes_per_decode / max(1, bandwidth)
        else:
            t_memory = 0.0

        # Speculative decoding overhead
        if self.spec_dec_enabled and self.spec_dec_config and decode_tokens > 0:
            draft_overhead = (
                self.spec_dec_config.draft_model_latency_ms
                * self.spec_dec_config.num_speculative_tokens
                / 1000.0
            )
            t_memory += draft_overhead

        return max(t_compute, t_memory, 0.0001)  # min 0.1ms

    def estimate_next_slot_time(self) -> float:
        """Estimate when the next batch slot will open."""
        if self.current_batch_size == 0:
            return 0.0

        # Rough estimate: time for the shortest remaining request to complete
        min_remaining = float("inf")
        for slot in self.active_batch.values():
            if slot.phase == "decode":
                remaining = slot.request.estimated_output_tokens - slot.tokens_decoded
            else:
                remaining = slot.request.estimated_output_tokens + (
                    slot.prefill_chunks_remaining * self.chunk_size
                )
            min_remaining = min(min_remaining, remaining)

        # Estimate time per token during decode
        iter_time = self._estimate_iteration_time(0, max(1, self.current_batch_size))
        return iter_time * min_remaining

    def drain_completed(self) -> List[InferenceRequest]:
        """Return and clear the list of completed requests."""
        completed = list(self._completed_requests)
        self._completed_requests.clear()
        return completed

    def supports_request_type(self, req_type: RequestType) -> bool:
        """Check if this replica can serve a given request type."""
        return req_type in self.config.capabilities

    def set_active(self, active: bool) -> None:
        self._is_active = active

    @property
    def is_active(self) -> bool:
        return self._is_active
