"""
KV Cache Simulator.

Models paged KV cache allocation (PagedAttention / vLLM style), prefix caching
via a prefix trie, and eviction policies (LRU, LFU, urgency-weighted).

Reference: Design document §3.4.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..models import EvictionPolicy


@dataclass
class CacheBlock:
    """A single KV cache block (fixed number of tokens)."""
    block_id: int
    request_id: Optional[str] = None
    prefix_hash: Optional[str] = None
    last_access_time: float = 0.0
    access_count: int = 0
    is_prefix_cached: bool = False


class PrefixTrie:
    """
    Simple prefix trie for matching prompt prefix hashes.

    Maps prefix hashes to sets of block IDs that store the cached KV state
    for that prefix.
    """

    def __init__(self):
        self._entries: Dict[str, PrefixEntry] = {}

    def insert(self, prefix_hash: str, block_ids: List[int], sim_time: float) -> None:
        """Register a prefix in the cache."""
        self._entries[prefix_hash] = PrefixEntry(
            prefix_hash=prefix_hash,
            block_ids=list(block_ids),
            cached_tokens=len(block_ids) * 16,  # block_size tokens per block
            last_access=sim_time,
            access_count=1,
        )

    def lookup(self, prefix_hash: str, sim_time: float) -> Optional[PrefixEntry]:
        """Look up a prefix. Returns None if not cached."""
        entry = self._entries.get(prefix_hash)
        if entry is not None:
            entry.last_access = sim_time
            entry.access_count += 1
        return entry

    def remove(self, prefix_hash: str) -> Optional[PrefixEntry]:
        """Remove a prefix entry."""
        return self._entries.pop(prefix_hash, None)

    def entries(self) -> List[PrefixEntry]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class PrefixEntry:
    """An entry in the prefix trie tracking cached prefix KV blocks."""
    prefix_hash: str
    block_ids: List[int]
    cached_tokens: int
    last_access: float = 0.0
    access_count: int = 0


class KVCacheSimulator:
    """
    Simulates paged KV cache with prefix caching and eviction.

    The cache is divided into fixed-size blocks. Each block stores KV state
    for `block_size` tokens. Blocks are allocated on demand and freed on
    request completion.

    Reference: §3.4
    """

    def __init__(
        self,
        total_memory_bytes: int,
        block_size: int = 16,
        bytes_per_token: int = 256,  # 2 * n_layers * d_head * n_heads * dtype_bytes (simplified)
    ):
        self.block_size = block_size
        self.bytes_per_token = bytes_per_token
        self.bytes_per_block = block_size * bytes_per_token

        # Calculate total blocks
        self.total_blocks = max(1, total_memory_bytes // self.bytes_per_block)
        self.used_blocks = 0

        # Block tracking
        self._blocks: Dict[int, CacheBlock] = {}
        self._free_block_ids: List[int] = list(range(self.total_blocks))
        self._request_blocks: Dict[str, List[int]] = {}  # request_id -> block_ids

        # Prefix cache
        self.prefix_trie = PrefixTrie()

        # Eviction policy
        self.eviction_policy = EvictionPolicy.LRU

        # Metrics
        self._total_lookups = 0
        self._total_hits = 0

    @property
    def utilization(self) -> float:
        """Current cache utilization as a fraction [0, 1]."""
        return self.used_blocks / max(1, self.total_blocks)

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - self.used_blocks

    @property
    def hit_rate(self) -> float:
        """Overall prefix cache hit rate."""
        if self._total_lookups == 0:
            return 0.0
        return self._total_hits / self._total_lookups

    def allocate(
        self,
        request_id: str,
        num_tokens: int,
        prefix_hash: Optional[str],
        sim_time: float,
    ) -> Tuple[int, int]:
        """
        Allocate KV cache blocks for a request.

        Returns:
            (allocated_blocks, cached_tokens) — cached_tokens > 0 if prefix hit
        """
        import math
        needed_blocks = math.ceil(num_tokens / self.block_size)
        cached_tokens = 0

        # Check prefix cache
        if prefix_hash:
            self._total_lookups += 1
            entry = self.prefix_trie.lookup(prefix_hash, sim_time)
            if entry is not None:
                self._total_hits += 1
                cached_tokens = entry.cached_tokens
                # Reduce needed blocks by cached prefix blocks
                cached_blocks = min(len(entry.block_ids), needed_blocks)
                needed_blocks = max(0, needed_blocks - cached_blocks)

        # Allocate remaining blocks
        allocated = []
        while needed_blocks > 0 and self._free_block_ids:
            block_id = self._free_block_ids.pop()
            block = CacheBlock(
                block_id=block_id,
                request_id=request_id,
                prefix_hash=prefix_hash,
                last_access_time=sim_time,
                access_count=1,
            )
            self._blocks[block_id] = block
            allocated.append(block_id)
            self.used_blocks += 1
            needed_blocks -= 1

        self._request_blocks[request_id] = allocated

        # If we had a prefix hash, register the new allocation in the trie
        if prefix_hash and allocated:
            self.prefix_trie.insert(prefix_hash, allocated, sim_time)

        return len(allocated), cached_tokens

    def free(self, request_id: str) -> int:
        """Free all blocks allocated to a request. Returns number freed."""
        block_ids = self._request_blocks.pop(request_id, [])
        for bid in block_ids:
            block = self._blocks.pop(bid, None)
            if block is not None:
                self._free_block_ids.append(bid)
                self.used_blocks -= 1
        return len(block_ids)

    def can_allocate(self, num_tokens: int, prefix_hash: Optional[str] = None) -> bool:
        """Check if allocation is possible without eviction."""
        import math
        needed = math.ceil(num_tokens / self.block_size)

        if prefix_hash:
            entry = self.prefix_trie.lookup(prefix_hash, 0.0)
            if entry:
                needed = max(0, needed - len(entry.block_ids))

        return needed <= self.free_blocks

    def evict(self, num_blocks_needed: int, sim_time: float) -> int:
        """
        Evict blocks according to the current eviction policy.

        Returns number of blocks actually freed.
        """
        if self.eviction_policy == EvictionPolicy.LRU:
            return self._evict_lru(num_blocks_needed, sim_time)
        elif self.eviction_policy == EvictionPolicy.LFU:
            return self._evict_lfu(num_blocks_needed, sim_time)
        else:
            return self._evict_lru(num_blocks_needed, sim_time)

    def _evict_lru(self, num_needed: int, sim_time: float) -> int:
        """Evict least recently used blocks."""
        # Sort allocated blocks by last access time
        allocated_blocks = sorted(
            self._blocks.values(),
            key=lambda b: b.last_access_time,
        )

        freed = 0
        for block in allocated_blocks:
            if freed >= num_needed:
                break
            # Remove from request tracking
            req_id = block.request_id
            if req_id and req_id in self._request_blocks:
                if block.block_id in self._request_blocks[req_id]:
                    self._request_blocks[req_id].remove(block.block_id)

            # Free the block
            self._blocks.pop(block.block_id, None)
            self._free_block_ids.append(block.block_id)
            self.used_blocks -= 1
            freed += 1

        return freed

    def _evict_lfu(self, num_needed: int, sim_time: float) -> int:
        """Evict least frequently used blocks."""
        allocated_blocks = sorted(
            self._blocks.values(),
            key=lambda b: b.access_count,
        )

        freed = 0
        for block in allocated_blocks:
            if freed >= num_needed:
                break
            req_id = block.request_id
            if req_id and req_id in self._request_blocks:
                if block.block_id in self._request_blocks[req_id]:
                    self._request_blocks[req_id].remove(block.block_id)

            self._blocks.pop(block.block_id, None)
            self._free_block_ids.append(block.block_id)
            self.used_blocks -= 1
            freed += 1

        return freed

    def set_eviction_policy(self, policy: EvictionPolicy) -> None:
        """Change the eviction policy."""
        self.eviction_policy = policy

    def reset_metrics(self) -> None:
        """Reset hit/miss counters."""
        self._total_lookups = 0
        self._total_hits = 0
