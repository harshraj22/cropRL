"""
MarketEngine — Supply-demand price clearing and hype crop cycle for multi-agent CropRL.

Two sub-systems:
1. MarketEngine: queues sell orders within a month, then resolves them all at
   month-end using a demand-response model so that collective sell volume
   drives down the clearing price.

2. HypeEngine: manages the four-phase (DORMANT → BUILDING → PEAK → COLLAPSING)
   hype cycle for specialty crops (Matcha, Quinoa, Turmeric).  Integrates with
   MarketEngine.resolve_month() to trigger early collapse on over-supply.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cropRL.config import EnvConfig, MultiAgentConfig
from cropRL.enums import CropType, HypePhase
from cropRL.models import HypeCropStatus

# Hype crop indices
HYPE_CROP_TYPES = (
    CropType.MATCHA,
    CropType.QUINOA,
    CropType.TURMERIC,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hype Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _HypeCrop:
    """Internal state for a single hype crop's demand cycle."""

    # Phase transition constants (from the plan)
    HYPE_BUILD_RATE = 0.08        # hype_level increase per month in BUILDING
    HYPE_COLLAPSE_RATE = 0.20     # hype_level decrease per month in COLLAPSING
    HYPE_BUILD_THRESHOLD = 0.9    # hype_level at which BUILDING → PEAK
    PEAK_MIN_MONTHS = 2
    PEAK_MAX_MONTHS = 4

    def __init__(
        self,
        crop_type: int,
        crop_name: str,
        max_hype_premium: float,
        rng: np.random.Generator,
    ) -> None:
        self.crop_type = crop_type
        self.crop_name = crop_name
        self.max_hype_premium = max_hype_premium
        self._rng = rng

        self.hype_level: float = 0.0
        self.phase: HypePhase = HypePhase.DORMANT
        self.months_in_phase: int = 0
        self._peak_duration: int = self.PEAK_MIN_MONTHS  # set on PEAK entry

    def tick(self, trigger_prob: float, collapse_threshold: float, sold_volume: float, market_capacity: float) -> None:
        """Advance the hype cycle by one month."""
        self.months_in_phase += 1

        if self.phase == HypePhase.DORMANT:
            # Random trigger to BUILDING phase
            if self._rng.random() < trigger_prob:
                self._enter_building()

        elif self.phase == HypePhase.BUILDING:
            self.hype_level = min(1.0, self.hype_level + self.HYPE_BUILD_RATE)
            if self.hype_level >= self.HYPE_BUILD_THRESHOLD:
                self._enter_peak()

        elif self.phase == HypePhase.PEAK:
            self.hype_level = 1.0
            # Early collapse trigger: over-supply
            over_supplied = (
                market_capacity > 0
                and sold_volume > market_capacity * collapse_threshold
            )
            # Timeout trigger: peak duration exceeded
            timed_out = self.months_in_phase >= self._peak_duration
            if over_supplied or timed_out:
                self._enter_collapsing()

        elif self.phase == HypePhase.COLLAPSING:
            self.hype_level = max(0.0, self.hype_level - self.HYPE_COLLAPSE_RATE)
            if self.hype_level <= 0.0:
                self._enter_dormant()

    def get_hype_mult(self) -> float:
        """
        Return the price multiplier for this crop given current hype_level.

        hype_mult = 1.0 + hype_level × (max_hype_premium − 1.0)
        """
        return 1.0 + self.hype_level * (self.max_hype_premium - 1.0)

    def status(self) -> HypeCropStatus:
        return HypeCropStatus(
            crop_type=self.crop_type,
            crop_name=self.crop_name,
            hype_level=round(self.hype_level, 4),
            phase=self.phase,
            months_in_phase=self.months_in_phase,
        )

    # ── Phase transitions ──────────────────────────────────────

    def _enter_building(self) -> None:
        self.phase = HypePhase.BUILDING
        self.hype_level = 0.05       # small nudge to signal start
        self.months_in_phase = 0

    def _enter_peak(self) -> None:
        self.phase = HypePhase.PEAK
        self.hype_level = 1.0
        self.months_in_phase = 0
        self._peak_duration = int(self._rng.integers(
            self.PEAK_MIN_MONTHS, self.PEAK_MAX_MONTHS + 1
        ))

    def _enter_collapsing(self) -> None:
        self.phase = HypePhase.COLLAPSING
        self.months_in_phase = 0

    def _enter_dormant(self) -> None:
        self.phase = HypePhase.DORMANT
        self.hype_level = 0.0
        self.months_in_phase = 0


class HypeEngine:
    """
    Manages hype cycles for all hype crops.
    """

    def __init__(
        self,
        ma_config: MultiAgentConfig,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> None:
        self._cfg = ma_config
        self._env_cfg = env_config
        self._crops: Dict[int, _HypeCrop] = {}

        for ct in HYPE_CROP_TYPES:
            premium = ma_config.max_hype_premium.get(int(ct), 2.0)
            self._crops[int(ct)] = _HypeCrop(
                crop_type=int(ct),
                crop_name=env_config.crop_names[int(ct)],
                max_hype_premium=premium,
                rng=rng,
            )

    def tick_month(
        self, sold_volumes: Dict[int, float]
    ) -> List[HypeCropStatus]:
        """
        Advance all hype cycles by one month.

        Parameters
        ----------
        sold_volumes : dict
            Total volume sold per crop this month (crop_type_int → tons).

        Returns
        -------
        list of HypeCropStatus
        """
        statuses = []
        for ct_int, crop in self._crops.items():
            vol = sold_volumes.get(ct_int, 0.0)
            cap = self._cfg.market_capacity.get(ct_int, 1.0)
            crop.tick(
                trigger_prob=self._cfg.hype_trigger_prob,
                collapse_threshold=self._cfg.hype_collapse_supply_threshold,
                sold_volume=vol,
                market_capacity=cap,
            )
            statuses.append(crop.status())
        return statuses

    def get_hype_mult(self, crop_type: int) -> float:
        """Return the current hype price multiplier for a crop (1.0 if non-hype)."""
        if crop_type not in self._crops:
            return 1.0
        return self._crops[crop_type].get_hype_mult()

    def statuses(self) -> List[HypeCropStatus]:
        return [c.status() for c in self._crops.values()]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Market Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _SellOrder:
    """A pending sell order queued during the month."""
    __slots__ = ("agent_id", "crop_type", "volume", "is_inventory")

    def __init__(
        self,
        agent_id: int,
        crop_type: int,
        volume: float,
        is_inventory: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.crop_type = crop_type
        self.volume = volume
        self.is_inventory = is_inventory


class MarketEngine:
    """
    Shared market clearing engine for the multi-agent environment.

    Responsibilities:
    - Holds the authoritative ``realised_prices`` dict (post-supply-shock).
    - Queues sell orders during the month via ``queue_sell()``.
    - Resolves all queued orders at month-end via ``resolve_month()``,
      applying demand-response to produce clearing prices.
    - Manages base price generation (random walk from existing dynamics).
    """

    def __init__(
        self,
        ma_config: MultiAgentConfig,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> None:
        self._ma_cfg = ma_config
        self._env_cfg = env_config
        self._rng = rng

        # Base prices (the mean-reverting random walk target)
        # Initialised to config base prices; updated each month
        self._base_prices: List[float] = list(env_config.base_market_prices)

        # Realised prices after supply-demand adjustment (updated at resolve_month)
        # Indexed 0..num_crop_types-1  (0 = Fallow, always 0)
        self.realised_prices: List[float] = list(env_config.base_market_prices)

        # Last month's realised prices (exposed in observation)
        self.last_month_realised_prices: Tuple[float, ...] = tuple(
            env_config.base_market_prices[1:4]  # Corn, Wheat, Chickpea
        )

        # Pending sell orders this month
        self._pending_orders: List[_SellOrder] = []

        # Hype engine
        self._hype_engine: Optional[HypeEngine] = None
        if ma_config.enable_hype_crops:
            self._hype_engine = HypeEngine(ma_config, env_config, rng)

        # Previous tick's prices for autocorrelation random walk
        self._prev_prices: Optional[Tuple[float, ...]] = None

    # ──────────────────────────────────────────────────────────────
    # Sell Queue
    # ──────────────────────────────────────────────────────────────

    def queue_sell(
        self,
        agent_id: int,
        crop_type: int,
        volume: float,
        is_inventory: bool = False,
    ) -> None:
        """
        Queue a sell order for the current month.
        All queued orders are cleared at ``resolve_month()``.
        """
        if volume > 0:
            self._pending_orders.append(
                _SellOrder(agent_id, crop_type, volume, is_inventory)
            )

    # ──────────────────────────────────────────────────────────────
    # Month Resolution
    # ──────────────────────────────────────────────────────────────

    def resolve_month(self, month: int) -> Dict[int, float]:
        """
        Resolve all pending sell orders for this month.

        Steps:
        1. Aggregate total sold volume per crop.
        2. Compute demand_response(V, crop) for each crop.
        3. Compute realised_price = base × seasonal × hype × demand_response.
        4. Credit each sell order with realised_price × volume.
        5. Tick hype cycles with actual sold volumes.

        Returns
        -------
        dict
            agent_id → total revenue earned this month from sells.
        """
        # Aggregate volumes
        total_volume: Dict[int, float] = {}
        for order in self._pending_orders:
            total_volume[order.crop_type] = (
                total_volume.get(order.crop_type, 0.0) + order.volume
            )

        # Compute demand response & realised prices per crop
        clearing_prices: Dict[int, float] = {}
        for crop_idx in range(1, self._env_cfg.num_crop_types):
            base = self._base_prices[crop_idx]
            vol = total_volume.get(crop_idx, 0.0)
            demand_resp = self._demand_response(vol, crop_idx)
            hype_mult = (
                self._hype_engine.get_hype_mult(crop_idx)
                if self._hype_engine
                else 1.0
            )
            clearing_prices[crop_idx] = base * demand_resp * hype_mult
            self.realised_prices[crop_idx] = clearing_prices[crop_idx]

        # Credit revenues per agent
        revenues: Dict[int, float] = {}
        for order in self._pending_orders:
            price = clearing_prices.get(order.crop_type, 0.0)
            rev = order.volume * price
            revenues[order.agent_id] = revenues.get(order.agent_id, 0.0) + rev

        # Snapshot last_month_realised_prices (All 6 crops)
        self.last_month_realised_prices = tuple(
            self.realised_prices[i] for i in range(1, self._env_cfg.num_crop_types)
        )

        # Tick hype engine (uses actual sold volumes)
        if self._hype_engine:
            self._hype_engine.tick_month(total_volume)

        # Clear pending orders for next month
        self._pending_orders = []

        return revenues

    def generate_base_prices(
        self,
        month: int,
        inflated_base_prices: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Generate new base prices using the existing mean-reverting random walk.
        This replaces the per-agent call to dynamics.generate_market_prices()
        for the *shared* commodity baseline.

        Returns the new base prices list (indexed 0..num_crops-1).
        """
        from cropRL.dynamics import generate_market_prices

        base = inflated_base_prices or self._base_prices

        # The existing dynamics function generates 3 prices (crops 1-3).
        p1, p2, p3 = generate_market_prices(
            month=month,
            config=self._env_cfg,
            rng=self._rng,
            prev_prices=self._prev_prices,
            effective_base_prices=tuple(base),
        )
        # Extend for hype crops (they use their own base price + hype_mult)
        new_prices = list(base)  # copy
        new_prices[1] = p1
        new_prices[2] = p2
        new_prices[3] = p3
        # Hype crops 4-6 get small random walk independently
        for hc in range(4, self._env_cfg.num_crop_types):
            hc_base = base[hc]
            noise = float(self._rng.normal(0.0, self._env_cfg.market_price_sigma))
            noise = float(np.clip(noise, -3 * self._env_cfg.market_price_sigma,
                                  3 * self._env_cfg.market_price_sigma))
            floor = hc_base * self._env_cfg.price_min_multiplier
            ceiling = hc_base * self._env_cfg.price_max_multiplier
            if self._prev_prices is not None and len(self._prev_prices) > hc - 1:
                prev = self._prev_prices[hc - 1]
                target = hc_base
                drift = self._env_cfg.price_reversion_speed * (target - prev) / max(target, 1.0)
                new_prices[hc] = float(np.clip(prev * (1.0 + drift + noise), floor, ceiling))
            else:
                new_prices[hc] = float(np.clip(hc_base * (1.0 + noise), floor, ceiling))

        self._base_prices = new_prices
        self._prev_prices = tuple(new_prices[1:])  # for next month's autocorrelation

        return new_prices

    def hype_statuses(self) -> List[HypeCropStatus]:
        """Return current hype status for all hype crops."""
        if self._hype_engine:
            return self._hype_engine.statuses()
        return []

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _demand_response(self, volume: float, crop_type: int) -> float:
        """
        Compute the demand-response multiplier for a given sold volume.

        demand_response(V) = max(price_floor_mult,
                                 1 - price_impact_coeff × (V / market_capacity))
        """
        capacity = self._ma_cfg.market_capacity.get(crop_type, 50.0)
        if capacity <= 0:
            return 1.0
        raw = 1.0 - self._ma_cfg.price_impact_coeff * (volume / capacity)
        return max(self._ma_cfg.price_floor_mult, raw)
