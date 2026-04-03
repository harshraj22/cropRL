"""
CropRL Core Environment.

Implements the OpenEnv Environment interface for the farm management
simulation. Orchestrates the monthly step loop by delegating physics
to the dynamics engine.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

import numpy as np

from openenv.core.env_server.types import Observation, State
from openenv.core.env_server.interfaces import Environment

from cropRL.config import EnvConfig
from cropRL.dynamics import (
    apply_spoilage,
    calculate_expected_yield_potential,
    calculate_interest_rate,
    calculate_yield,
    format_text_observation,
    generate_market_prices,
    generate_rainfall,
)
from cropRL.models import CroprlAction, CroprlObservation, CroprlState


class CroprlEnvironment(Environment[CroprlAction, CroprlObservation, CroprlState]):
    """
    Farm management RL environment.

    The agent manages a small Indian farm over ``config.max_steps`` monthly
    steps.  Each step the agent picks one of 11 discrete actions (plant,
    irrigate, fertilize, harvest, sell, loan management, or wait).

    The environment tracks:
    - Crop lifecycle (planting → growth → harvest)
    - Soil nitrogen depletion / recovery
    - Stochastic weather and market prices
    - Financial state (cash, debt, interest)
    - Single-slot crop storage with spoilage
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        task_id: str = "default",
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.task_id = task_id

        # Will be initialised in reset()
        self._rng: Optional[np.random.Generator] = None
        self._internal: dict[str, Any] = {}
        self._state = CroprlState(task_id=task_id)

    # ──────────────────────────────────────────────────────────────
    # OpenEnv interface: reset
    # ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CroprlObservation:
        """
        Start a new episode.

        Initialises all farm state to starting conditions, generates
        initial weather and market prices, and returns the first
        observation.
        """
        self._rng = np.random.default_rng(seed)
        cfg = self.config

        # Calendar
        month = 1  # January
        step = 0

        # Generate stochastic values for month 1
        rainfall = generate_rainfall(month, cfg, self._rng)
        prices = generate_market_prices(month, cfg, self._rng)

        # Interest rate (use crop optimal 0.0 since no crop planted)
        interest_rate = calculate_interest_rate(
            cfg.base_interest_rate, month, rainfall, 0.0
        )

        # Internal farm state
        self._internal = {
            "month": month,
            "step": step,
            "rainfall": rainfall,
            "prices": prices,
            "interest_rate": interest_rate,
            # Crop
            "active_crop_type": 0,
            "crop_age_months": 0,
            # Soil
            "soil_nitrogen": cfg.initial_soil_nitrogen,
            # Finance
            "cash": cfg.initial_cash,
            "debt": 0.0,
            "has_active_loan": False,
            # Storage
            "stored_crop_type": 0,
            "stored_amount": 0.0,
            "stored_age_months": 0,
            # Per-step flags
            "irrigated": False,
            "fertilized": False,
            # Reward
            "previous_cash": cfg.initial_cash,
        }

        # Compute derived fields
        yield_potential = calculate_expected_yield_potential(
            crop_type=0,
            crop_age=0,
            soil_nitrogen=cfg.initial_soil_nitrogen,
            expected_rainfall=rainfall,
            config=cfg,
        )

        # Build state object
        self._state = CroprlState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            irrigated_this_month=False,
            fertilized_this_month=False,
            previous_cash=cfg.initial_cash,
            has_active_loan=False,
            task_id=self.task_id,
        )

        return self._build_observation(
            yield_potential=yield_potential,
            reward=0.0,
            done=False,
            message="New episode started. Your farm awaits!",
        )

    # ──────────────────────────────────────────────────────────────
    # OpenEnv interface: step
    # ──────────────────────────────────────────────────────────────

    def step(
        self,
        action: CroprlAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CroprlObservation:
        """
        Execute one monthly step.

        1. Validate & execute the chosen action
        2. Advance monthly dynamics (weather, prices, interest, ageing)
        3. Check termination (bankruptcy / horizon)
        4. Calculate reward
        5. Return observation
        """
        cfg = self.config
        s = self._internal
        action_id = action.action_id
        messages: list[str] = []
        penalty = 0.0

        # ── 1. Execute action ──────────────────────────────────────
        s["irrigated"] = False
        s["fertilized"] = False

        if action_id == 0:
            # Wait
            messages.append("You waited this month.")

        elif action_id in (1, 2, 3):
            # Plant crop
            crop_idx = action_id
            if s["active_crop_type"] != 0:
                penalty += cfg.invalid_action_penalty
                messages.append(
                    f"INVALID: Cannot plant — land already has "
                    f"{cfg.crop_names[s['active_crop_type']]} growing."
                )
            elif s["cash"] < cfg.seed_costs[crop_idx]:
                penalty += cfg.invalid_action_penalty
                messages.append(
                    f"INVALID: Not enough cash to plant "
                    f"{cfg.crop_names[crop_idx]} "
                    f"(need ₹{cfg.seed_costs[crop_idx]:,.0f}, "
                    f"have ₹{s['cash']:,.0f})."
                )
            else:
                s["cash"] -= cfg.seed_costs[crop_idx]
                s["active_crop_type"] = crop_idx
                s["crop_age_months"] = 0
                messages.append(
                    f"Planted {cfg.crop_names[crop_idx]}. "
                    f"Cost: ₹{cfg.seed_costs[crop_idx]:,.0f}."
                )

        elif action_id == 4:
            # Irrigate
            if s["active_crop_type"] == 0:
                penalty += cfg.invalid_action_penalty
                messages.append("INVALID: Nothing to irrigate — land is fallow.")
            elif s["cash"] < cfg.cost_irrigate:
                penalty += cfg.invalid_action_penalty
                messages.append(
                    f"INVALID: Not enough cash to irrigate "
                    f"(need ₹{cfg.cost_irrigate:,.0f})."
                )
            else:
                s["cash"] -= cfg.cost_irrigate
                s["irrigated"] = True
                messages.append(f"Irrigated. Cost: ₹{cfg.cost_irrigate:,.0f}.")

        elif action_id == 5:
            # Fertilize (always valid if cash available)
            if s["cash"] < cfg.cost_fertilize:
                penalty += cfg.invalid_action_penalty
                messages.append(
                    f"INVALID: Not enough cash to fertilize "
                    f"(need ₹{cfg.cost_fertilize:,.0f})."
                )
            else:
                s["cash"] -= cfg.cost_fertilize
                s["soil_nitrogen"] = min(
                    1.0, s["soil_nitrogen"] + cfg.fertilize_nitrogen_boost
                )
                s["fertilized"] = True
                messages.append(
                    f"Fertilized. Soil nitrogen boosted to "
                    f"{s['soil_nitrogen']:.2f}. "
                    f"Cost: ₹{cfg.cost_fertilize:,.0f}."
                )

        elif action_id == 6:
            # Harvest & Store
            penalty, msg = self._do_harvest_store(s, cfg)
            messages.append(msg)

        elif action_id == 7:
            # Harvest & Sell
            penalty, msg = self._do_harvest_sell(s, cfg)
            messages.append(msg)

        elif action_id == 8:
            # Sell Inventory
            if s["stored_amount"] <= 0:
                penalty += cfg.invalid_action_penalty
                messages.append("INVALID: Storage is empty — nothing to sell.")
            else:
                crop_t = s["stored_crop_type"]
                price = s["prices"][crop_t - 1]  # prices is 0-indexed for crops 1,2,3
                revenue = s["stored_amount"] * price
                s["cash"] += revenue
                messages.append(
                    f"Sold {s['stored_amount']:.1f} tons of "
                    f"{cfg.crop_names[crop_t]} at ₹{price:,.0f}/ton. "
                    f"Revenue: ₹{revenue:,.0f}."
                )
                s["stored_crop_type"] = 0
                s["stored_amount"] = 0.0
                s["stored_age_months"] = 0

        elif action_id == 9:
            # Take Loan
            if s["has_active_loan"]:
                penalty += cfg.invalid_action_penalty
                messages.append(
                    "INVALID: You already have an active loan. "
                    "Repay it first before taking another."
                )
            else:
                s["cash"] += cfg.loan_chunk
                s["debt"] += cfg.loan_chunk
                s["has_active_loan"] = True
                messages.append(
                    f"Took a loan of ₹{cfg.loan_chunk:,.0f}. "
                    f"Total debt: ₹{s['debt']:,.0f}."
                )

        elif action_id == 10:
            # Repay Loan
            if not s["has_active_loan"]:
                penalty += cfg.invalid_action_penalty
                messages.append("INVALID: No active loan to repay.")
            elif s["cash"] < s["debt"]:
                penalty += cfg.invalid_action_penalty
                messages.append(
                    f"INVALID: Not enough cash to repay full debt. "
                    f"Need ₹{s['debt']:,.0f}, have ₹{s['cash']:,.0f}."
                )
            else:
                repay_amount = s["debt"]
                s["cash"] -= repay_amount
                s["debt"] = 0.0
                s["has_active_loan"] = False
                messages.append(
                    f"Repaid full loan of ₹{repay_amount:,.0f}. "
                    f"You are now debt-free."
                )

        # ── 2. Advance monthly dynamics ────────────────────────────

        # Advance step/month
        s["step"] += 1
        s["month"] = (s["month"] % 12) + 1  # 1→2→…→12→1

        # Age crop
        if s["active_crop_type"] > 0:
            s["crop_age_months"] += 1

        # Age storage & check spoilage
        if s["stored_amount"] > 0:
            s["stored_age_months"] += 1
            remaining, spoiled = apply_spoilage(
                s["stored_age_months"], s["stored_amount"], cfg.max_storage_age
            )
            if spoiled:
                messages.append(
                    f"SPOILAGE: Your stored {cfg.crop_names[s['stored_crop_type']]} "
                    f"({s['stored_amount']:.1f} tons) has rotted!"
                )
                s["stored_amount"] = 0.0
                s["stored_crop_type"] = 0
                s["stored_age_months"] = 0
            else:
                s["stored_amount"] = remaining

        # Natural soil recovery
        s["soil_nitrogen"] = min(
            1.0, s["soil_nitrogen"] + cfg.natural_nitrogen_recovery
        )

        # Apply interest on debt
        if s["has_active_loan"] and s["debt"] > 0:
            monthly_rate = s["interest_rate"] / 12.0
            s["debt"] *= 1.0 + monthly_rate

        # [Future] Storage cost
        if cfg.enable_storage_cost and s["stored_amount"] > 0:
            storage_cost = cfg.cost_storage_monthly * s["stored_amount"]
            s["cash"] -= storage_cost

        # Generate new weather & prices for the new month
        s["rainfall"] = generate_rainfall(s["month"], cfg, self._rng)
        prev_prices = s["prices"]  # save for autocorrelation
        s["prices"] = generate_market_prices(
            s["month"], cfg, self._rng, prev_prices=prev_prices,
        )

        # Update interest rate
        optimal = (
            cfg.optimal_rainfall[s["active_crop_type"]]
            if s["active_crop_type"] > 0
            else 0.0
        )
        s["interest_rate"] = calculate_interest_rate(
            cfg.base_interest_rate, s["month"], s["rainfall"], optimal
        )

        # Apply nitrogen impact from crop growth (passive depletion while growing)
        # Note: main nitrogen_impact is applied at harvest time

        # ── 3. Check termination ───────────────────────────────────
        done = False
        terminal_bonus = 0.0

        if s["step"] >= cfg.max_steps:
            # Terminal liquidation
            done = True
            soil_bonus = s["soil_nitrogen"] * cfg.terminal_soil_bonus_factor

            # Value of active crop (estimated)
            active_value = 0.0
            if s["active_crop_type"] > 0:
                est_yield = calculate_yield(
                    s["active_crop_type"],
                    s["crop_age_months"],
                    s["soil_nitrogen"],
                    s["rainfall"],
                    s["irrigated"],
                    cfg,
                )
                price_idx = s["active_crop_type"] - 1
                active_value = est_yield * s["prices"][price_idx]

            # Value of stored crop
            stored_value = 0.0
            if s["stored_amount"] > 0:
                stored_price_idx = s["stored_crop_type"] - 1
                stored_value = s["stored_amount"] * s["prices"][stored_price_idx]

            terminal_bonus = soil_bonus + active_value + stored_value
            messages.append(
                f"EPISODE COMPLETE! Terminal bonus: ₹{terminal_bonus:,.0f} "
                f"(soil: ₹{soil_bonus:,.0f}, "
                f"crop: ₹{active_value:,.0f}, "
                f"storage: ₹{stored_value:,.0f})."
            )

        elif s["cash"] < 0 and s["has_active_loan"]:
            # Bankruptcy
            done = True
            penalty += cfg.bankruptcy_penalty
            messages.append(
                "BANKRUPTCY! Cash is negative and you have outstanding debt."
            )

        # ── 4. Calculate reward ────────────────────────────────────
        cash_delta = s["cash"] - s["previous_cash"]
        reward = cash_delta + penalty + terminal_bonus
        s["previous_cash"] = s["cash"]

        # ── 5. Compute derived observation fields ──────────────────
        yield_potential = calculate_expected_yield_potential(
            s["active_crop_type"],
            s["crop_age_months"],
            s["soil_nitrogen"],
            s["rainfall"],
            cfg,
        )

        # Update state
        self._state.step_count = s["step"]
        self._state.irrigated_this_month = s["irrigated"]
        self._state.fertilized_this_month = s["fertilized"]
        self._state.previous_cash = s["previous_cash"]
        self._state.has_active_loan = s["has_active_loan"]

        return self._build_observation(
            yield_potential=yield_potential,
            reward=reward,
            done=done,
            message=" | ".join(messages),
        )

    # ──────────────────────────────────────────────────────────────
    # OpenEnv interface: state
    # ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> CroprlState:
        """Return the current internal state."""
        return self._state

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        yield_potential: float,
        reward: float,
        done: bool,
        message: str,
    ) -> CroprlObservation:
        """Construct a CroprlObservation from current internal state."""
        s = self._internal
        cfg = self.config

        obs_dict = {
            "current_month": s["month"],
            "current_step": s["step"],
            "expected_rainfall": s["rainfall"],
            "active_crop_type": s["active_crop_type"],
            "crop_age_months": s["crop_age_months"],
            "expected_yield_potential": yield_potential,
            "soil_nitrogen": s["soil_nitrogen"],
            "cash_balance": s["cash"],
            "current_debt": s["debt"],
            "current_interest_rate": s["interest_rate"],
            "market_price_crop_1": s["prices"][0],
            "market_price_crop_2": s["prices"][1],
            "market_price_crop_3": s["prices"][2],
            "cost_seed_1": cfg.seed_costs[1],
            "cost_seed_2": cfg.seed_costs[2],
            "cost_seed_3": cfg.seed_costs[3],
            "cost_irrigate": cfg.cost_irrigate,
            "cost_fertilize": cfg.cost_fertilize,
            "stored_crop_type": s["stored_crop_type"],
            "stored_amount": s["stored_amount"],
            "stored_age_months": s["stored_age_months"],
            "message": message,
        }

        # Text mode
        text_summary = ""
        if cfg.text_mode:
            valid_actions = self._get_valid_actions()
            text_summary = format_text_observation(
                obs_dict, cfg, s["has_active_loan"], valid_actions
            )

        return CroprlObservation(
            **obs_dict,
            text_summary=text_summary,
            done=done,
            reward=reward,
        )

    def _get_valid_actions(self) -> list[int]:
        """Return the list of currently valid action IDs."""
        s = self._internal
        cfg = self.config
        valid = [0]  # Wait is always valid

        # Plant actions (1, 2, 3)
        if s["active_crop_type"] == 0:
            for crop_idx in (1, 2, 3):
                if s["cash"] >= cfg.seed_costs[crop_idx]:
                    valid.append(crop_idx)

        # Irrigate (4)
        if s["active_crop_type"] > 0 and s["cash"] >= cfg.cost_irrigate:
            valid.append(4)

        # Fertilize (5)
        if s["cash"] >= cfg.cost_fertilize:
            valid.append(5)

        # Harvest & Store (6), Harvest & Sell (7)
        if s["active_crop_type"] > 0 and s["crop_age_months"] >= 1:
            valid.append(6)
            valid.append(7)

        # Sell Inventory (8)
        if s["stored_amount"] > 0:
            valid.append(8)

        # Take Loan (9)
        if not s["has_active_loan"]:
            valid.append(9)

        # Repay Loan (10)
        if s["has_active_loan"] and s["cash"] >= s["debt"]:
            valid.append(10)

        return sorted(valid)

    def _do_harvest_store(
        self, s: dict, cfg: EnvConfig
    ) -> tuple[float, str]:
        """Execute Harvest & Store action. Returns (penalty, message)."""
        if s["active_crop_type"] == 0 or s["crop_age_months"] < 1:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to harvest — "
                "no crop planted or crop too young."
            )

        # Calculate yield
        crop_type = s["active_crop_type"]
        harvested = calculate_yield(
            crop_type,
            s["crop_age_months"],
            s["soil_nitrogen"],
            s["rainfall"],
            s["irrigated"],
            cfg,
            rng=self._rng,  # stochastic harvest
        )

        parts: list[str] = []

        # Auto-sell existing storage if occupied
        if s["stored_amount"] > 0:
            old_type = s["stored_crop_type"]
            old_price = s["prices"][old_type - 1]
            old_revenue = s["stored_amount"] * old_price
            s["cash"] += old_revenue
            parts.append(
                f"Auto-sold {s['stored_amount']:.1f} tons of "
                f"{cfg.crop_names[old_type]} for ₹{old_revenue:,.0f}."
            )

        # Apply nitrogen impact from harvesting this crop
        s["soil_nitrogen"] = max(
            0.0, min(1.0, s["soil_nitrogen"] + cfg.nitrogen_impact[crop_type])
        )

        # Store new harvest
        s["stored_crop_type"] = crop_type
        s["stored_amount"] = harvested
        s["stored_age_months"] = 0

        # Reset land
        s["active_crop_type"] = 0
        s["crop_age_months"] = 0

        parts.append(
            f"Harvested {harvested:.1f} tons of {cfg.crop_names[crop_type]} "
            f"and stored it."
        )
        return 0.0, " ".join(parts)

    def _do_harvest_sell(
        self, s: dict, cfg: EnvConfig
    ) -> tuple[float, str]:
        """Execute Harvest & Sell action. Returns (penalty, message)."""
        if s["active_crop_type"] == 0 or s["crop_age_months"] < 1:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to harvest — "
                "no crop planted or crop too young."
            )

        crop_type = s["active_crop_type"]
        harvested = calculate_yield(
            crop_type,
            s["crop_age_months"],
            s["soil_nitrogen"],
            s["rainfall"],
            s["irrigated"],
            cfg,
            rng=self._rng,  # stochastic harvest
        )

        price = s["prices"][crop_type - 1]
        revenue = harvested * price

        # Apply nitrogen impact
        s["soil_nitrogen"] = max(
            0.0, min(1.0, s["soil_nitrogen"] + cfg.nitrogen_impact[crop_type])
        )

        # Cash in
        s["cash"] += revenue

        # Reset land
        s["active_crop_type"] = 0
        s["crop_age_months"] = 0

        return 0.0, (
            f"Harvested {harvested:.1f} tons of {cfg.crop_names[crop_type]} "
            f"and sold at ₹{price:,.0f}/ton. Revenue: ₹{revenue:,.0f}."
        )
