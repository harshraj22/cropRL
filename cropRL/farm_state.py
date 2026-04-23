"""
FarmState — one farmer's private state and action logic.

No OpenEnv dependency. This is a plain class used by MultiAgentCroprlEnvironment
as the per-farm state container. All action methods are public.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from cropRL.config import EnvConfig
from cropRL.dynamics import (
    apply_spoilage,
    calculate_expected_yield_potential,
    calculate_interest_rate,
    calculate_yield,
    generate_market_prices,
    generate_rainfall,
    realise_rainfall,
)
from cropRL.enums import ActionType, CropType


class FarmState:
    """
    Holds the mutable state for a single farm and exposes action helpers.

    All ``do_*`` methods return ``(penalty, message)`` — exactly like the
    old ``CroprlEnvironment._do_*`` private methods, but now public.
    """

    def __init__(self, config: EnvConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._rng = rng
        self._s: dict = {}
        self._init_state()

    # ──────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────

    def _init_state(self) -> None:
        cfg = self.config
        month = 1
        rainfall = generate_rainfall(month, cfg, self._rng)
        prices = generate_market_prices(month, cfg, self._rng)
        interest_rate = calculate_interest_rate(
            cfg.base_interest_rate, month, rainfall, 0.0
        )
        self._s = {
            "month": month,
            "step": 0,
            "month_count": 0,
            "year": 1,
            "expected_rainfall": rainfall,
            "prices": prices,
            "interest_rate": interest_rate,
            # Crop
            "active_crop_type": CropType.FALLOW,
            "crop_age_months": 0,
            "planting_month": 0,
            # Soil
            "soil_nitrogen": cfg.initial_soil_nitrogen,
            # Water
            "water_level": 0.0,
            # Finance
            "cash": cfg.initial_cash,
            "debt": 0.0,
            "has_active_loan": False,
            "loan_interest_rate": 0.0,
            # Storage
            "stored_crop_type": CropType.FALLOW,
            "stored_amount": 0.0,
            "stored_age_months": 0,
            # Per-step flags
            "irrigated": False,
            "fertilized": False,
            # Inflated values (mutated by inflation each year)
            "inflated_seed_costs": list(cfg.seed_costs),
            "inflated_cost_irrigate": cfg.cost_irrigate,
            "inflated_cost_fertilize": cfg.cost_fertilize,
            "inflated_loan_chunk": cfg.loan_chunk,
            "inflated_base_land_price": cfg.base_land_price,
            "inflated_monthly_fixed_cost": cfg.monthly_fixed_cost,
            "inflated_base_market_prices": list(cfg.base_market_prices),
        }
        self._s["prev_net_worth"] = self.compute_net_worth()

    def reset(self) -> None:
        """Reset farm state to initial conditions (re-uses existing rng)."""
        self._init_state()

    # ──────────────────────────────────────────────────────────────
    # State access helpers
    # ──────────────────────────────────────────────────────────────

    @property
    def s(self) -> dict:
        """Direct access to the state dict."""
        return self._s

    @property
    def month(self) -> int:
        return self._s["month"]

    @property
    def month_count(self) -> int:
        return self._s["month_count"]

    @property
    def cash(self) -> float:
        return self._s["cash"]

    @property
    def has_active_loan(self) -> bool:
        return self._s["has_active_loan"]

    # ──────────────────────────────────────────────────────────────
    # Action handlers — all return (penalty, message)
    # ──────────────────────────────────────────────────────────────

    def do_plant(self, action_id: int) -> Tuple[float, str]:
        """Plant a standard crop (action_id 1/2/3)."""
        s = self._s
        cfg = self.config
        crop_idx = action_id
        seed_cost = s["inflated_seed_costs"][crop_idx]

        if s["active_crop_type"] != CropType.FALLOW:
            return cfg.invalid_action_penalty, (
                f"INVALID: Cannot plant — land already has "
                f"{cfg.crop_names[s['active_crop_type']]} growing."
            )
        if s["cash"] < seed_cost:
            return cfg.invalid_action_penalty, (
                f"INVALID: Not enough cash to plant "
                f"{cfg.crop_names[crop_idx]} "
                f"(need ₹{seed_cost:,.0f}, have ₹{s['cash']:,.0f})."
            )

        s["cash"] -= seed_cost
        s["active_crop_type"] = crop_idx
        s["crop_age_months"] = 0
        s["planting_month"] = s["month"]
        return 0.0, (
            f"Planted {cfg.crop_names[crop_idx]}. Cost: ₹{seed_cost:,.0f}."
        )

    def do_plant_hype(self, crop_type: int) -> Tuple[float, str]:
        """Plant a hype crop (CropType 4/5/6)."""
        s = self._s
        cfg = self.config
        crop_idx = int(crop_type)
        seed_cost = s["inflated_seed_costs"][crop_idx]

        if s["active_crop_type"] != CropType.FALLOW:
            return cfg.invalid_action_penalty, (
                f"INVALID: Cannot plant — land already has "
                f"{cfg.crop_names[s['active_crop_type']]} growing."
            )
        if s["cash"] < seed_cost:
            return cfg.invalid_action_penalty, (
                f"INVALID: Not enough cash to plant "
                f"{cfg.crop_names[crop_idx]} "
                f"(need ₹{seed_cost:,.0f}, have ₹{s['cash']:,.0f})."
            )

        s["cash"] -= seed_cost
        s["active_crop_type"] = crop_idx
        s["crop_age_months"] = 0
        s["planting_month"] = s["month"]
        return 0.0, (
            f"Planted {cfg.crop_names[crop_idx]} (hype crop). "
            f"Cost: ₹{seed_cost:,.0f}."
        )

    def do_irrigate(self) -> Tuple[float, str]:
        s = self._s
        cfg = self.config
        cost = s["inflated_cost_irrigate"]

        if s["active_crop_type"] == CropType.FALLOW:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to irrigate — land is fallow."
            )
        if s["cash"] < cost:
            return cfg.invalid_action_penalty, (
                f"INVALID: Not enough cash to irrigate (need ₹{cost:,.0f})."
            )

        s["cash"] -= cost
        s["irrigated"] = True
        crop_t = s["active_crop_type"]
        s["water_level"] += cfg.irrigate_amount[crop_t]
        s["water_level"] = min(s["water_level"], cfg.optimal_water_level[crop_t])
        return 0.0, (
            f"Irrigated. Water level now {s['water_level']:.2f}. "
            f"Cost: ₹{cost:,.0f}."
        )

    def do_fertilize(self) -> Tuple[float, str]:
        s = self._s
        cfg = self.config
        cost = s["inflated_cost_fertilize"]

        if s["cash"] < cost:
            return cfg.invalid_action_penalty, (
                f"INVALID: Not enough cash to fertilize (need ₹{cost:,.0f})."
            )

        s["cash"] -= cost
        s["soil_nitrogen"] = min(1.0, s["soil_nitrogen"] + cfg.fertilize_nitrogen_boost)
        s["fertilized"] = True
        return 0.0, (
            f"Fertilized. Soil nitrogen boosted to "
            f"{s['soil_nitrogen']:.2f}. Cost: ₹{cost:,.0f}."
        )

    def do_harvest_store(self) -> Tuple[float, str]:
        s = self._s
        cfg = self.config
        if s["active_crop_type"] == CropType.FALLOW or s["crop_age_months"] < 1:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to harvest — no crop planted or crop too young."
            )

        crop_type = s["active_crop_type"]
        harvested = calculate_yield(
            crop_type, s["crop_age_months"], s["soil_nitrogen"],
            s["water_level"], s["planting_month"], cfg, rng=self._rng,
        )

        parts: list[str] = []
        if s["stored_amount"] > 0:
            old_type = s["stored_crop_type"]
            old_revenue = s["stored_amount"] * s["prices"][old_type - 1]
            s["cash"] += old_revenue
            parts.append(
                f"Auto-sold {s['stored_amount']:.1f} tons of "
                f"{cfg.crop_names[old_type]} for ₹{old_revenue:,.0f}."
            )

        s["stored_crop_type"] = crop_type
        s["stored_amount"] = harvested
        s["stored_age_months"] = 0
        s["active_crop_type"] = CropType.FALLOW
        s["crop_age_months"] = 0
        s["planting_month"] = 0
        parts.append(
            f"Harvested {harvested:.1f} tons of {cfg.crop_names[crop_type]} "
            f"and stored it."
        )
        return 0.0, " ".join(parts)

    def do_harvest_sell_queued(self) -> Tuple[float, str, int, float]:
        """
        Harvest and prepare for queued sale. Does NOT credit cash.
        Returns (penalty, message, crop_type, harvested_amount).
        The caller (MultiAgentCroprlEnvironment) queues the sale in the MarketEngine.
        """
        s = self._s
        cfg = self.config
        if s["active_crop_type"] == CropType.FALLOW or s["crop_age_months"] < 1:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to harvest — no crop planted or crop too young."
            ), 0, 0.0

        crop_type = s["active_crop_type"]
        harvested = calculate_yield(
            crop_type, s["crop_age_months"], s["soil_nitrogen"],
            s["water_level"], s["planting_month"], cfg, rng=self._rng,
        )

        s["active_crop_type"] = CropType.FALLOW
        s["crop_age_months"] = 0
        s["planting_month"] = 0

        return 0.0, (
            f"Harvested {harvested:.1f} tons of {cfg.crop_names[crop_type]}. "
            f"Sale queued for month-end market clearing."
        ), crop_type, harvested

    def do_sell_inventory_queued(self) -> Tuple[float, str, int, float]:
        """
        Queue inventory for sale. Does NOT credit cash.
        Returns (penalty, message, crop_type, volume).
        """
        s = self._s
        cfg = self.config
        if s["stored_amount"] <= 0:
            return cfg.invalid_action_penalty, (
                "INVALID: Storage is empty — nothing to sell."
            ), 0, 0.0

        crop_t = s["stored_crop_type"]
        volume = s["stored_amount"]
        s["stored_crop_type"] = CropType.FALLOW
        s["stored_amount"] = 0.0
        s["stored_age_months"] = 0

        return 0.0, (
            f"Queued {volume:.1f} tons of {cfg.crop_names[crop_t]} "
            f"for month-end market clearing."
        ), crop_t, volume

    def do_take_loan(self) -> Tuple[float, str]:
        s = self._s
        cfg = self.config
        if s["has_active_loan"]:
            return cfg.invalid_action_penalty, (
                "INVALID: You already have an active loan. "
                "Repay it first before taking another."
            )

        amount = s["inflated_loan_chunk"]
        s["cash"] += amount
        s["debt"] += amount
        s["has_active_loan"] = True
        s["loan_interest_rate"] = s["interest_rate"]
        return 0.0, (
            f"Took a loan of ₹{amount:,.0f} at "
            f"{s['loan_interest_rate'] * 100:.1f}% annual. "
            f"Total debt: ₹{s['debt']:,.0f}."
        )

    def do_repay_loan(self) -> Tuple[float, str]:
        s = self._s
        cfg = self.config
        if not s["has_active_loan"]:
            return cfg.invalid_action_penalty, "INVALID: No active loan to repay."
        if s["cash"] < s["debt"]:
            return cfg.invalid_action_penalty, (
                f"INVALID: Not enough cash to repay full debt. "
                f"Need ₹{s['debt']:,.0f}, have ₹{s['cash']:,.0f}."
            )

        repay = s["debt"]
        s["cash"] -= repay
        s["debt"] = 0.0
        s["has_active_loan"] = False
        s["loan_interest_rate"] = 0.0
        return 0.0, f"Repaid full loan of ₹{repay:,.0f}. You are now debt-free."

    # ──────────────────────────────────────────────────────────────
    # Monthly dynamics
    # ──────────────────────────────────────────────────────────────

    def advance_month(self) -> list[str]:
        """Advance all monthly dynamics. Returns list of event messages."""
        s = self._s
        cfg = self.config
        messages: list[str] = []

        old_month = s["month"]
        s["month"] = (s["month"] % 12) + 1
        s["month_count"] += 1

        # Inflation (year boundary)
        if s["month"] == 1 and old_month == 12:
            self.apply_inflation()
            s["year"] += 1
            messages.append(
                f"Year {s['year']} begins. "
                f"Inflation applied ({cfg.inflation_rate * 100:.0f}%)."
            )

        # Realise rainfall
        realised = realise_rainfall(
            s["expected_rainfall"], cfg.weather_sigma_realisation, self._rng,
        )

        # Water level
        crop_t = s["active_crop_type"]
        s["water_level"] += realised
        if crop_t != CropType.FALLOW:
            s["water_level"] -= cfg.water_utilised_monthly[crop_t]
        s["water_level"] = max(0.0, s["water_level"])
        s["water_level"] = min(s["water_level"], cfg.optimal_water_level[crop_t])

        # Age crop & nitrogen
        if crop_t != CropType.FALLOW:
            s["crop_age_months"] += 1
            s["soil_nitrogen"] += cfg.monthly_nitrogen_impact[crop_t]
            s["soil_nitrogen"] = max(0.0, min(1.0, s["soil_nitrogen"]))

        # Natural nitrogen recovery
        s["soil_nitrogen"] = min(1.0, s["soil_nitrogen"] + cfg.natural_nitrogen_recovery)

        # Storage spoilage
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
                s["stored_crop_type"] = CropType.FALLOW
                s["stored_age_months"] = 0
            else:
                s["stored_amount"] = remaining

        # Interest accrual
        if s["has_active_loan"] and s["debt"] > 0:
            s["debt"] *= 1.0 + s["loan_interest_rate"] / 12.0

        # Monthly fixed cost
        s["cash"] -= s["inflated_monthly_fixed_cost"]

        # Storage cost (future)
        if cfg.enable_storage_cost and s["stored_amount"] > 0:
            s["cash"] -= cfg.cost_storage_monthly * s["stored_amount"]

        # New prices and weather
        s["expected_rainfall"] = generate_rainfall(s["month"], cfg, self._rng)
        prev_prices = s["prices"]
        s["prices"] = generate_market_prices(
            s["month"], cfg, self._rng,
            prev_prices=prev_prices,
            effective_base_prices=tuple(s["inflated_base_market_prices"]),
        )

        # Interest rate
        optimal_water = (
            cfg.optimal_water_level[s["active_crop_type"]]
            if s["active_crop_type"] != CropType.FALLOW
            else 0.0
        )
        s["interest_rate"] = calculate_interest_rate(
            cfg.base_interest_rate, s["month"],
            s["expected_rainfall"], optimal_water,
        )

        # Reset per-step flags
        s["irrigated"] = False
        s["fertilized"] = False

        return messages

    def apply_inflation(self) -> None:
        """Apply compounding inflation to all inflatable values."""
        s = self._s
        factor = 1.0 + self.config.inflation_rate
        s["inflated_seed_costs"] = [c * factor for c in s["inflated_seed_costs"]]
        s["inflated_cost_irrigate"] *= factor
        s["inflated_cost_fertilize"] *= factor
        s["inflated_loan_chunk"] *= factor
        s["inflated_base_land_price"] *= factor
        s["inflated_monthly_fixed_cost"] *= factor
        s["inflated_base_market_prices"] = [
            p * factor for p in s["inflated_base_market_prices"]
        ]

    # ──────────────────────────────────────────────────────────────
    # Net worth
    # ──────────────────────────────────────────────────────────────

    def compute_net_worth(self) -> float:
        """net_worth = cash + land_value + stored_value + growing_value − debt"""
        s = self._s
        cfg = self.config

        land_value = s["inflated_base_land_price"] * s["soil_nitrogen"]

        stored_value = 0.0
        if s["stored_amount"] > 0 and s["stored_crop_type"] != CropType.FALLOW:
            stored_value = s["stored_amount"] * s["prices"][s["stored_crop_type"] - 1]

        growing_value = 0.0
        if s["active_crop_type"] != CropType.FALLOW:
            est_yield = calculate_yield(
                s["active_crop_type"], s["crop_age_months"],
                s["soil_nitrogen"], s["water_level"],
                s["planting_month"] or s["month"], cfg, rng=None,
            )
            growing_value = est_yield * s["prices"][s["active_crop_type"] - 1]

        return s["cash"] + land_value + stored_value + growing_value - s["debt"]

    def compute_terminal_value(self) -> float:
        """Terminal profit: final_value − initial_value."""
        cfg = self.config
        initial = cfg.initial_cash + (cfg.base_land_price * cfg.initial_soil_nitrogen)
        return self.compute_net_worth() - initial
