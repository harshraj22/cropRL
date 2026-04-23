"""
CropRL Core Environment.

Implements the OpenEnv Environment interface for the farm management
simulation. Orchestrates the step loop by delegating physics to the
dynamics engine.

Key design change (v2): Only the ``Wait`` action advances the calendar
month.  All other actions execute instantly within the current month.
The step counter increments on every action.
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
    realise_rainfall,
)
from cropRL.enums import ActionType, CropType
from cropRL.models import CroprlAction, CroprlObservation, CroprlState


class CroprlEnvironment(Environment[CroprlAction, CroprlObservation, CroprlState]):
    """
    Farm management RL environment.

    The agent manages a small Indian farm.  Each step the agent picks one
    of 11 discrete actions.  Only the ``Wait`` action advances the calendar
    month and triggers monthly dynamics (rainfall realisation, crop ageing,
    nitrogen drain, interest accrual, spoilage, fixed costs, and new
    weather/price generation).  All other actions execute instantly within
    the current month.
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        task_id: str = "default",
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.task_id = task_id

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
        """Start a new episode."""
        self._rng = np.random.default_rng(seed)
        cfg = self.config

        month = 1  # January
        step = 0

        # Generate stochastic values for month 1
        rainfall = generate_rainfall(month, cfg, self._rng)
        prices = generate_market_prices(month, cfg, self._rng)

        # Interest rate (no crop planted → optimal_water = 0.0)
        interest_rate = calculate_interest_rate(
            cfg.base_interest_rate, month, rainfall, 0.0
        )

        # Internal farm state
        self._internal = {
            "month": month, "step": step, "month_count": 0,
            "expected_rainfall": rainfall, "prices": prices, "interest_rate": interest_rate,
            "active_crop_type": CropType.FALLOW, "crop_age_months": 0, "planting_month": 0,
            "soil_nitrogen": cfg.initial_soil_nitrogen, "water_level": 0.0,
            "cash": cfg.initial_cash, "debt": 0.0, "has_active_loan": False, "loan_interest_rate": 0.0,
            "stored_crop_type": CropType.FALLOW, "stored_amount": 0.0, "stored_age_months": 0,
            "irrigated": False, "fertilized": False,
        }

        # Compute initial net worth for reward tracking
        self._internal["prev_net_worth"] = self._compute_net_worth()

        # Compute derived fields
        yield_potential = calculate_expected_yield_potential(
            crop_type=CropType.FALLOW,
            crop_age=0,
            soil_nitrogen=cfg.initial_soil_nitrogen,
            current_water_level=0.0,
            current_month=month,
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
            loan_interest_rate=0.0,
            current_month_count=0,
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
        Execute one step.

        1. Record previous net worth
        2. Execute the chosen action (instant effects)
        3. Increment step counter
        4. If action == Wait: advance monthly dynamics
        5. Check termination
        6. Calculate reward as Δ(net_worth) + penalties
        7. Return observation
        """
        cfg = self.config
        s = self._internal
        action_id = action.action_id
        messages: list[str] = []
        penalty = 0.0

        # ── 1. Execute action ──────────────────────────────────────
        s["irrigated"] = False
        s["fertilized"] = False

        if action_id == ActionType.NO_OP:
            messages.append("No-op.")

        elif action_id in (ActionType.PLANT_CORN, ActionType.PLANT_WHEAT, ActionType.PLANT_CHICKPEA):
            penalty, msg = self._do_plant(s, action_id)
            messages.append(msg)

        elif action_id == ActionType.IRRIGATE:
            penalty, msg = self._do_irrigate(s)
            messages.append(msg)

        elif action_id == ActionType.FERTILIZE:
            penalty, msg = self._do_fertilize(s)
            messages.append(msg)

        elif action_id == ActionType.HARVEST_STORE:
            penalty, msg = self._do_harvest_store(s, cfg)
            messages.append(msg)

        elif action_id == ActionType.HARVEST_SELL:
            penalty, msg = self._do_harvest_sell(s, cfg)
            messages.append(msg)

        elif action_id == ActionType.SELL_INVENTORY:
            penalty, msg = self._do_sell_inventory(s)
            messages.append(msg)

        elif action_id == ActionType.TAKE_LOAN:
            penalty, msg = self._do_take_loan(s)
            messages.append(msg)

        elif action_id == ActionType.REPAY_LOAN:
            penalty, msg = self._do_repay_loan(s)
            messages.append(msg)

        elif action_id == ActionType.POST_FORUM:
            messages.append("Posted to forum.")

        # ── 2. Increment step counter ──────────────────────────────
        s["step"] += 1

        # ── 3. If No-Op (Wait): advance monthly dynamics ─────────
        #    In multi-agent mode, month advance is handled by the
        #    MultiAgentCroprlEnv wrapper. This path is for backward
        #    compat with the single-agent CroprlEnvironment.
        if action_id == ActionType.NO_OP:
            month_messages = self._advance_month(s, cfg)
            messages.extend(month_messages)

        # ── 4. Check termination ───────────────────────────────────
        done = False
        terminal_bonus = 0.0

        if s["step"] >= cfg.max_steps or s["month_count"] >= cfg.max_months:
            done = True
            terminal_bonus = self._compute_terminal_value(s, cfg)
            messages.append(
                f"EPISODE COMPLETE! Terminal profit: ₹{terminal_bonus:,.0f}."
            )
        elif s["cash"] < 0 and s["has_active_loan"]:
            done = True
            penalty += cfg.bankruptcy_penalty
            messages.append(
                "BANKRUPTCY! Cash is negative and you have outstanding debt."
            )

        # ── 5. Calculate reward ────────────────────────────────────
        current_net_worth = self._compute_net_worth()
        if done and terminal_bonus != 0:
            # On terminal step, reward includes the profit calculation
            reward = terminal_bonus + penalty
        else:
            reward = (current_net_worth - s["prev_net_worth"]) + penalty
        s["prev_net_worth"] = current_net_worth

        # ── 6. Compute derived observation fields ──────────────────
        yield_potential = calculate_expected_yield_potential(
            s["active_crop_type"],
            s["crop_age_months"],
            s["soil_nitrogen"],
            s["water_level"],
            s["planting_month"] or s["month"],
            cfg,
        )

        # Update state object
        self._state.step_count = s["step"]
        self._state.irrigated_this_month = s["irrigated"]
        self._state.fertilized_this_month = s["fertilized"]
        self._state.previous_cash = s["cash"]
        self._state.has_active_loan = s["has_active_loan"]
        self._state.loan_interest_rate = s["loan_interest_rate"]
        self._state.current_month_count = s["month_count"]

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
    # Action handlers
    # ──────────────────────────────────────────────────────────────

    def _do_plant(self, s: dict, action_id: int) -> tuple[float, str]:
        """Execute a plant action. Returns (penalty, message)."""
        cfg = self.config
        crop_idx = action_id  # action 1→crop 1, 2→2, 3→3
        seed_cost = cfg.seed_costs[crop_idx]

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
        s["planting_month"] = s["month"]  # lock season at planting time
        return 0.0, (
            f"Planted {cfg.crop_names[crop_idx]}. Cost: ₹{seed_cost:,.0f}."
        )

    def _do_irrigate(self, s: dict) -> tuple[float, str]:
        """Execute irrigate action. Returns (penalty, message)."""
        cfg = self.config
        irrigate_cost = cfg.cost_irrigate

        if s["active_crop_type"] == CropType.FALLOW:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to irrigate — land is fallow."
            )
        if s["cash"] < irrigate_cost:
            return cfg.invalid_action_penalty, (
                f"INVALID: Not enough cash to irrigate "
                f"(need ₹{irrigate_cost:,.0f})."
            )

        s["cash"] -= irrigate_cost
        s["irrigated"] = True

        # Instant water level increase
        crop_t = s["active_crop_type"]
        s["water_level"] += cfg.irrigate_amount[crop_t]
        optimal = cfg.optimal_water_level[crop_t]
        s["water_level"] = min(s["water_level"], optimal)

        return 0.0, (
            f"Irrigated. Water level now {s['water_level']:.2f}. "
            f"Cost: ₹{irrigate_cost:,.0f}."
        )

    def _do_fertilize(self, s: dict) -> tuple[float, str]:
        """Execute fertilize action. Returns (penalty, message)."""
        cfg = self.config
        fert_cost = cfg.cost_fertilize

        if s["cash"] < fert_cost:
            return cfg.invalid_action_penalty, (
                f"INVALID: Not enough cash to fertilize "
                f"(need ₹{fert_cost:,.0f})."
            )

        s["cash"] -= fert_cost
        s["soil_nitrogen"] = min(
            1.0, s["soil_nitrogen"] + cfg.fertilize_nitrogen_boost
        )
        s["fertilized"] = True
        return 0.0, (
            f"Fertilized. Soil nitrogen boosted to "
            f"{s['soil_nitrogen']:.2f}. Cost: ₹{fert_cost:,.0f}."
        )

    def _do_harvest_store(
        self, s: dict, cfg: EnvConfig
    ) -> tuple[float, str]:
        """Execute Harvest & Store action. Returns (penalty, message)."""
        if s["active_crop_type"] == CropType.FALLOW or s["crop_age_months"] < 1:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to harvest — "
                "no crop planted or crop too young."
            )

        crop_type = s["active_crop_type"]
        harvested = calculate_yield(
            crop_type,
            s["crop_age_months"],
            s["soil_nitrogen"],
            s["water_level"],
            s["planting_month"],
            cfg,
            rng=self._rng,
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

        # Store new harvest
        s["stored_crop_type"] = crop_type
        s["stored_amount"] = harvested
        s["stored_age_months"] = 0

        # Reset land
        s["active_crop_type"] = CropType.FALLOW
        s["crop_age_months"] = 0
        s["planting_month"] = 0

        parts.append(
            f"Harvested {harvested:.1f} tons of {cfg.crop_names[crop_type]} "
            f"and stored it."
        )
        return 0.0, " ".join(parts)

    def _do_harvest_sell(
        self, s: dict, cfg: EnvConfig
    ) -> tuple[float, str]:
        """Execute Harvest & Sell action. Returns (penalty, message)."""
        if s["active_crop_type"] == CropType.FALLOW or s["crop_age_months"] < 1:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to harvest — "
                "no crop planted or crop too young."
            )

        crop_type = s["active_crop_type"]
        harvested = calculate_yield(
            crop_type,
            s["crop_age_months"],
            s["soil_nitrogen"],
            s["water_level"],
            s["planting_month"],
            cfg,
            rng=self._rng,
        )

        price = s["prices"][crop_type - 1]
        revenue = harvested * price
        s["cash"] += revenue

        # Reset land
        s["active_crop_type"] = CropType.FALLOW
        s["crop_age_months"] = 0
        s["planting_month"] = 0

        return 0.0, (
            f"Harvested {harvested:.1f} tons of {cfg.crop_names[crop_type]} "
            f"and sold at ₹{price:,.0f}/ton. Revenue: ₹{revenue:,.0f}."
        )

    def _do_sell_inventory(self, s: dict) -> tuple[float, str]:
        """Execute Sell Inventory action. Returns (penalty, message)."""
        cfg = self.config
        if s["stored_amount"] <= 0:
            return cfg.invalid_action_penalty, (
                "INVALID: Storage is empty — nothing to sell."
            )

        crop_t = s["stored_crop_type"]
        price = s["prices"][crop_t - 1]
        revenue = s["stored_amount"] * price
        s["cash"] += revenue
        msg = (
            f"Sold {s['stored_amount']:.1f} tons of "
            f"{cfg.crop_names[crop_t]} at ₹{price:,.0f}/ton. "
            f"Revenue: ₹{revenue:,.0f}."
        )
        s["stored_crop_type"] = CropType.FALLOW
        s["stored_amount"] = 0.0
        s["stored_age_months"] = 0
        return 0.0, msg

    def _do_take_loan(self, s: dict) -> tuple[float, str]:
        """Execute Take Loan action. Returns (penalty, message)."""
        cfg = self.config
        if s["has_active_loan"]:
            return cfg.invalid_action_penalty, (
                "INVALID: You already have an active loan. "
                "Repay it first before taking another."
            )

        loan_amount = cfg.loan_chunk
        s["cash"] += loan_amount
        s["debt"] += loan_amount
        s["has_active_loan"] = True
        # Lock the interest rate at loan origination
        s["loan_interest_rate"] = s["interest_rate"]
        return 0.0, (
            f"Took a loan of ₹{loan_amount:,.0f} at "
            f"{s['loan_interest_rate'] * 100:.1f}% annual. "
            f"Total debt: ₹{s['debt']:,.0f}."
        )

    def _do_repay_loan(self, s: dict) -> tuple[float, str]:
        """Execute Repay Loan action. Returns (penalty, message)."""
        cfg = self.config
        if not s["has_active_loan"]:
            return cfg.invalid_action_penalty, (
                "INVALID: No active loan to repay."
            )
        if s["cash"] < s["debt"]:
            return cfg.invalid_action_penalty, (
                f"INVALID: Not enough cash to repay full debt. "
                f"Need ₹{s['debt']:,.0f}, have ₹{s['cash']:,.0f}."
            )

        repay_amount = s["debt"]
        s["cash"] -= repay_amount
        s["debt"] = 0.0
        s["has_active_loan"] = False
        s["loan_interest_rate"] = 0.0
        return 0.0, (
            f"Repaid full loan of ₹{repay_amount:,.0f}. "
            f"You are now debt-free."
        )

    # ──────────────────────────────────────────────────────────────
    # Monthly dynamics (called only from Wait action)
    # ──────────────────────────────────────────────────────────────

    def _advance_month(self, s: dict, cfg: EnvConfig) -> list[str]:
        """
        Advance all monthly dynamics.  Called once per ``Wait`` action.

        Order of operations:
        1. Increment month & month_count
        2. Check & apply inflation (on year boundary)
        3. Realise rainfall
        4. Update water level (rain + consumption)
        5. Age crop & apply monthly nitrogen impact
        6. Natural nitrogen recovery
        7. Age storage & check spoilage
        8. Accrue interest on debt (locked rate)
        9. Deduct monthly fixed cost
        10. Generate new expected rainfall & market prices
        11. Update interest rate
        """
        messages: list[str] = []

        # 1. Increment month
        old_month = s["month"]
        s["month"] = (s["month"] % 12) + 1
        s["month_count"] += 1

        # 2. Realise rainfall
        realised = realise_rainfall(
            s["expected_rainfall"],
            cfg.weather_sigma_realisation,
            self._rng,
        )

        # 4. Update water level
        crop_t = s["active_crop_type"]
        s["water_level"] += realised
        if crop_t != CropType.FALLOW:
            s["water_level"] -= cfg.water_utilised_monthly[crop_t]
        s["water_level"] = max(0.0, s["water_level"])
        optimal = cfg.optimal_water_level[crop_t]
        s["water_level"] = min(s["water_level"], optimal)

        # 5. Age crop & monthly nitrogen impact
        if crop_t != CropType.FALLOW:
            s["crop_age_months"] += 1
            s["soil_nitrogen"] += cfg.monthly_nitrogen_impact[crop_t]
            s["soil_nitrogen"] = max(0.0, min(1.0, s["soil_nitrogen"]))

        # 6. Natural nitrogen recovery
        s["soil_nitrogen"] = min(
            1.0, s["soil_nitrogen"] + cfg.natural_nitrogen_recovery
        )

        # 7. Age storage & check spoilage
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

        # 8. Accrue interest (using locked rate from loan origination)
        if s["has_active_loan"] and s["debt"] > 0:
            monthly_rate = s["loan_interest_rate"] / 12.0
            s["debt"] *= 1.0 + monthly_rate

        # 9. Monthly fixed cost
        s["cash"] -= cfg.monthly_fixed_cost

        # [Future] Storage cost
        if cfg.enable_storage_cost and s["stored_amount"] > 0:
            storage_cost = cfg.cost_storage_monthly * s["stored_amount"]
            s["cash"] -= storage_cost

        # 10. Generate new expected rainfall & market prices
        s["expected_rainfall"] = generate_rainfall(s["month"], cfg, self._rng)
        prev_prices = s["prices"]
        s["prices"] = generate_market_prices(
            s["month"], cfg, self._rng,
            prev_prices=prev_prices,
        )

        # 11. Update interest rate
        optimal_water = (
            cfg.optimal_water_level[s["active_crop_type"]]
            if s["active_crop_type"] != CropType.FALLOW
            else 0.0
        )
        s["interest_rate"] = calculate_interest_rate(
            cfg.base_interest_rate, s["month"],
            s["expected_rainfall"], optimal_water,
        )

        return messages

    # Inflation removed.

    # ──────────────────────────────────────────────────────────────
    # Net worth & terminal value
    # ──────────────────────────────────────────────────────────────

    def _compute_net_worth(self) -> float:
        """
        Compute the agent's current net worth.

        net_worth = cash + land_value + stored_value + growing_crop_value − debt

        Used for the Δ(net_worth) reward signal (telescoping sum
        decomposition that is mathematically equivalent to the sparse
        final_value − initial_value objective).
        """
        s = self._internal
        cfg = self.config

        land_value = cfg.base_land_price * s["soil_nitrogen"]

        stored_value = 0.0
        if s["stored_amount"] > 0 and s["stored_crop_type"] != CropType.FALLOW:
            stored_value = s["stored_amount"] * s["prices"][s["stored_crop_type"] - 1]

        growing_value = 0.0
        if s["active_crop_type"] != CropType.FALLOW:
            est_yield = calculate_yield(
                s["active_crop_type"],
                s["crop_age_months"],
                s["soil_nitrogen"],
                s["water_level"],
                s["planting_month"] or s["month"],
                cfg,
                rng=None,  # deterministic estimate
            )
            growing_value = est_yield * s["prices"][s["active_crop_type"] - 1]

        return s["cash"] + land_value + stored_value + growing_value - s["debt"]

    def _compute_terminal_value(self, s: dict, cfg: EnvConfig) -> float:
        """
        Compute the terminal profit: final_value − initial_value.

        final_value  = cash + land_value + stored_value + unharvested_value − debt
        initial_value = initial_cash + (base_land_price × initial_soil_nitrogen)
        """
        final_value = self._compute_net_worth()
        initial_value = cfg.initial_cash + (cfg.base_land_price * cfg.initial_soil_nitrogen)
        return final_value - initial_value

    # ──────────────────────────────────────────────────────────────
    # Observation builder
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

        land_price = cfg.base_land_price * s["soil_nitrogen"]

        obs_dict = {
            "current_month": s["month"],
            "current_step": s["step"],
            "expected_rainfall": s["expected_rainfall"],
            "active_crop_type": s["active_crop_type"],
            "crop_age_months": s["crop_age_months"],
            "expected_yield_potential": yield_potential,
            "soil_nitrogen": s["soil_nitrogen"],
            "current_water_level": s["water_level"],
            "cash_balance": s["cash"],
            "current_debt": s["debt"],
            "current_interest_rate": s["interest_rate"],
            "current_land_price": land_price,
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
            # Build a copy with extra display-only fields for the text formatter
            text_dict = {**obs_dict, "monthly_fixed_cost": cfg.monthly_fixed_cost}
            text_summary = format_text_observation(
                text_dict, cfg, s["has_active_loan"], valid_actions
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
        valid = [ActionType.WAIT]  # Wait is always valid

        # Plant actions (1, 2, 3)
        if s["active_crop_type"] == CropType.FALLOW:
            for crop_idx in (CropType.CORN, CropType.WHEAT, CropType.CHICKPEA):
                if s["cash"] >= cfg.seed_costs[crop_idx]:
                    valid.append(crop_idx)

        # Irrigate (4)
        if (s["active_crop_type"] != CropType.FALLOW
                and s["cash"] >= cfg.cost_irrigate):
            valid.append(ActionType.IRRIGATE)

        # Fertilize (5)
        if s["cash"] >= cfg.cost_fertilize:
            valid.append(ActionType.FERTILIZE)

        # Harvest & Store (6), Harvest & Sell (7)
        if s["active_crop_type"] != CropType.FALLOW and s["crop_age_months"] >= 1:
            valid.append(ActionType.HARVEST_STORE)
            valid.append(ActionType.HARVEST_SELL)

        # Sell Inventory (8)
        if s["stored_amount"] > 0:
            valid.append(ActionType.SELL_INVENTORY)

        # Take Loan (9)
        if not s["has_active_loan"]:
            valid.append(ActionType.TAKE_LOAN)

        # Repay Loan (10)
        if s["has_active_loan"] and s["cash"] >= s["debt"]:
            valid.append(ActionType.REPAY_LOAN)

        return sorted(valid)
