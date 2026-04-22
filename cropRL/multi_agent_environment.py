"""
MultiAgentCroprlEnvironment — orchestrator for N competing/cooperating farms.

Sits on top of N independent CroprlEnvironment instances without modifying
the underlying dynamics.  Key responsibilities:

- Route agent actions to the correct inner farm.
- Gate month advancement behind the slot-based TimeController.
- Intercept sell actions to queue them through the MarketEngine (batch clearing).
- Maintain the PublicLedger and Forum for inter-agent information.
- Emit MultiAgentObservation combining private farm state with shared world state.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from cropRL.config import EnvConfig, MultiAgentConfig
from cropRL.dynamics import (
    apply_spoilage,
    calculate_expected_yield_potential,
    calculate_yield,
    format_text_observation,
)
from cropRL.enums import ActionType, CropType, LedgerEventType
from cropRL.market_engine import MarketEngine
from cropRL.models import (
    LedgerEvent,
    MultiAgentAction,
    MultiAgentObservation,
    MultiAgentResult,
)
from cropRL.public_ledger import Forum, PublicLedger
from cropRL.time_controller import TimeController, TurnOverError
from cropRL.server.cropRL_environment import CroprlEnvironment


class MultiAgentCroprlEnvironment:
    """
    Multi-agent farm management environment.

    N agents each own a private ``CroprlEnvironment`` (their farm).
    A shared ``TimeController`` synchronises the calendar month using a
    slot-based budget: the month advances only when every agent has
    either used all slots or called End Turn (action 0).

    Sell actions (HARVEST_SELL, SELL_INVENTORY) are **deferred**: they are
    queued in the ``MarketEngine`` and cleared at month-end, so collective
    sell volume affects the clearing price for all sellers that month.
    """

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        ma_config: Optional[MultiAgentConfig] = None,
        task_id: str = "multi_default",
    ) -> None:
        self._env_cfg = env_config or EnvConfig()
        self._ma_cfg = ma_config or MultiAgentConfig()
        self._task_id = task_id

        n = self._ma_cfg.num_agents

        # N independent farms
        self._farms: List[CroprlEnvironment] = [
            CroprlEnvironment(config=self._env_cfg, task_id=task_id)
            for _ in range(n)
        ]

        # Shared infrastructure (initialised in reset())
        self._time_ctrl: TimeController = TimeController(
            n, self._ma_cfg.action_slots_per_month
        )
        self._ledger: PublicLedger = PublicLedger()
        self._forum: Forum = Forum(
            n, self._ma_cfg.forum_messages_per_month, self._ledger
        )

        # Market engine needs an rng — created fresh in reset()
        self._market: Optional[MarketEngine] = None
        self._shared_rng: Optional[np.random.Generator] = None

        # Per-agent pending revenue (credited after resolve_month)
        self._pending_revenue: Dict[int, float] = {i: 0.0 for i in range(n)}

        # Hype statuses cache (updated each month)
        self._hype_statuses = []

        # Last month's realised prices (Corn, Wheat, Chickpea tuple)
        self._last_realised: Tuple[float, ...] = (
            self._env_cfg.base_market_prices[1],
            self._env_cfg.base_market_prices[2],
            self._env_cfg.base_market_prices[3],
        )

        self.episode_id: str = ""

    # ──────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[int, MultiAgentObservation]:
        """
        Start a new multi-agent episode.

        Returns
        -------
        dict
            agent_id → initial MultiAgentObservation.
        """
        self.episode_id = episode_id or str(uuid4())
        base_seed = seed if seed is not None else 42

        # Shared RNG (for MarketEngine)
        self._shared_rng = np.random.default_rng(base_seed)

        # Reset each farm with a unique but deterministic seed
        for i, farm in enumerate(self._farms):
            farm.reset(seed=base_seed + i + 1, episode_id=self.episode_id)

        # Reset shared state
        self._time_ctrl.reset()
        self._ledger.reset_month()
        self._forum.reset_month()
        self._pending_revenue = {i: 0.0 for i in range(self._ma_cfg.num_agents)}

        # Initialise a fresh MarketEngine
        self._market = MarketEngine(
            self._ma_cfg, self._env_cfg, self._shared_rng
        )
        self._hype_statuses = self._market.hype_statuses()

        return {i: self._build_ma_obs(i, "Episode started.", 0.0, False)
                for i in range(self._ma_cfg.num_agents)}

    def step(
        self,
        agent_id: int,
        action: MultiAgentAction,
    ) -> MultiAgentObservation:
        """
        Execute one step for a single agent.

        The underlying month advances only when all agents have signalled
        End Turn (action 0) or exhausted their slot budgets.
        """
        n = self._ma_cfg.num_agents
        if agent_id < 0 or agent_id >= n:
            raise ValueError(f"Invalid agent_id {agent_id}; expected 0..{n-1}")

        # ── Guard: already ended turn ─────────────────────────────
        if self._time_ctrl.is_turn_done(agent_id):
            return self._build_ma_obs(
                agent_id,
                "You already ended your turn this month. Waiting for others.",
                self._env_cfg.invalid_action_penalty,
                False,
            )

        action_id = action.action_id
        farm = self._farms[agent_id]
        s = farm._internal
        penalty = 0.0
        messages: List[str] = []

        # ── Handle End Turn (action 0) ────────────────────────────
        if action_id == ActionType.WAIT:  # == END_TURN
            self._ledger.record(LedgerEvent(
                agent_id=agent_id,
                month=self._current_month(),
                slot=self._time_ctrl.current_slot_for(agent_id),
                event_type=LedgerEventType.END_TURN,
            ))
            all_done = self._time_ctrl.submit_turn_end(agent_id)
            messages.append("Turn ended for this month.")

            if all_done:
                month_msgs = self._do_advance_month()
                messages.extend(month_msgs)

            done = self._check_termination(farm)
            return self._build_ma_obs(agent_id, " | ".join(messages), penalty, done)

        # ── Guard: no slots remaining ─────────────────────────────
        if self._time_ctrl.slots_remaining(agent_id) <= 0:
            # Auto-end turn (budget exhausted)
            all_done = self._time_ctrl.submit_turn_end(agent_id)
            messages.append("Action budget exhausted — turn auto-ended.")
            if all_done:
                messages.extend(self._do_advance_month())
            return self._build_ma_obs(agent_id, " | ".join(messages), penalty, False)

        # ── Execute action ────────────────────────────────────────
        slot = self._time_ctrl.current_slot_for(agent_id)

        if action_id == ActionType.POST_MESSAGE:   # 11
            penalty, msg = self._do_post_message(agent_id, slot, action)
            messages.append(msg)

        elif action_id in (
            ActionType.PLANT_CORN,
            ActionType.PLANT_WHEAT,
            ActionType.PLANT_CHICKPEA,
        ):
            penalty, msg = farm._do_plant(s, action_id)
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id,
                    month=self._current_month(),
                    slot=slot,
                    event_type=LedgerEventType.PLANTED,
                    payload={"crop_type": action_id},
                ))

        elif action_id in (ActionType.PLANT_MATCHA, ActionType.PLANT_QUINOA,
                           ActionType.PLANT_TURMERIC):
            # Hype crop planting — map action id to crop id
            crop_map = {
                ActionType.PLANT_MATCHA:   CropType.MATCHA,
                ActionType.PLANT_QUINOA:   CropType.QUINOA,
                ActionType.PLANT_TURMERIC: CropType.TURMERIC,
            }
            penalty, msg = self._do_plant_hype(farm, s, crop_map[action_id])
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id,
                    month=self._current_month(),
                    slot=slot,
                    event_type=LedgerEventType.PLANTED,
                    payload={"crop_type": int(crop_map[action_id])},
                ))

        elif action_id == ActionType.IRRIGATE:
            penalty, msg = farm._do_irrigate(s)
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id,
                    month=self._current_month(),
                    slot=slot,
                    event_type=LedgerEventType.IRRIGATED,
                ))

        elif action_id == ActionType.FERTILIZE:
            penalty, msg = farm._do_fertilize(s)
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id,
                    month=self._current_month(),
                    slot=slot,
                    event_type=LedgerEventType.FERTILIZED,
                ))

        elif action_id == ActionType.HARVEST_STORE:
            penalty, msg = farm._do_harvest_store(s, self._env_cfg)
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id,
                    month=self._current_month(),
                    slot=slot,
                    event_type=LedgerEventType.HARVESTED_STORED,
                    payload={
                        "crop_type": s["stored_crop_type"],
                        "amount": s["stored_amount"],
                    },
                ))

        elif action_id == ActionType.HARVEST_SELL:
            penalty, msg = self._do_harvest_sell_queued(farm, s, agent_id, slot)
            messages.append(msg)

        elif action_id == ActionType.SELL_INVENTORY:
            penalty, msg = self._do_sell_inventory_queued(s, agent_id, slot)
            messages.append(msg)

        elif action_id == ActionType.TAKE_LOAN:
            penalty, msg = farm._do_take_loan(s)
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id,
                    month=self._current_month(),
                    slot=slot,
                    event_type=LedgerEventType.LOAN_TAKEN,
                ))

        elif action_id == ActionType.REPAY_LOAN:
            penalty, msg = farm._do_repay_loan(s)
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id,
                    month=self._current_month(),
                    slot=slot,
                    event_type=LedgerEventType.LOAN_REPAID,
                ))

        else:
            messages.append(f"INVALID: Unknown action id {action_id}.")
            penalty = self._env_cfg.invalid_action_penalty

        # ── Consume slot ──────────────────────────────────────────
        self._time_ctrl.consume_slot(agent_id)

        # ── Step counter on inner farm ────────────────────────────
        s["step"] += 1

        # ── Auto-end turn if budget just exhausted ────────────────
        if self._time_ctrl.is_turn_done(agent_id) and self._time_ctrl.all_done():
            adv_msgs = self._do_advance_month()
            messages.extend(adv_msgs)

        done = self._check_termination(farm)
        return self._build_ma_obs(agent_id, " | ".join(messages), penalty, done)

    # ──────────────────────────────────────────────────────────────
    # Multi-agent grading
    # ──────────────────────────────────────────────────────────────

    def compute_result(self, trajectories: Optional[Dict[int, list]] = None) -> MultiAgentResult:
        """
        Compute final per-agent scores and aggregate metrics.
        """
        from cropRL.tasks import grader, TASKS

        # Determine base task for grading (strip multi-agent prefix)
        base_task = self._task_id.replace("_4agent", "").replace("_8agent", "")
        if base_task not in TASKS:
            base_task = "medium"

        agent_scores: Dict[int, float] = {}
        net_worths: Dict[int, float] = {}

        for i, farm in enumerate(self._farms):
            s = farm._internal
            nw = farm._compute_net_worth()
            net_worths[i] = nw
            traj = (trajectories or {}).get(i, [])
            bankrupt = s["cash"] < 0 and s["has_active_loan"]
            score = grader(base_task, nw, bankrupt, traj)
            agent_scores[i] = float(score)

        # Aggregate score based on objective mode
        mode = self._ma_cfg.objective_mode
        if mode == "cooperative":
            agg = float(np.mean(list(agent_scores.values())))
        elif mode == "mixed":
            w = self._ma_cfg.mixed_mode_village_weight
            village_avg = float(np.mean(list(agent_scores.values())))
            agg_scores = {
                i: (1 - w) * agent_scores[i] + w * village_avg
                for i in agent_scores
            }
            agg = float(np.mean(list(agg_scores.values())))
            agent_scores = agg_scores
        else:  # competitive
            agg = float(max(agent_scores.values()))

        winner = (
            max(agent_scores, key=agent_scores.get)  # type: ignore
            if mode != "cooperative"
            else None
        )

        gini = self._gini(list(net_worths.values()))
        total_nw = float(sum(net_worths.values()))

        return MultiAgentResult(
            agent_scores=agent_scores,
            aggregate_score=float(np.clip(agg, 0.01, 0.99)),
            winner_agent_id=winner,
            gini_coefficient=gini,
            total_village_nw=total_nw,
        )

    # ──────────────────────────────────────────────────────────────
    # Specialised action handlers
    # ──────────────────────────────────────────────────────────────

    def _do_plant_hype(
        self, farm: CroprlEnvironment, s: dict, crop_type: CropType
    ) -> Tuple[float, str]:
        """Plant a hype crop — reuses the same logic as standard planting."""
        cfg = self._env_cfg
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

    def _do_harvest_sell_queued(
        self,
        farm: CroprlEnvironment,
        s: dict,
        agent_id: int,
        slot: int,
    ) -> Tuple[float, str]:
        """
        Harvest and queue the sale for month-end clearing.
        Revenue is NOT credited immediately — it comes from resolve_month().
        """
        cfg = self._env_cfg
        if s["active_crop_type"] == CropType.FALLOW or s["crop_age_months"] < 1:
            return cfg.invalid_action_penalty, (
                "INVALID: Nothing to harvest — no crop planted or crop too young."
            )

        crop_type = s["active_crop_type"]
        harvested = calculate_yield(
            crop_type,
            s["crop_age_months"],
            s["soil_nitrogen"],
            s["water_level"],
            s["planting_month"],
            cfg,
            rng=farm._rng,
        )

        # Reset land
        s["active_crop_type"] = CropType.FALLOW
        s["crop_age_months"] = 0
        s["planting_month"] = 0

        # Queue the sell
        if self._market:
            self._market.queue_sell(agent_id, crop_type, harvested, is_inventory=False)

        self._ledger.record(LedgerEvent(
            agent_id=agent_id,
            month=self._current_month(),
            slot=slot,
            event_type=LedgerEventType.HARVESTED_SOLD,
            payload={"crop_type": crop_type, "amount": round(harvested, 2)},
        ))

        return 0.0, (
            f"Harvested {harvested:.1f} tons of {cfg.crop_names[crop_type]}. "
            f"Sale queued for month-end market clearing."
        )

    def _do_sell_inventory_queued(
        self,
        s: dict,
        agent_id: int,
        slot: int,
    ) -> Tuple[float, str]:
        """
        Queue an inventory sell order for month-end clearing.
        Revenue is NOT credited immediately.
        """
        cfg = self._env_cfg
        if s["stored_amount"] <= 0:
            return cfg.invalid_action_penalty, (
                "INVALID: Storage is empty — nothing to sell."
            )

        crop_t = s["stored_crop_type"]
        volume = s["stored_amount"]

        # Clear storage
        s["stored_crop_type"] = CropType.FALLOW
        s["stored_amount"] = 0.0
        s["stored_age_months"] = 0

        # Queue sell
        if self._market:
            self._market.queue_sell(agent_id, crop_t, volume, is_inventory=True)

        self._ledger.record(LedgerEvent(
            agent_id=agent_id,
            month=self._current_month(),
            slot=slot,
            event_type=LedgerEventType.SOLD_INVENTORY,
            payload={"crop_type": crop_t, "amount": round(volume, 2)},
        ))

        return 0.0, (
            f"Queued {volume:.1f} tons of {cfg.crop_names[crop_t]} "
            f"for month-end market clearing."
        )

    def _do_post_message(
        self,
        agent_id: int,
        slot: int,
        action: MultiAgentAction,
    ) -> Tuple[float, str]:
        """Post a message to the public forum."""
        text = action.forum_message or "(no message)"
        success, msg = self._forum.post(
            agent_id=agent_id,
            month=self._current_month(),
            slot=slot,
            text=text,
        )
        if not success:
            return self._env_cfg.invalid_action_penalty, msg
        return 0.0, msg

    # ──────────────────────────────────────────────────────────────
    # Month advancement
    # ──────────────────────────────────────────────────────────────

    def _do_advance_month(self) -> List[str]:
        """
        Called when all agents have ended their turn.

        1. Resolve market (collect revenues, tick hype).
        2. Credit revenues to each farm.
        3. Advance all inner farms' monthly physics.
        4. Reset ledger / forum for the new month.
        5. Advance TimeController.
        """
        assert self._market is not None
        messages: List[str] = []

        # 1. Resolve market clearing
        revenues = self._market.resolve_month(self._current_month())
        self._last_realised = self._market.last_month_realised_prices
        self._hype_statuses = self._market.hype_statuses()

        # 2. Credit revenue to each farm
        for agent_id, rev in revenues.items():
            if rev > 0:
                self._farms[agent_id]._internal["cash"] += rev
                messages.append(
                    f"Agent {agent_id} received ₹{rev:,.0f} from market clearing."
                )

        # 3. Advance physics for every farm (in lockstep)
        for i, farm in enumerate(self._farms):
            farm_msgs = farm._advance_month(farm._internal, self._env_cfg)
            for m in farm_msgs:
                messages.append(f"[Farm {i}] {m}")

        # 4. Update shared prices from market engine base prices
        new_prices = self._market.generate_base_prices(
            month=self._current_month(),
            inflated_base_prices=list(
                self._farms[0]._internal["inflated_base_market_prices"]
            ),
        )
        for farm in self._farms:
            # Overlay shared prices (crops 1-3) onto each farm's prices
            farm._internal["prices"] = (
                new_prices[1], new_prices[2], new_prices[3]
            )

        # 5. Reset ledger and forum for next month
        self._ledger.reset_month()
        self._forum.reset_month()

        # 6. Advance TimeController
        self._time_ctrl.advance_month()

        messages.insert(0, f"=== Month advanced. ===")
        return messages

    # ──────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────

    def _build_ma_obs(
        self,
        agent_id: int,
        message: str,
        reward: float,
        done: bool,
    ) -> MultiAgentObservation:
        """Construct a MultiAgentObservation for agent *agent_id*."""
        farm = self._farms[agent_id]
        s = farm._internal
        cfg = self._env_cfg

        slot = self._time_ctrl.current_slot_for(agent_id)

        yield_potential = calculate_expected_yield_potential(
            s["active_crop_type"],
            s["crop_age_months"],
            s["soil_nitrogen"],
            s["water_level"],
            s["planting_month"] or s["month"],
            cfg,
        )

        land_price = s["inflated_base_land_price"] * s["soil_nitrogen"]

        # What other agents have planted so far this month
        other_crops = self._ledger.planted_crops_this_month(before_slot=slot)

        # Text summary (if text_mode enabled)
        text_summary = ""
        if cfg.text_mode:
            obs_dict_for_text = {
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
                "cost_seed_1": s["inflated_seed_costs"][1],
                "cost_seed_2": s["inflated_seed_costs"][2],
                "cost_seed_3": s["inflated_seed_costs"][3],
                "cost_irrigate": s["inflated_cost_irrigate"],
                "cost_fertilize": s["inflated_cost_fertilize"],
                "stored_crop_type": s["stored_crop_type"],
                "stored_amount": s["stored_amount"],
                "stored_age_months": s["stored_age_months"],
                "message": message,
                "monthly_fixed_cost": s["inflated_monthly_fixed_cost"],
            }
            text_summary = self._format_ma_text(
                agent_id, obs_dict_for_text, slot, other_crops
            )

        return MultiAgentObservation(
            # Base CroprlObservation fields
            current_month=s["month"],
            current_step=s["step"],
            expected_rainfall=s["expected_rainfall"],
            active_crop_type=s["active_crop_type"],
            crop_age_months=s["crop_age_months"],
            expected_yield_potential=yield_potential,
            soil_nitrogen=s["soil_nitrogen"],
            current_water_level=s["water_level"],
            cash_balance=s["cash"],
            current_debt=s["debt"],
            current_interest_rate=s["interest_rate"],
            current_land_price=land_price,
            market_price_crop_1=s["prices"][0],
            market_price_crop_2=s["prices"][1],
            market_price_crop_3=s["prices"][2],
            cost_seed_1=s["inflated_seed_costs"][1],
            cost_seed_2=s["inflated_seed_costs"][2],
            cost_seed_3=s["inflated_seed_costs"][3],
            cost_irrigate=s["inflated_cost_irrigate"],
            cost_fertilize=s["inflated_cost_fertilize"],
            stored_crop_type=s["stored_crop_type"],
            stored_amount=s["stored_amount"],
            stored_age_months=s["stored_age_months"],
            message=message,
            text_summary=text_summary,
            done=done,
            reward=reward,
            # Multi-agent extensions
            agent_id=agent_id,
            month_slot=slot,
            slots_remaining=self._time_ctrl.slots_remaining(agent_id),
            forum_posts_remaining=self._forum.posts_remaining(agent_id),
            other_agents_crops={k: v for k, v in other_crops.items() if k != agent_id},
            ledger_this_month=self._ledger.events_before_slot(slot),
            forum_this_month=self._forum.messages_this_month(),
            last_month_realised_prices=self._last_realised,
            hype_crop_statuses=self._hype_statuses,
        )

    def _format_ma_text(
        self,
        agent_id: int,
        obs_dict: dict,
        slot: int,
        other_crops: Dict[int, int],
    ) -> str:
        """Append multi-agent sections to the standard text observation."""
        cfg = self._env_cfg
        base = format_text_observation(
            obs_dict, cfg, self._farms[agent_id]._internal["has_active_loan"]
        )

        lines = [base, "", "=== MULTI-AGENT ==="]
        lines.append(f"Agent: {agent_id} | Slot: {slot}/{self._ma_cfg.action_slots_per_month} "
                     f"| Slots remaining: {self._time_ctrl.slots_remaining(agent_id)}")

        # Neighbours' planted crops (visible after they plant)
        if other_crops:
            lines.append("")
            lines.append("NEIGHBOURS (crops planted this month so far):")
            for aid, ct in other_crops.items():
                lines.append(f"  Agent {aid}: {cfg.crop_names[ct]}")

        # Hype crop status
        if self._hype_statuses:
            lines.append("")
            lines.append("SOCIAL MEDIA TRENDS (Hype Crops):")
            for hs in self._hype_statuses:
                bar = "█" * int(hs.hype_level * 10) + "░" * (10 - int(hs.hype_level * 10))
                lines.append(
                    f"  {hs.crop_name}: [{bar}] {hs.hype_level:.0%} ({hs.phase.value})"
                )

        # Forum messages
        msgs = self._forum.messages_this_month()
        if msgs:
            lines.append("")
            lines.append("FORUM:")
            for m in msgs:
                lines.append(f"  Agent {m.agent_id}: {m.text}")

        # Last month's realised prices
        if self._last_realised:
            lines.append("")
            lines.append("LAST MONTH CLEARING PRICES:")
            names = ["Corn", "Wheat", "Chickpea"]
            for name, price in zip(names, self._last_realised):
                lines.append(f"  {name}: ₹{price:,.0f}/ton")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────

    def _current_month(self) -> int:
        """Return the current calendar month from the first farm (they stay in sync)."""
        return self._farms[0]._internal.get("month", 1)

    def _check_termination(self, farm: CroprlEnvironment) -> bool:
        """Return True if *any* termination condition is met for this farm."""
        s = farm._internal
        cfg = self._env_cfg
        if s["step"] >= cfg.max_steps:
            return True
        if s["month_count"] >= cfg.max_months:
            return True
        if s["cash"] < 0 and s["has_active_loan"]:
            return True  # bankruptcy
        return False

    @staticmethod
    def _gini(values: List[float]) -> float:
        """Compute the Gini coefficient of a list of net worths."""
        if not values or len(values) < 2:
            return 0.0
        arr = sorted(float(max(v, 0)) for v in values)
        n = len(arr)
        total = sum(arr)
        if total <= 0:
            return 0.0
        cum = 0.0
        for i, v in enumerate(arr):
            cum += (2 * (i + 1) - n - 1) * v
        return cum / (n * total)
