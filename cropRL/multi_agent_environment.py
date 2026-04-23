"""
MultiAgentCroprlEnvironment — the OpenEnv-compatible multi-agent environment.

Implements ``Environment[MultiAgentAction, MultiAgentObservation, MultiAgentState]``.
Manages N ``FarmState`` instances (plain state containers, NOT OpenEnv environments).

Key responsibilities:
- Route agent actions to the correct FarmState.
- Gate month advancement behind the slot-based TimeController.
- Intercept sell actions to queue them through the MarketEngine (batch clearing).
- Maintain the PublicLedger and Forum for inter-agent information.
- Emit MultiAgentObservation combining private farm state with shared world state.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from cropRL.config import EnvConfig, MultiAgentConfig
from cropRL.dynamics import (
    calculate_expected_yield_potential,
    format_text_observation,
)
from cropRL.enums import ActionType, CropType, LedgerEventType
from cropRL.farm_state import FarmState
from cropRL.market_engine import MarketEngine
from cropRL.models import (
    LedgerEvent,
    MultiAgentAction,
    MultiAgentObservation,
    MultiAgentResult,
    MultiAgentState,
)
from cropRL.public_ledger import Forum, PublicLedger
from cropRL.time_controller import TimeController


class MultiAgentCroprlEnvironment(
    Environment[MultiAgentAction, MultiAgentObservation, MultiAgentState]
):
    """
    Multi-agent farm management environment implementing the OpenEnv interface.

    N agents each own a private ``FarmState`` (their farm).
    A shared ``TimeController`` synchronises the calendar month using a
    slot-based budget: the month advances only when every agent has
    used all slots.

    Sell actions (HARVEST_SELL, SELL_INVENTORY) are **deferred**: they are
    queued in the ``MarketEngine`` and cleared at month-end, so collective
    sell volume affects the clearing price for all sellers that month.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        ma_config: Optional[MultiAgentConfig] = None,
        task_id: str = "multi_default",
    ) -> None:
        super().__init__()
        self._env_cfg = env_config or EnvConfig()
        self._ma_cfg = ma_config or MultiAgentConfig()
        self._task_id = task_id

        n = self._ma_cfg.num_agents

        # Shared infrastructure
        self._time_ctrl = TimeController(n, self._ma_cfg.action_slots_per_month)
        self._ledger = PublicLedger()
        self._forum = Forum(n, self._ma_cfg.forum_messages_per_month, self._ledger)

        # Per-farm state (initialised in reset)
        self._farms: List[FarmState] = []
        self._market: Optional[MarketEngine] = None
        self._shared_rng: Optional[np.random.Generator] = None
        self._hype_statuses: list = []
        self._last_realised: Tuple[float, ...] = (
            self._env_cfg.base_market_prices[1],
            self._env_cfg.base_market_prices[2],
            self._env_cfg.base_market_prices[3],
        )

        self.episode_id: str = ""
        self._state = MultiAgentState(
            num_agents=n, task_id=task_id,
        )

    # ──────────────────────────────────────────────────────────────
    # OpenEnv interface: reset
    # ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MultiAgentObservation:
        """
        Reset the environment and return initial observation (agent 0).

        Callers should use ``get_obs(agent_id)`` to get each agent's
        initial observation before the episode loop.
        """
        self.episode_id = episode_id or str(uuid4())
        base_seed = seed if seed is not None else 42
        n = self._ma_cfg.num_agents

        self._shared_rng = np.random.default_rng(base_seed)

        # Create N independent FarmStates with unique seeds
        self._farms = []
        for i in range(n):
            rng_i = np.random.default_rng(base_seed + i + 1)
            farm = FarmState(config=self._env_cfg, rng=rng_i)
            self._farms.append(farm)

        # Reset shared state
        self._time_ctrl.reset()
        self._ledger.reset_month()
        self._forum.reset_month()

        # Fresh MarketEngine
        self._market = MarketEngine(
            self._ma_cfg, self._env_cfg, self._shared_rng
        )
        self._hype_statuses = self._market.hype_statuses()

        # Update state
        self._state = MultiAgentState(
            num_agents=n,
            current_month=self._farms[0].month,
            month_count=0,
            episode_id=self.episode_id,
            task_id=self._task_id,
        )

        return self._build_ma_obs(0, "Episode started.", 0.0, False)

    # ──────────────────────────────────────────────────────────────
    # OpenEnv interface: step
    # ──────────────────────────────────────────────────────────────

    def step(
        self,
        action: MultiAgentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiAgentObservation:
        """
        Execute one step for the agent identified by ``action.agent_id``.

        The underlying month advances only when all agents have exhausted
        their slot budgets.
        """
        agent_id = action.agent_id
        n = self._ma_cfg.num_agents
        if agent_id < 0 or agent_id >= n:
            raise ValueError(f"Invalid agent_id {agent_id}; expected 0..{n-1}")

        # Guard: out of slots
        if self._time_ctrl.slots_remaining(agent_id) <= 0:
            return self._build_ma_obs(
                agent_id,
                "You have exhausted your action slots for this month. Waiting for others.",
                self._env_cfg.invalid_action_penalty,
                False,
            )

        action_id = action.action_id
        farm = self._farms[agent_id]
        s = farm.s
        penalty = 0.0
        messages: List[str] = []

        # ── Handle Wait / No-Op (action 0) ────────────────────────────
        if action_id == ActionType.WAIT:
            self._ledger.record(LedgerEvent(
                agent_id=agent_id,
                month=self._current_month(),
                slot=self._time_ctrl.current_slot_for(agent_id),
                event_type=LedgerEventType.WAIT,
            ))
            messages.append("You waited / took no action this slot.")

        # ── Execute action ────────────────────────────────────────
        slot = self._time_ctrl.current_slot_for(agent_id)

        if action_id == ActionType.WAIT:
            pass  # Already handled above
        
        elif action_id == ActionType.POST_MESSAGE:
            penalty, msg = self._do_post_message(agent_id, slot, action)
            messages.append(msg)

        elif action_id in (
            ActionType.PLANT_CORN, ActionType.PLANT_WHEAT, ActionType.PLANT_CHICKPEA,
        ):
            penalty, msg = farm.do_plant(action_id)
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.PLANTED,
                    payload={"crop_type": action_id},
                ))

        elif action_id in (ActionType.PLANT_MATCHA, ActionType.PLANT_QUINOA,
                           ActionType.PLANT_TURMERIC):
            crop_map = {
                ActionType.PLANT_MATCHA: CropType.MATCHA,
                ActionType.PLANT_QUINOA: CropType.QUINOA,
                ActionType.PLANT_TURMERIC: CropType.TURMERIC,
            }
            penalty, msg = farm.do_plant_hype(crop_map[action_id])
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.PLANTED,
                    payload={"crop_type": int(crop_map[action_id])},
                ))

        elif action_id == ActionType.IRRIGATE:
            penalty, msg = farm.do_irrigate()
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.IRRIGATED,
                ))

        elif action_id == ActionType.FERTILIZE:
            penalty, msg = farm.do_fertilize()
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.FERTILIZED,
                ))

        elif action_id == ActionType.HARVEST_STORE:
            penalty, msg, old_crop_type, old_volume = farm.do_harvest_store()
            messages.append(msg)
            if penalty == 0.0:
                if old_volume > 0 and self._market is not None:
                    self._market.queue_sell(agent_id, old_crop_type, old_volume, is_inventory=True)
                
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.HARVESTED_STORED,
                    payload={
                        "crop_type": s["stored_crop_type"],
                        "amount": s["stored_amount"],
                    },
                ))

        elif action_id == ActionType.HARVEST_SELL:
            penalty, msg, crop_t, volume = farm.do_harvest_sell_queued()
            messages.append(msg)
            if penalty == 0.0 and self._market:
                self._market.queue_sell(agent_id, crop_t, volume, is_inventory=False)
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.HARVESTED_SOLD,
                    payload={"crop_type": crop_t, "amount": round(volume, 2)},
                ))

        elif action_id == ActionType.SELL_INVENTORY:
            penalty, msg, crop_t, volume = farm.do_sell_inventory_queued()
            messages.append(msg)
            if penalty == 0.0 and self._market:
                self._market.queue_sell(agent_id, crop_t, volume, is_inventory=True)
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.SOLD_INVENTORY,
                    payload={"crop_type": crop_t, "amount": round(volume, 2)},
                ))

        elif action_id == ActionType.TAKE_LOAN:
            penalty, msg = farm.do_take_loan()
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.LOAN_TAKEN,
                ))

        elif action_id == ActionType.REPAY_LOAN:
            penalty, msg = farm.do_repay_loan()
            messages.append(msg)
            if penalty == 0.0:
                self._ledger.record(LedgerEvent(
                    agent_id=agent_id, month=self._current_month(), slot=slot,
                    event_type=LedgerEventType.LOAN_REPAID,
                ))

        else:
            messages.append(f"INVALID: Unknown action id {action_id}.")
            penalty = self._env_cfg.invalid_action_penalty

        # ── Consume slot ──────────────────────────────────────────
        self._time_ctrl.consume_slot(agent_id)
        s["step"] += 1

        # ── Auto-advance month if all agents exhausted budgets ────────────
        if self._time_ctrl.all_done():
            adv_msgs = self._do_advance_month()
            messages.extend(adv_msgs)

        done = self._check_termination(farm)
        return self._build_ma_obs(agent_id, " | ".join(messages), penalty, done)

    # ──────────────────────────────────────────────────────────────
    # OpenEnv interface: state
    # ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> MultiAgentState:
        """Return current environment state."""
        self._state.current_month = self._current_month()
        if self._farms:
            self._state.month_count = self._farms[0].month_count
        return self._state

    # ──────────────────────────────────────────────────────────────
    # Public: get_obs
    # ──────────────────────────────────────────────────────────────

    def get_obs(self, agent_id: int) -> MultiAgentObservation:
        """Get the current observation for a specific agent."""
        done = self._check_termination(self._farms[agent_id])
        return self._build_ma_obs(agent_id, "", 0.0, done)

    # ──────────────────────────────────────────────────────────────
    # Grading
    # ──────────────────────────────────────────────────────────────

    def compute_result(self, trajectories: Optional[Dict[int, list]] = None) -> MultiAgentResult:
        """Compute final per-agent scores and aggregate metrics."""
        from cropRL.tasks import grader, TASKS

        base_task = self._task_id
        for _n in (2, 4, 8):
            base_task = base_task.replace(f"_{_n}agent", "")
        if base_task not in TASKS:
            base_task = "medium"

        agent_scores: Dict[int, float] = {}
        net_worths: Dict[int, float] = {}

        for i, farm in enumerate(self._farms):
            nw = farm.compute_net_worth()
            net_worths[i] = nw
            traj = (trajectories or {}).get(i, [])
            bankrupt = farm.cash < 0 and farm.has_active_loan
            score = grader(base_task, nw, bankrupt, traj)
            agent_scores[i] = float(score)

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
    # Forum action handler
    # ──────────────────────────────────────────────────────────────

    def _do_post_message(
        self, agent_id: int, slot: int, action: MultiAgentAction,
    ) -> Tuple[float, str]:
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
        3. Advance all farms' monthly physics.
        4. Update shared prices from market engine.
        5. Reset ledger / forum for the new month.
        6. Advance TimeController.
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
                self._farms[agent_id].s["cash"] += rev
                messages.append(
                    f"Agent {agent_id} received ₹{rev:,.0f} from market clearing."
                )

        # 3. Advance physics for every farm (in lockstep)
        for i, farm in enumerate(self._farms):
            farm_msgs = farm.advance_month()
            for m in farm_msgs:
                messages.append(f"[Farm {i}] {m}")

        # 4. Update shared prices from market engine
        new_prices = self._market.generate_base_prices(
            month=self._current_month(),
            inflated_base_prices=list(
                self._farms[0].s["inflated_base_market_prices"]
            ),
        )
        for farm in self._farms:
            farm.s["prices"] = tuple(new_prices[1:])

        # 5. Reset ledger and forum
        self._ledger.reset_month()
        self._forum.reset_month()

        # 6. Advance TimeController
        self._time_ctrl.advance_month()

        messages.insert(0, "=== Month advanced. ===")
        return messages

    # ──────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────

    def _build_ma_obs(
        self, agent_id: int, message: str, reward: float, done: bool,
    ) -> MultiAgentObservation:
        """Construct a MultiAgentObservation for agent *agent_id*."""
        farm = self._farms[agent_id]
        s = farm.s
        cfg = self._env_cfg

        slot = self._time_ctrl.current_slot_for(agent_id)

        yield_potential = calculate_expected_yield_potential(
            s["active_crop_type"], s["crop_age_months"],
            s["soil_nitrogen"], s["water_level"],
            s["planting_month"] or s["month"], cfg,
        )

        land_price = s["inflated_base_land_price"] * s["soil_nitrogen"]
        other_crops = self._ledger.planted_crops_this_month(before_slot=slot)

        # Text summary
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
            market_price_crop_4=s["prices"][3] if len(s["prices"]) > 3 else 0.0,
            market_price_crop_5=s["prices"][4] if len(s["prices"]) > 4 else 0.0,
            market_price_crop_6=s["prices"][5] if len(s["prices"]) > 5 else 0.0,
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
        self, agent_id: int, obs_dict: dict, slot: int,
        other_crops: Dict[int, int],
    ) -> str:
        """Append multi-agent sections to the standard text observation."""
        cfg = self._env_cfg
        base = format_text_observation(
            obs_dict, cfg, self._farms[agent_id].has_active_loan
        )

        lines = [base, "", "=== MULTI-AGENT ==="]
        lines.append(
            f"Agent: {agent_id} | Slot: {slot}/{self._ma_cfg.action_slots_per_month} "
            f"| Slots remaining: {self._time_ctrl.slots_remaining(agent_id)}"
        )

        if other_crops:
            lines.append("")
            lines.append("NEIGHBOURS (crops planted this month so far):")
            for aid, ct in other_crops.items():
                lines.append(f"  Agent {aid}: {cfg.crop_names[ct]}")

        if self._hype_statuses:
            lines.append("")
            lines.append("SOCIAL MEDIA TRENDS (Hype Crops):")
            for hs in self._hype_statuses:
                bar = "█" * int(hs.hype_level * 10) + "░" * (10 - int(hs.hype_level * 10))
                lines.append(
                    f"  {hs.crop_name}: [{bar}] {hs.hype_level:.0%} ({hs.phase.value})"
                )

        msgs = self._forum.messages_this_month()
        if msgs:
            lines.append("")
            lines.append("FORUM:")
            for m in msgs:
                lines.append(f"  Agent {m.agent_id}: {m.text}")

        if self._last_realised:
            lines.append("")
            lines.append("LAST MONTH CLEARING PRICES:")
            names = self._cfg.crop_names[1:]
            for name, price in zip(names, self._last_realised):
                lines.append(f"  {name}: ₹{price:,.0f}/ton")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────

    def _current_month(self) -> int:
        return self._farms[0].month if self._farms else 1

    def get_turn_order(self) -> List[int]:
        """Return the current month's agent turn order, accounting for the rotating offset."""
        n = self._ma_cfg.num_agents
        offset = self._time_ctrl._first_agent_offset
        return [(i + offset) % n for i in range(n)]

    def _check_termination(self, farm: FarmState) -> bool:
        s = farm.s
        cfg = self._env_cfg
        if s["step"] >= cfg.max_steps:
            return True
        if s["month_count"] >= cfg.max_months:
            return True
        if s["cash"] < 0 and s["has_active_loan"]:
            return True
        return False

    @staticmethod
    def _gini(values: List[float]) -> float:
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
