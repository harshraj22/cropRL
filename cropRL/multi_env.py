"""
Multi-Agent CropRL Environment.

Wraps N independent farmer states with shared weather/market/forum.
Two-phase step: immediate actions execute now, deferred (sell/plant) resolve at month end.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np
from .config import EnvConfig
from .dynamics import (apply_spoilage, calculate_expected_yield_potential,
    calculate_interest_rate, calculate_supply_adjusted_price, calculate_yield,
    format_text_observation, generate_market_prices, generate_rainfall, realise_rainfall)
from .enums import ActionType, CropType
from .models import CroprlAction, CroprlObservation

DEFERRED_ACTIONS = frozenset({
    ActionType.PLANT_CORN, ActionType.PLANT_WHEAT, ActionType.PLANT_CHICKPEA,
    ActionType.HARVEST_SELL, ActionType.SELL_INVENTORY,
})

def _new_farmer(cfg: EnvConfig) -> dict:
    return {"active_crop_type": CropType.FALLOW, "crop_age_months": 0, "planting_month": 0,
            "soil_nitrogen": cfg.initial_soil_nitrogen, "water_level": 0.0,
            "cash": cfg.initial_cash, "debt": 0.0, "has_active_loan": False,
            "loan_interest_rate": 0.0, "stored_crop_type": CropType.FALLOW,
            "stored_amount": 0.0, "stored_age_months": 0, "irrigated": False, "fertilized": False}

class MultiAgentCroprlEnv:
    def __init__(self, config: Optional[EnvConfig] = None, task_id: str = "default"):
        self.config = config or EnvConfig()
        self.task_id = task_id
        self._rng: Optional[np.random.Generator] = None
        self._shared: Dict[str, Any] = {}
        self._farmers: Dict[int, dict] = {}
        self._forum: List[str] = []
        self._deferred: Dict[int, List[int]] = {}
        self._substep: Dict[int, int] = {}
        self._phase2_done: set = set()
        self._global_step = 0
        self._prev_nw: Dict[int, float] = {}
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Dict[int, CroprlObservation]:
        cfg = self.config
        self._rng = np.random.default_rng(seed)
        self._done = False; self._global_step = 0
        month = 1
        self._shared = {"month": month, "month_count": 0,
            "expected_rainfall": generate_rainfall(month, cfg, self._rng),
            "prices": generate_market_prices(month, cfg, self._rng),
            "interest_rate": calculate_interest_rate(cfg.base_interest_rate, month,
                generate_rainfall(month, cfg, self._rng), 0.0)}
        self._farmers = {}; self._forum = []; self._deferred = {}
        self._substep = {}; self._phase2_done = set(); self._prev_nw = {}
        obs = {}
        for fid in range(cfg.num_farmers):
            self._farmers[fid] = _new_farmer(cfg)
            self._deferred[fid] = []; self._substep[fid] = 0
            self._prev_nw[fid] = self._net_worth(fid)
            obs[fid] = self._obs(fid, 0.0, False, "New episode started!")
        return obs

    def obs(self, farmer_id: int) -> CroprlObservation:
        return self._obs(farmer_id, 0.0, self._done, "")

    def step(self, action: Optional[CroprlAction], farmer_id: int, end: bool = False) -> CroprlObservation:
        if self._done:
            return self._obs(farmer_id, 0.0, True, "Episode ended.")
        return self._phase2(farmer_id) if end else self._phase1(action, farmer_id)

    def _phase1(self, action: CroprlAction, farmer_id: int) -> CroprlObservation:
        cfg = self.config; fs = self._farmers[farmer_id]; aid = action.action_id
        msgs: List[str] = []; pen = 0.0
        if self._substep[farmer_id] >= cfg.steps_per_agent_per_month:
            return self._obs(farmer_id, cfg.invalid_action_penalty, False, "INVALID: Sub-steps exhausted.")
        if aid in DEFERRED_ACTIONS:
            self._deferred[farmer_id].append(aid)
            msgs.append(f"Queued {cfg.action_names[aid]} for end-of-month.")
        else:
            pen, msg = self._exec_imm(action, farmer_id)
            msgs.append(msg)
        self._substep[farmer_id] += 1; self._global_step += 1
        r = self._delta(farmer_id, pen)
        return self._obs(farmer_id, r, False, " | ".join(msgs))

    def _exec_imm(self, action: CroprlAction, fid: int) -> tuple[float, str]:
        cfg = self.config; fs = self._farmers[fid]; aid = action.action_id
        if aid == ActionType.NO_OP: return 0.0, "No-op."
        if aid == ActionType.IRRIGATE:
            if fs["active_crop_type"] == CropType.FALLOW: return cfg.invalid_action_penalty, "INVALID: Fallow."
            if fs["cash"] < cfg.cost_irrigate: return cfg.invalid_action_penalty, "INVALID: No cash."
            fs["cash"] -= cfg.cost_irrigate; ct = fs["active_crop_type"]
            fs["water_level"] = min(fs["water_level"] + cfg.irrigate_amount[ct], cfg.optimal_water_level[ct])
            return 0.0, f"Irrigated. Water={fs['water_level']:.2f}"
        if aid == ActionType.FERTILIZE:
            if fs["cash"] < cfg.cost_fertilize: return cfg.invalid_action_penalty, "INVALID: No cash."
            fs["cash"] -= cfg.cost_fertilize
            fs["soil_nitrogen"] = min(1.0, fs["soil_nitrogen"] + cfg.fertilize_nitrogen_boost)
            return 0.0, f"Fertilized. N={fs['soil_nitrogen']:.2f}"
        if aid == ActionType.HARVEST_STORE:
            if fs["active_crop_type"] == CropType.FALLOW or fs["crop_age_months"] < 1:
                return cfg.invalid_action_penalty, "INVALID: Nothing to harvest."
            ct = fs["active_crop_type"]
            harvested = calculate_yield(ct, fs["crop_age_months"], fs["soil_nitrogen"],
                fs["water_level"], fs["planting_month"], cfg, rng=self._rng)
            parts = []
            if fs["stored_amount"] > 0:
                rev = fs["stored_amount"] * self._shared["prices"][fs["stored_crop_type"]-1]
                fs["cash"] += rev; parts.append(f"Auto-sold stored for {rev:,.0f}")
            fs["stored_crop_type"] = ct; fs["stored_amount"] = harvested; fs["stored_age_months"] = 0
            fs["active_crop_type"] = CropType.FALLOW; fs["crop_age_months"] = 0; fs["planting_month"] = 0
            parts.append(f"Harvested {harvested:.1f}t {cfg.crop_names[ct]}, stored.")
            return 0.0, " ".join(parts)
        if aid == ActionType.TAKE_LOAN:
            if fs["has_active_loan"]: return cfg.invalid_action_penalty, "INVALID: Has loan."
            fs["cash"] += cfg.loan_chunk; fs["debt"] += cfg.loan_chunk
            fs["has_active_loan"] = True; fs["loan_interest_rate"] = self._shared["interest_rate"]
            return 0.0, f"Took loan {cfg.loan_chunk:,.0f}"
        if aid == ActionType.REPAY_LOAN:
            if not fs["has_active_loan"]: return cfg.invalid_action_penalty, "INVALID: No loan."
            if fs["cash"] < fs["debt"]: return cfg.invalid_action_penalty, "INVALID: Not enough cash."
            fs["cash"] -= fs["debt"]; fs["debt"] = 0.0; fs["has_active_loan"] = False; fs["loan_interest_rate"] = 0.0
            return 0.0, "Repaid loan."
        if aid == ActionType.POST_FORUM:
            txt = (action.forum_message or "")[:cfg.forum_message_max_chars]
            if txt: self._forum.append(f"Farmer {fid}: {txt}")
            return 0.0, f"Posted to forum." if txt else "Empty forum post."
        return cfg.invalid_action_penalty, f"Unknown action {aid}"

    def _phase2(self, farmer_id: int) -> CroprlObservation:
        cfg = self.config; msgs: List[str] = []; pen = 0.0
        sc = self._count_sells()
        for aid in self._deferred[farmer_id]:
            p, m = self._exec_def(farmer_id, aid, sc); pen += p; msgs.append(m)
        self._deferred[farmer_id] = []; self._phase2_done.add(farmer_id)
        if len(self._phase2_done) == cfg.num_farmers:
            msgs.extend(self._advance_month())
            self._phase2_done.clear()
            for fid in range(cfg.num_farmers): self._substep[fid] = 0
        done = False; tb = 0.0
        if self._shared["month_count"] >= cfg.max_months:
            done = True; self._done = True; tb = self._terminal(farmer_id)
            msgs.append(f"EPISODE COMPLETE! Profit: {tb:,.0f}")
        elif self._farmers[farmer_id]["cash"] < 0 and self._farmers[farmer_id]["has_active_loan"]:
            done = True; self._done = True; pen += cfg.bankruptcy_penalty; msgs.append("BANKRUPTCY!")
        r = self._delta(farmer_id, pen + tb)
        return self._obs(farmer_id, r, done, " | ".join(msgs))

    def _count_sells(self) -> Dict[int, int]:
        c: Dict[int, int] = defaultdict(int)
        for fid, q in self._deferred.items():
            for a in q:
                if a == ActionType.HARVEST_SELL:
                    ct = self._farmers[fid]["active_crop_type"]
                    if ct != CropType.FALLOW: c[ct] += 1
                elif a == ActionType.SELL_INVENTORY:
                    fs = self._farmers[fid]
                    if fs["stored_crop_type"] != CropType.FALLOW and fs["stored_amount"] > 0:
                        c[fs["stored_crop_type"]] += 1
        return c

    def _exec_def(self, fid: int, aid: int, sc: Dict[int, int]) -> tuple[float, str]:
        cfg = self.config; fs = self._farmers[fid]
        if aid in (ActionType.PLANT_CORN, ActionType.PLANT_WHEAT, ActionType.PLANT_CHICKPEA):
            ci = aid; cost = cfg.seed_costs[ci]
            if fs["active_crop_type"] != CropType.FALLOW: return cfg.invalid_action_penalty, "INVALID: Land occupied."
            if fs["cash"] < cost: return cfg.invalid_action_penalty, "INVALID: No cash to plant."
            fs["cash"] -= cost; fs["active_crop_type"] = ci; fs["crop_age_months"] = 0
            fs["planting_month"] = self._shared["month"]
            return 0.0, f"Planted {cfg.crop_names[ci]}."
        if aid == ActionType.HARVEST_SELL:
            if fs["active_crop_type"] == CropType.FALLOW or fs["crop_age_months"] < 1:
                return cfg.invalid_action_penalty, "INVALID: Nothing to harvest."
            ct = fs["active_crop_type"]
            h = calculate_yield(ct, fs["crop_age_months"], fs["soil_nitrogen"],
                fs["water_level"], fs["planting_month"], cfg, rng=self._rng)
            bp = self._shared["prices"][ct-1]
            p = calculate_supply_adjusted_price(bp, sc.get(ct, 1), cfg.supply_price_alpha)
            fs["cash"] += h * p; fs["active_crop_type"] = CropType.FALLOW
            fs["crop_age_months"] = 0; fs["planting_month"] = 0
            return 0.0, f"Sold {h:.1f}t {cfg.crop_names[ct]} @{p:,.0f}/t (supply-adj)."
        if aid == ActionType.SELL_INVENTORY:
            if fs["stored_amount"] <= 0: return cfg.invalid_action_penalty, "INVALID: Empty storage."
            ct = fs["stored_crop_type"]; bp = self._shared["prices"][ct-1]
            p = calculate_supply_adjusted_price(bp, sc.get(ct, 1), cfg.supply_price_alpha)
            rev = fs["stored_amount"] * p; fs["cash"] += rev
            msg = f"Sold {fs['stored_amount']:.1f}t {cfg.crop_names[ct]} @{p:,.0f}/t."
            fs["stored_crop_type"] = CropType.FALLOW; fs["stored_amount"] = 0.0; fs["stored_age_months"] = 0
            return 0.0, msg
        return cfg.invalid_action_penalty, "Unknown deferred."

    def _advance_month(self) -> List[str]:
        cfg = self.config; sh = self._shared; msgs: List[str] = []
        sh["month"] = (sh["month"] % 12) + 1; sh["month_count"] += 1
        realised = realise_rainfall(sh["expected_rainfall"], cfg.weather_sigma_realisation, self._rng)
        for fid in range(cfg.num_farmers):
            fs = self._farmers[fid]; fs["irrigated"] = False; fs["fertilized"] = False
            ct = fs["active_crop_type"]
            fs["water_level"] += realised
            if ct != CropType.FALLOW: fs["water_level"] -= cfg.water_utilised_monthly[ct]
            fs["water_level"] = max(0.0, min(fs["water_level"], cfg.optimal_water_level[ct]))
            if ct != CropType.FALLOW:
                fs["crop_age_months"] += 1
                fs["soil_nitrogen"] = max(0.0, min(1.0, fs["soil_nitrogen"] + cfg.monthly_nitrogen_impact[ct]))
            fs["soil_nitrogen"] = min(1.0, fs["soil_nitrogen"] + cfg.natural_nitrogen_recovery)
            if fs["stored_amount"] > 0:
                fs["stored_age_months"] += 1
                rem, spoiled = apply_spoilage(fs["stored_age_months"], fs["stored_amount"], cfg.max_storage_age)
                if spoiled:
                    msgs.append(f"Farmer {fid}: SPOILAGE!")
                    fs["stored_amount"] = 0.0; fs["stored_crop_type"] = CropType.FALLOW; fs["stored_age_months"] = 0
                else: fs["stored_amount"] = rem
            if fs["has_active_loan"] and fs["debt"] > 0:
                fs["debt"] *= 1.0 + fs["loan_interest_rate"] / 12.0
            fs["cash"] -= cfg.monthly_fixed_cost
        sh["expected_rainfall"] = generate_rainfall(sh["month"], cfg, self._rng)
        sh["prices"] = generate_market_prices(sh["month"], cfg, self._rng, prev_prices=sh["prices"])
        sh["interest_rate"] = calculate_interest_rate(cfg.base_interest_rate, sh["month"], sh["expected_rainfall"], 0.0)
        self._forum = []
        return msgs

    def _net_worth(self, fid: int) -> float:
        cfg = self.config; fs = self._farmers[fid]; sh = self._shared
        lv = cfg.base_land_price * fs["soil_nitrogen"]
        sv = fs["stored_amount"] * sh["prices"][fs["stored_crop_type"]-1] if fs["stored_amount"] > 0 and fs["stored_crop_type"] != CropType.FALLOW else 0.0
        gv = 0.0
        if fs["active_crop_type"] != CropType.FALLOW:
            ey = calculate_yield(fs["active_crop_type"], fs["crop_age_months"], fs["soil_nitrogen"],
                fs["water_level"], fs["planting_month"] or sh["month"], cfg, rng=None)
            gv = ey * sh["prices"][fs["active_crop_type"]-1]
        return fs["cash"] + lv + sv + gv - fs["debt"]

    def _terminal(self, fid: int) -> float:
        cfg = self.config
        return self._net_worth(fid) - (cfg.initial_cash + cfg.base_land_price * cfg.initial_soil_nitrogen)

    def _delta(self, fid: int, extra: float = 0.0) -> float:
        cur = self._net_worth(fid); prev = self._prev_nw[fid]; self._prev_nw[fid] = cur
        return (cur - prev) + extra

    def _obs(self, fid: int, reward: float, done: bool, message: str) -> CroprlObservation:
        cfg = self.config; fs = self._farmers[fid]; sh = self._shared
        oc = [self._farmers[f]["active_crop_type"] for f in range(cfg.num_farmers) if f != fid] if fs["active_crop_type"] != CropType.FALLOW else []
        yp = calculate_expected_yield_potential(fs["active_crop_type"], fs["crop_age_months"],
            fs["soil_nitrogen"], fs["water_level"], fs["planting_month"] or sh["month"], cfg)
        od = {"farmer_id": fid, "current_month": sh["month"], "current_step": self._global_step,
              "month_step": self._substep.get(fid, 0), "expected_rainfall": sh["expected_rainfall"],
              "active_crop_type": fs["active_crop_type"], "crop_age_months": fs["crop_age_months"],
              "expected_yield_potential": yp, "soil_nitrogen": fs["soil_nitrogen"],
              "current_water_level": fs["water_level"], "cash_balance": fs["cash"],
              "current_debt": fs["debt"], "current_interest_rate": sh["interest_rate"],
              "current_land_price": cfg.base_land_price * fs["soil_nitrogen"],
              "market_price_crop_1": sh["prices"][0], "market_price_crop_2": sh["prices"][1],
              "market_price_crop_3": sh["prices"][2], "cost_seed_1": cfg.seed_costs[1],
              "cost_seed_2": cfg.seed_costs[2], "cost_seed_3": cfg.seed_costs[3],
              "cost_irrigate": cfg.cost_irrigate, "cost_fertilize": cfg.cost_fertilize,
              "stored_crop_type": fs["stored_crop_type"], "stored_amount": fs["stored_amount"],
              "stored_age_months": fs["stored_age_months"], "other_farmers_crops": oc,
              "forum_messages": list(self._forum), "message": message}
        ts = ""
        if cfg.text_mode:
            ts = format_text_observation({**od, "monthly_fixed_cost": cfg.monthly_fixed_cost}, cfg, fs["has_active_loan"])
        return CroprlObservation(**od, text_summary=ts, done=done, reward=reward)
