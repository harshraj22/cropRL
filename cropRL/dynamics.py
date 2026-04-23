"""
CropRL Dynamics Engine.

All simulation physics live here, separate from the step() orchestration.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
from .config import EnvConfig
from .enums import MONTH_NAMES, Season, get_season


def _get_seasonal_baseline(month: int, config: EnvConfig) -> float:
    season = get_season(month)
    for s, b in config.weather_seasonal_baselines:
        if s == season:
            return b
    return 0.5

def generate_rainfall(month: int, config: EnvConfig, rng: np.random.Generator) -> float:
    baseline = _get_seasonal_baseline(month, config)
    noise = rng.normal(0.0, config.weather_sigma)
    noise = float(np.clip(noise, -3 * config.weather_sigma, 3 * config.weather_sigma))
    return float(np.clip(baseline + noise, 0.0, 1.0))

def realise_rainfall(expected: float, sigma: float, rng: np.random.Generator) -> float:
    noise = rng.normal(0.0, sigma)
    noise = float(np.clip(noise, -3 * sigma, 3 * sigma))
    return float(np.clip(expected + noise, 0.0, 1.0))

def calculate_interest_rate(base_rate: float, month: int, rainfall: float, optimal_water_level: float) -> float:
    if month in (6, 7):
        dl = 0.03
    elif month in (10, 11):
        dl = -0.02
    else:
        dl = 0.0
    w_deficit = max(0.0, optimal_water_level - rainfall)
    dr = 0.05 if w_deficit > 0.3 else 0.0
    return max(0.0, base_rate + dl + dr)

def _get_market_seasonal_multiplier(month: int, config: EnvConfig) -> float:
    season = get_season(month)
    for s, m in config.market_seasonal_multipliers:
        if s == season:
            return m
    return 1.0

def generate_market_prices(month: int, config: EnvConfig, rng: np.random.Generator,
                           prev_prices: Optional[Tuple[float, float, float]] = None) -> Tuple[float, float, float]:
    sm = _get_market_seasonal_multiplier(month, config)
    bp = config.base_market_prices
    prices = []
    use_rw = config.enable_price_autocorrelation and prev_prices is not None
    for i in range(1, 4):
        base = bp[i]; target = base * sm
        noise = rng.normal(0.0, config.market_price_sigma)
        noise = float(np.clip(noise, -3 * config.market_price_sigma, 3 * config.market_price_sigma))
        if use_rw:
            drift = config.price_reversion_speed * (target - prev_prices[i-1]) / max(target, 1.0)
            price = prev_prices[i-1] * (1.0 + drift + noise)
        else:
            price = target * (1.0 + noise)
        prices.append(float(np.clip(price, base * config.price_min_multiplier, base * config.price_max_multiplier)))
    if config.demand_shock_probability > 0 and rng.random() < config.demand_shock_probability:
        ci = rng.integers(0, 3); d = rng.choice([-1, 1])
        lo, hi = config.demand_shock_magnitude; mag = rng.uniform(lo, hi)
        base = bp[ci+1]
        prices[ci] = float(np.clip(prices[ci] * (1.0 + d * mag), base * config.price_min_multiplier, base * config.price_max_multiplier))
    return (prices[0], prices[1], prices[2])

def calculate_supply_adjusted_price(base_price: float, num_sellers: int, alpha: float) -> float:
    """effective_price = base_price / (1 + alpha * max(0, num_sellers - 1))"""
    return base_price / (1.0 + alpha * max(0, num_sellers - 1))

def _maturity_factor(crop_age: int, growth_months: int) -> float:
    if crop_age == 0: return 0.0
    if crop_age < growth_months: return (crop_age / growth_months) ** 2
    return max(0.0, 1.0 - 0.5 * (crop_age - growth_months))

def _nitrogen_factor(soil_nitrogen: float, min_req: float) -> float:
    if soil_nitrogen <= 0: return 0.0
    if min_req <= 0: return 1.0
    if soil_nitrogen < min_req: return 0.3 * (soil_nitrogen / min_req)
    ratio = min((soil_nitrogen - min_req) / (1.0 - min_req), 1.0)
    return 0.3 + 0.7 * (1.0 - (1.0 - ratio) ** 2)

def _water_factor(current: float, optimal: float) -> float:
    if optimal <= 0: return 1.0
    if current >= optimal: return 1.0
    return max(0.1, math.sqrt(max(0.0, current / optimal)))

def _season_factor(month: int, crop_type: int, config: EnvConfig) -> float:
    if crop_type == 0: return 1.0
    if get_season(month) in config.optimal_seasons_per_crop[crop_type]: return 1.0
    return config.non_optimal_season_multiplier

def calculate_yield(crop_type: int, crop_age: int, soil_nitrogen: float,
                    current_water_level: float, current_month: int,
                    config: EnvConfig, rng: Optional[np.random.Generator] = None) -> float:
    if crop_type == 0: return 0.0
    y = (config.base_yield_tons[crop_type] *
         _maturity_factor(crop_age, config.growth_months[crop_type]) *
         _nitrogen_factor(soil_nitrogen, config.minimum_nitrogen_requirement[crop_type]) *
         _water_factor(current_water_level, config.optimal_water_level[crop_type]) *
         _season_factor(current_month, crop_type, config))
    if rng is not None and config.yield_sigma > 0:
        noise = float(np.clip(rng.normal(0.0, config.yield_sigma), -3*config.yield_sigma, 3*config.yield_sigma))
        y *= (1.0 + noise)
    return max(0.0, y)

def calculate_expected_yield_potential(crop_type: int, crop_age: int, soil_nitrogen: float,
                                       current_water_level: float, current_month: int, config: EnvConfig) -> float:
    if crop_type == 0: return 0.0
    raw = calculate_yield(crop_type, crop_age, soil_nitrogen, current_water_level, current_month, config, rng=None)
    mx = config.base_yield_tons[crop_type]
    return float(np.clip(raw / mx, 0.0, 1.0)) if mx > 0 else 0.0

def apply_spoilage(stored_age: int, stored_amount: float, max_age: int) -> Tuple[float, bool]:
    if stored_amount > 0 and stored_age > max_age: return 0.0, True
    return stored_amount, False

def format_text_observation(obs_dict: dict, config: EnvConfig, has_active_loan: bool,
                            valid_actions: list[int] | None = None) -> str:
    month = obs_dict["current_month"]; ms = obs_dict.get("month_step", 0)
    spm = config.steps_per_agent_per_month; season = get_season(month)
    ct = obs_dict["active_crop_type"]; fid = obs_dict.get("farmer_id", 0)
    lines = [
        f"===== Farmer {fid} (Step {obs_dict['current_step']}, Sub {ms}/{spm}) =====",
        f"Month: {MONTH_NAMES[month]} | Season: {season.value}",
        f"Rainfall: {obs_dict['expected_rainfall']:.2f}",
        "",
    ]
    if ct == 0:
        lines.append("Crop: Fallow")
    else:
        lines.append(f"Crop: {config.crop_names[ct]} | Age: {obs_dict['crop_age_months']}/{config.growth_months[ct]}mo")
    lines.append(f"Soil N: {obs_dict['soil_nitrogen']:.2f} | Water: {obs_dict['current_water_level']:.2f} | Yield Pot: {obs_dict['expected_yield_potential']:.2f}")
    oc = obs_dict.get("other_farmers_crops", [])
    if oc:
        lines.append("Others: " + ", ".join(config.crop_names[c] for c in oc))
    elif ct == 0:
        lines.append("Others: (fallow — hidden)")
    lines.append(f"Cash: {obs_dict['cash_balance']:,.0f} | Debt: {obs_dict['current_debt']:,.0f}" + (" (loan)" if has_active_loan else ""))
    lines.append(f"Prices — Corn: {obs_dict['market_price_crop_1']:,.0f} Wheat: {obs_dict['market_price_crop_2']:,.0f} Chick: {obs_dict['market_price_crop_3']:,.0f}")
    st = obs_dict["stored_crop_type"]; sa = obs_dict["stored_amount"]
    if st > 0 and sa > 0:
        lines.append(f"Storage: {sa:.1f}t {config.crop_names[st]} (age {obs_dict['stored_age_months']}/{config.max_storage_age})")
    fm = obs_dict.get("forum_messages", [])
    if fm:
        lines.append("Forum: " + " | ".join(fm))
    if valid_actions is not None:
        lines.append("Actions: " + " ".join(f"{a}:{config.action_names[a]}" for a in valid_actions))
    return "\n".join(lines)
