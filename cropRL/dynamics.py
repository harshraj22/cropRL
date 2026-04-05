"""
CropRL Dynamics Engine.

All simulation physics live here, separate from the step() orchestration.
Each function is pure (given inputs → deterministic output for a given rng state),
making them independently unit-testable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .config import EnvConfig
from .enums import MONTH_NAMES, Season, get_season


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. Weather Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_seasonal_baseline(month: int, config: EnvConfig) -> float:
    """Return the seasonal rainfall baseline μ(m) for the given month."""
    season = get_season(month)
    for cfg_season, baseline in config.weather_seasonal_baselines:
        if cfg_season == season:
            return baseline
    # Fallback (should not happen with well-configured baselines)
    return 0.5


def generate_rainfall(
    month: int, config: EnvConfig, rng: np.random.Generator
) -> float:
    """
    Generate expected (forecasted) seasonal rainfall with Gaussian noise.

    W_expected = clip(μ(m) + ε, 0, 1)  where ε ~ N(0, weather_sigma²)

    This is the forecast shown to the agent. The actual realised rainfall
    is sampled separately via ``realise_rainfall()`` when the month ticks.
    """
    baseline = _get_seasonal_baseline(month, config)
    noise = rng.normal(0.0, config.weather_sigma)
    noise = float(np.clip(noise, -3 * config.weather_sigma,
                          3 * config.weather_sigma))
    return float(np.clip(baseline + noise, 0.0, 1.0))


def realise_rainfall(
    expected: float,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    """
    Sample actual rainfall close to the expected forecast.

    W_actual = clip(W_expected + ε, 0, 1)  where ε ~ N(0, σ_realisation²)

    Called once per month when the ``wait`` action triggers a month advance.
    """
    noise = rng.normal(0.0, sigma)
    noise = float(np.clip(noise, -3 * sigma, 3 * sigma))
    return float(np.clip(expected + noise, 0.0, 1.0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. Dynamic Interest Rate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculate_interest_rate(
    base_rate: float,
    month: int,
    rainfall: float,
    optimal_water_level: float,
) -> float:
    """
    Calculate the current annual interest rate.

    R = R_base + Δ_liquidity(m) + Δ_risk(W_deficit)

    Parameters
    ----------
    base_rate : float
        Base annual interest rate (e.g. 0.08).
    month : int
        Calendar month 1-12.
    rainfall : float
        Expected rainfall this month (used as drought proxy).
    optimal_water_level : float
        Optimal water level for the active crop (0.0 if fallow).

    Returns
    -------
    float
        Current annual interest rate (always >= 0).
    """
    # Liquidity premium
    if month in (6, 7):
        delta_liquidity = 0.03  # planting season demand
    elif month in (10, 11):
        delta_liquidity = -0.02  # harvest season surplus
    else:
        delta_liquidity = 0.0

    # Risk premium (drought)
    w_deficit = max(0.0, optimal_water_level - rainfall)
    delta_risk = 0.05 if w_deficit > 0.3 else 0.0

    return max(0.0, base_rate + delta_liquidity + delta_risk)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. Market Price Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_market_seasonal_multiplier(month: int, config: EnvConfig) -> float:
    """Return the seasonal price multiplier for the given month."""
    season = get_season(month)
    for cfg_season, multiplier in config.market_seasonal_multipliers:
        if cfg_season == season:
            return multiplier
    return 1.0


def generate_market_prices(
    month: int,
    config: EnvConfig,
    rng: np.random.Generator,
    prev_prices: Optional[Tuple[float, float, float]] = None,
    effective_base_prices: Optional[Tuple[float, ...]] = None,
) -> Tuple[float, float, float]:
    """
    Generate market prices for each crop type.

    Supports two modes:

    (a) Independent monthly draw (when autocorrelation disabled or no prev_prices):
        P_t(i) = base_i × seasonal(m) × (1 + ε_i)

    (b) Mean-reverting random walk (when autocorrelation enabled + prev_prices):
        target_i = base_i × seasonal(m)
        drift_i  = reversion_speed × (target_i - P_{t-1,i}) / target_i
        P_t(i)   = P_{t-1,i} × (1 + drift_i + ε_i)

    All prices are clamped to [base × price_min_multiplier, base × price_max_multiplier].

    Parameters
    ----------
    month : int
        Calendar month 1-12.
    config : EnvConfig
        Environment configuration.
    rng : np.random.Generator
        Seeded random generator.
    prev_prices : tuple of 3 floats, optional
        Previous month's prices (used for autocorrelation mode).
    effective_base_prices : tuple of floats, optional
        Inflated base prices. If None, uses config.base_market_prices.
    """
    seasonal_mult = _get_market_seasonal_multiplier(month, config)
    base_prices = effective_base_prices or config.base_market_prices
    prices = []

    use_rw = (
        config.enable_price_autocorrelation
        and prev_prices is not None
    )

    for i in range(1, 4):  # crops 1, 2, 3
        base = base_prices[i]
        target = base * seasonal_mult
        noise = rng.normal(0.0, config.market_price_sigma)
        # Clamp noise to ±3σ
        noise = float(np.clip(noise, -3 * config.market_price_sigma,
                              3 * config.market_price_sigma))

        if use_rw:
            prev = prev_prices[i - 1]
            drift = config.price_reversion_speed * (target - prev) / max(target, 1.0)
            price = prev * (1.0 + drift + noise)
        else:
            price = target * (1.0 + noise)

        # Clamp: floor at base × min_multiplier, ceiling at base × max_multiplier
        floor = base * config.price_min_multiplier
        ceiling = base * config.price_max_multiplier
        price = float(np.clip(price, floor, ceiling))
        prices.append(price)

    # Demand shock: rare event affecting one random crop
    if config.demand_shock_probability > 0 and rng.random() < config.demand_shock_probability:
        crop_idx = rng.integers(0, 3)
        direction = rng.choice([-1, 1])
        lo, hi = config.demand_shock_magnitude
        magnitude = rng.uniform(lo, hi)
        shock_mult = 1.0 + direction * magnitude
        base = base_prices[crop_idx + 1]
        floor = base * config.price_min_multiplier
        ceiling = base * config.price_max_multiplier
        prices[crop_idx] = float(np.clip(prices[crop_idx] * shock_mult, floor, ceiling))

    return (prices[0], prices[1], prices[2])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. Yield Calculation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _maturity_factor(crop_age: int, growth_months: int) -> float:
    """
    Maturity sub-factor for yield.

    - Age 0: always 0 (just planted).
    - Growing: quadratic ramp (crop_age / growth_months)².
    - Peak: 1.0 at exactly growth_months.
    - Rotting: drops 0.5 per month past peak → reaches 0.0 in 2 months.
    """
    if crop_age == 0:
        return 0.0
    if crop_age < growth_months:
        return (crop_age / growth_months) ** 2
    months_over = crop_age - growth_months
    return max(0.0, 1.0 - 0.5 * months_over)


def _nitrogen_factor(soil_nitrogen: float, min_requirement: float) -> float:
    """
    Nitrogen sub-factor for yield (piecewise smooth saturation).

    Below minimum requirement:
        Linear ramp from 0 → 0.3 as nitrogen goes 0 → min_req.
    Above minimum requirement:
        Quadratic saturation from 0.3 → 1.0 as nitrogen goes min_req → 1.0.

    Real-world analogy: Liebig's law of the minimum — below a threshold,
    growth is severely limited. Above it, returns diminish.
    """
    if soil_nitrogen <= 0:
        return 0.0
    if min_requirement <= 0:
        return 1.0
    if soil_nitrogen < min_requirement:
        return 0.3 * (soil_nitrogen / min_requirement)
    # Above minimum: quadratic saturation toward 1.0
    ratio = (soil_nitrogen - min_requirement) / (1.0 - min_requirement)
    ratio = min(ratio, 1.0)  # guard against division issues
    return 0.3 + 0.7 * (1.0 - (1.0 - ratio) ** 2)


def _water_factor(current_water_level: float, optimal_water_level: float) -> float:
    """
    Water sub-factor for yield (square-root model).

    Inspired by FAO crop-water response curves (Doorenbos & Kassam):
    the first unit of water rescues a dying crop (high marginal value),
    while topping up from 75% to 100% gives modest improvement.

    - Full water (≥ optimal): 1.0
    - Zero water: 0.1 (crop barely survives)
    - In between: sqrt(ratio), floored at 0.1
    """
    if optimal_water_level <= 0:
        return 1.0  # fallow, water irrelevant
    if current_water_level >= optimal_water_level:
        return 1.0
    ratio = max(0.0, current_water_level / optimal_water_level)
    return max(0.1, math.sqrt(ratio))


def _season_factor(
    month: int,
    crop_type: int,
    config: EnvConfig,
) -> float:
    """
    Seasonal sub-factor for yield.

    Returns 1.0 if the current season is optimal for this crop,
    otherwise returns the configured penalty multiplier (default 0.4).
    """
    if crop_type == 0:
        return 1.0
    season = get_season(month)
    optimal_seasons = config.optimal_seasons_per_crop[crop_type]
    if season in optimal_seasons:
        return 1.0
    return config.non_optimal_season_multiplier


def calculate_yield(
    crop_type: int,
    crop_age: int,
    soil_nitrogen: float,
    current_water_level: float,
    current_month: int,
    config: EnvConfig,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Calculate harvest yield in tons.

    yield = base_yield × maturity × nitrogen × water × season × (1 + ε)

    Parameters
    ----------
    crop_type : int
        1, 2, or 3. Returns 0.0 if 0 (fallow).
    crop_age : int
        Months since planting.
    soil_nitrogen : float
        Current soil nitrogen level (0-1).
    current_water_level : float
        Current water level in the field (0-1).
    current_month : int
        Calendar month 1-12 (for seasonal factor).
    config : EnvConfig
        Environment configuration.
    rng : np.random.Generator, optional
        If provided, adds Gaussian noise to yield. If None, yield is
        deterministic (used for expected_yield_potential).

    Returns
    -------
    float
        Tons of crop produced (>= 0).
    """
    if crop_type == 0:
        return 0.0

    base = config.base_yield_tons[crop_type]
    maturity = _maturity_factor(crop_age, config.growth_months[crop_type])
    nitrogen = _nitrogen_factor(soil_nitrogen, config.minimum_nitrogen_requirement[crop_type])
    water = _water_factor(current_water_level, config.optimal_water_level[crop_type])
    season = _season_factor(current_month, crop_type, config)

    deterministic_yield = base * maturity * nitrogen * water * season

    # Yield noise (stochastic harvest outcomes)
    if rng is not None and config.yield_sigma > 0:
        noise = rng.normal(0.0, config.yield_sigma)
        noise = float(np.clip(noise, -3 * config.yield_sigma, 3 * config.yield_sigma))
        deterministic_yield *= (1.0 + noise)

    return max(0.0, deterministic_yield)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. Expected Yield Potential
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculate_expected_yield_potential(
    crop_type: int,
    crop_age: int,
    soil_nitrogen: float,
    current_water_level: float,
    current_month: int,
    config: EnvConfig,
) -> float:
    """
    Estimate the normalized yield potential if harvested this step.

    potential = raw_yield / max_possible_yield, clipped to [0.0, 1.0]
    max_possible_yield = base_yield × 1.0 (all sub-factors at maximum)

    Uses deterministic yield (no noise) as a planning aide for the agent.
    """
    if crop_type == 0:
        return 0.0

    raw_yield = calculate_yield(
        crop_type=crop_type,
        crop_age=crop_age,
        soil_nitrogen=soil_nitrogen,
        current_water_level=current_water_level,
        current_month=current_month,
        config=config,
        rng=None,  # deterministic
    )

    max_possible = config.base_yield_tons[crop_type]
    if max_possible <= 0:
        return 0.0

    return float(np.clip(raw_yield / max_possible, 0.0, 1.0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# F. Spoilage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def apply_spoilage(
    stored_age: int, stored_amount: float, max_age: int
) -> Tuple[float, bool]:
    """
    Check whether stored crop has spoiled.

    Returns
    -------
    (remaining_amount, spoiled)
        remaining_amount: 0.0 if spoiled, else stored_amount
        spoiled: True if crop rotted this step
    """
    if stored_amount > 0 and stored_age > max_age:
        return 0.0, True
    return stored_amount, False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G. Text Observation Formatter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def format_text_observation(
    obs_dict: dict,
    config: EnvConfig,
    has_active_loan: bool,
    valid_actions: list[int] | None = None,
) -> str:
    """
    Convert observation data into a human-readable text block for LLM agents.
    """
    month = obs_dict["current_month"]
    step = obs_dict["current_step"]
    max_steps = config.max_steps
    season = get_season(month)
    month_name = MONTH_NAMES[month]

    crop_type = obs_dict["active_crop_type"]
    crop_name = config.crop_names[crop_type]
    crop_cat = config.crop_categories[crop_type]
    crop_age = obs_dict["crop_age_months"]
    growth_req = config.growth_months[crop_type] if crop_type > 0 else 0

    lines = [
        f"============= Farm Dashboard (Step {step}/{max_steps}) =============",
        f"Month: {month_name} ({month}) | Season: {season.value}",
        f"Weather: Expected rainfall {obs_dict['expected_rainfall']:.2f}/1.0",
        "",
        "FARM STATUS:",
    ]

    if crop_type == 0:
        lines.append("Active Crop: None (Fallow land)")
    else:
        lines.append(
            f"Active Crop: {crop_name} ({crop_cat}) | "
            f"Age: {crop_age}/{growth_req} months"
        )
    lines.append(f"Soil Nitrogen: {obs_dict['soil_nitrogen']:.2f}/1.0")
    lines.append(f"Water Level: {obs_dict['current_water_level']:.2f}/1.0")
    lines.append(
        f"Expected Yield Potential: {obs_dict['expected_yield_potential']:.2f}/1.0"
    )

    lines.append("")
    lines.append("FINANCES:")
    loan_status = " (active loan)" if has_active_loan else ""
    lines.append(
        f"Cash: ₹{obs_dict['cash_balance']:,.0f} | "
        f"Debt: ₹{obs_dict['current_debt']:,.0f}{loan_status}"
    )
    lines.append(
        f"Interest Rate: {obs_dict['current_interest_rate'] * 100:.1f}% annual"
    )
    lines.append(
        f"Land Value: ₹{obs_dict['current_land_price']:,.0f}"
    )

    lines.append("")
    lines.append("MARKET PRICES (per ton):")
    lines.append(
        f"Corn: ₹{obs_dict['market_price_crop_1']:,.0f} | "
        f"Wheat: ₹{obs_dict['market_price_crop_2']:,.0f} | "
        f"Chickpea: ₹{obs_dict['market_price_crop_3']:,.0f}"
    )

    lines.append("")
    lines.append("STORAGE:")
    stored_type = obs_dict["stored_crop_type"]
    stored_amt = obs_dict["stored_amount"]
    if stored_type == 0 or stored_amt <= 0:
        lines.append("Empty")
    else:
        stored_name = config.crop_names[stored_type]
        stored_age = obs_dict["stored_age_months"]
        lines.append(
            f"{stored_amt:.1f} tons of {stored_name} "
            f"(age: {stored_age}/{config.max_storage_age} months)"
        )

    lines.append("")
    lines.append("COSTS:")
    lines.append(
        f"Plant Corn: ₹{obs_dict['cost_seed_1']:,.0f} | "
        f"Plant Wheat: ₹{obs_dict['cost_seed_2']:,.0f} | "
        f"Plant Chickpea: ₹{obs_dict['cost_seed_3']:,.0f}"
    )
    lines.append(
        f"Irrigate: ₹{obs_dict['cost_irrigate']:,.0f} | "
        f"Fertilize: ₹{obs_dict['cost_fertilize']:,.0f} | "
        f"Monthly Fixed: ₹{obs_dict.get('monthly_fixed_cost', 0):,.0f}"
    )

    if valid_actions is not None:
        lines.append("")
        lines.append("AVAILABLE ACTIONS:")
        action_strs = [
            f"{a}: {config.action_names[a]}" for a in valid_actions
        ]
        lines.append(" | ".join(action_strs))

    return "\n".join(lines)
