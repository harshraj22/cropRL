"""
CropRL Dynamics Engine.

All simulation physics live here, separate from the step() orchestration.
Each function is pure (given inputs → deterministic output for a given rng state),
making them independently unit-testable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import EnvConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. Weather Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_seasonal_baseline(month: int, config: EnvConfig) -> float:
    """Return the seasonal rainfall baseline μ(m) for the given month."""
    for months_tuple, baseline in config.weather_seasonal_baselines:
        if month in months_tuple:
            return baseline
    # Fallback (should not happen with well-configured baselines)
    return 0.5


def generate_rainfall(
    month: int, config: EnvConfig, rng: np.random.Generator
) -> float:
    """
    Generate seasonal rainfall with Gaussian noise.

    W_t = clip(μ(m) + ε, 0, 1)  where ε ~ N(0, σ²)

    Parameters
    ----------
    month : int
        Calendar month 1-12.
    config : EnvConfig
        Environment configuration (provides sigma and seasonal baselines).
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    float
        Rainfall value in [0.0, 1.0].
    """
    baseline = _get_seasonal_baseline(month, config)
    noise = rng.normal(0.0, config.weather_sigma)
    return float(np.clip(baseline + noise, 0.0, 1.0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. Dynamic Interest Rate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculate_interest_rate(
    base_rate: float,
    month: int,
    rainfall: float,
    optimal_rainfall: float,
) -> float:
    """
    Calculate the current annual interest rate.

    R_interest = R_base + Δ_liquidity(m) + Δ_risk(W_deficit)

    Parameters
    ----------
    base_rate : float
        Base annual interest rate (e.g. 0.08).
    month : int
        Calendar month 1-12.
    rainfall : float
        Actual rainfall this month.
    optimal_rainfall : float
        Optimal rainfall for the active crop (0.0 if fallow).

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
    w_deficit = max(0.0, optimal_rainfall - rainfall)
    delta_risk = 0.05 if w_deficit > 0.3 else 0.0

    return max(0.0, base_rate + delta_liquidity + delta_risk)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. Market Price Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_market_seasonal_multiplier(month: int, config: EnvConfig) -> float:
    """Return the seasonal price multiplier for the given month."""
    for months_tuple, multiplier in config.market_seasonal_multipliers:
        if month in months_tuple:
            return multiplier
    return 1.0


def generate_market_prices(
    month: int,
    config: EnvConfig,
    rng: np.random.Generator,
    prev_prices: Optional[Tuple[float, float, float]] = None,
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

    Demand shocks: with probability `demand_shock_probability`, one random
    crop gets a price spike or dip of 30-60%.

    All prices are clamped to [1.0, base_i × price_max_multiplier].

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

    Returns
    -------
    tuple of 3 floats
        (price_crop1, price_crop2, price_crop3)
    """
    seasonal_mult = _get_market_seasonal_multiplier(month, config)
    prices = []

    use_rw = (
        config.enable_price_autocorrelation
        and prev_prices is not None
    )

    for i in range(1, 4):  # crops 1, 2, 3
        base = config.base_market_prices[i]
        target = base * seasonal_mult
        noise = rng.normal(0.0, config.market_price_sigma)
        # Clamp noise to ±3σ
        noise = float(np.clip(noise, -3 * config.market_price_sigma,
                              3 * config.market_price_sigma))

        if use_rw:
            # Mean-reverting random walk
            prev = prev_prices[i - 1]
            drift = config.price_reversion_speed * (target - prev) / max(target, 1.0)
            price = prev * (1.0 + drift + noise)
        else:
            # Independent draw
            price = target * (1.0 + noise)

        # Clamp: floor at 1.0, ceiling at base × max_multiplier
        ceiling = base * config.price_max_multiplier
        price = float(np.clip(price, 1.0, ceiling))
        prices.append(price)

    # Demand shock: rare event affecting one random crop
    if config.demand_shock_probability > 0 and rng.random() < config.demand_shock_probability:
        crop_idx = rng.integers(0, 3)  # which crop (0-indexed into prices list)
        direction = rng.choice([-1, 1])
        lo, hi = config.demand_shock_magnitude
        magnitude = rng.uniform(lo, hi)
        shock_mult = 1.0 + direction * magnitude
        base = config.base_market_prices[crop_idx + 1]
        ceiling = base * config.price_max_multiplier
        prices[crop_idx] = float(np.clip(prices[crop_idx] * shock_mult, 1.0, ceiling))

    return (prices[0], prices[1], prices[2])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. Yield Calculation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculate_yield(
    crop_type: int,
    crop_age: int,
    soil_nitrogen: float,
    rainfall: float,
    irrigated: bool,
    config: EnvConfig,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Calculate harvest yield in tons.

    yield = base_yield × nitrogen_factor × water_factor × maturity_factor × (1 + ε)

    nitrogen_factor = clip(soil_nitrogen × 2, 0.3, 1.5)
    water_factor    = max(0.2, 1.0 − deficit × 2)
                      where deficit adjusted by irrigation
    maturity_factor = 1.0 if mature, 0.3 if early harvest
    ε ~ N(0, yield_sigma²)  -- harvest noise (if rng provided)

    Parameters
    ----------
    crop_type : int
        1, 2, or 3. Returns 0.0 if 0 (fallow).
    crop_age : int
        Months since planting.
    soil_nitrogen : float
        Current soil nitrogen level (0-1).
    rainfall : float
        Actual rainfall this month.
    irrigated : bool
        Whether the agent irrigated this month.
    config : EnvConfig
        Environment configuration.
    rng : np.random.Generator, optional
        If provided, adds Gaussian noise to yield. If None, yield is
        deterministic (backward compatible for tests/expected_yield).

    Returns
    -------
    float
        Tons of crop produced (>= 0).
    """
    if crop_type == 0:
        return 0.0

    base = config.base_yield_tons[crop_type]

    # Nitrogen factor
    nitrogen_factor = float(np.clip(soil_nitrogen * 2.0, 0.3, 1.5))

    # Water factor
    optimal = config.optimal_rainfall[crop_type]
    deficit = max(0.0, optimal - rainfall)
    if irrigated:
        deficit *= 0.3  # irrigation mitigates 70% of drought
    water_factor = max(0.2, 1.0 - deficit * 2.0)

    # Maturity factor: peaks at exactly required_months, lower before/after.
    required_months = config.growth_months[crop_type]
    if crop_age == 0:
        maturity_factor = 0.0
    elif crop_age < required_months:
        maturity_factor = (crop_age / required_months) ** 2
    else:
        months_over = crop_age - required_months
        maturity_factor = max(0.0, 1.0 - 0.2 * months_over)

    deterministic_yield = base * nitrogen_factor * water_factor * maturity_factor

    # Yield noise (stochastic harvest outcomes)
    if rng is not None and config.yield_sigma > 0:
        noise = rng.normal(0.0, config.yield_sigma)
        # Clamp noise to ±3σ to prevent extreme outliers
        noise = float(np.clip(noise, -3 * config.yield_sigma, 3 * config.yield_sigma))
        deterministic_yield *= (1.0 + noise)

    # Floor at 0 — can't have negative yield
    return max(0.0, deterministic_yield)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. Expected Yield Potential
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calculate_expected_yield_potential(
    crop_type: int,
    crop_age: int,
    soil_nitrogen: float,
    expected_rainfall: float,
    config: EnvConfig,
) -> float:
    """
    Estimate the normalized yield potential if harvested this step.

    This is a derived, read-only field in the observation. It gives the agent
    information to decide whether to harvest now or wait for maturity.

    potential = raw_yield / max_possible_yield, clipped to [0.0, 1.0]
    max_possible_yield = base_yield × 1.5 (max nitrogen factor)

    Parameters
    ----------
    crop_type : int
        Current crop type (0 = fallow → returns 0.0).
    crop_age : int
        Months since planting.
    soil_nitrogen : float
        Current soil nitrogen level.
    expected_rainfall : float
        Expected rainfall for this month.
    config : EnvConfig
        Environment configuration.

    Returns
    -------
    float
        Normalized yield potential in [0.0, 1.0].
    """
    if crop_type == 0:
        return 0.0

    raw_yield = calculate_yield(
        crop_type=crop_type,
        crop_age=crop_age,
        soil_nitrogen=soil_nitrogen,
        rainfall=expected_rainfall,
        irrigated=False,  # conservative estimate without irrigation
        config=config,
    )

    max_possible = config.base_yield_tons[crop_type] * 1.5  # max nitrogen factor
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

    Parameters
    ----------
    stored_age : int
        Months since the crop was stored.
    stored_amount : float
        Current amount in storage.
    max_age : int
        Maximum storage age before rot.

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

# Season names for display
_SEASON_MAP = {
    1: "Winter",
    2: "Spring",
    3: "Spring",
    4: "Summer",
    5: "Summer",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Winter",
    11: "Winter",
    12: "Winter",
}

_MONTH_NAMES = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def format_text_observation(
    obs_dict: dict,
    config: EnvConfig,
    has_active_loan: bool,
    valid_actions: list[int] | None = None,
) -> str:
    """
    Convert observation data into a human-readable text block for LLM agents.

    Parameters
    ----------
    obs_dict : dict
        Dictionary of observation field values.
    config : EnvConfig
        Environment config (for action names, crop names).
    has_active_loan : bool
        Whether the agent currently has an active loan.
    valid_actions : list of int, optional
        List of currently valid action IDs.

    Returns
    -------
    str
        Multi-line text summary.
    """
    month = obs_dict["current_month"]
    step = obs_dict["current_step"]
    max_steps = config.max_steps
    season = _SEASON_MAP.get(month, "Unknown")
    month_name = _MONTH_NAMES[month]

    crop_type = obs_dict["active_crop_type"]
    crop_name = config.crop_names[crop_type]
    crop_cat = config.crop_categories[crop_type]
    crop_age = obs_dict["crop_age_months"]
    growth_req = config.growth_months[crop_type] if crop_type > 0 else 0

    lines = [
        f"=== Farm Dashboard (Step {step}/{max_steps}) ===",
        f"Month: {month_name} ({month}) | Season: {season}",
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
    lines.append(
        f"Soil Nitrogen: {obs_dict['soil_nitrogen']:.2f}/1.0"
    )
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
        f"Fertilize: ₹{obs_dict['cost_fertilize']:,.0f}"
    )

    if valid_actions is not None:
        lines.append("")
        lines.append("AVAILABLE ACTIONS:")
        action_strs = [
            f"{a}: {config.action_names[a]}" for a in valid_actions
        ]
        lines.append(" | ".join(action_strs))

    return "\n".join(lines)
