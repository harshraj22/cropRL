"""
CropRL Environment Configuration.

Centralized dataclass holding every tunable parameter.
Task difficulty is controlled by overriding specific fields.
Future-scope features are gated behind enable_* flags.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    """
    Configuration for the CropRL farm management environment.

    All numeric constants live here so that:
    - Tasks can override a subset for difficulty tuning
    - Future-scope features can be toggled without code changes
    - Tests can inject custom configs for reproducibility
    """

    # ── Episode ─────────────────────────────────────────────────
    max_steps: int = 60  # 5 years of monthly steps (future: 120, 240)

    # ── Crop Definitions ────────────────────────────────────────
    # Index 0 = Fallow (no crop)
    # Index 1 = Heavy Feeder (Corn)   — high cost, high profit, destroys N
    # Index 2 = Medium Feeder (Wheat) — moderate cost/profit, mild N drain
    # Index 3 = Legume (Chickpea)     — low cost, lower profit, restores N
    num_crop_types: int = 4  # including fallow

    seed_costs: Tuple[float, ...] = (0.0, 800.0, 500.0, 200.0)
    growth_months: Tuple[int, ...] = (0, 4, 3, 3)
    optimal_rainfall: Tuple[float, ...] = (0.0, 0.6, 0.4, 0.3)
    nitrogen_impact: Tuple[float, ...] = (0.0, -0.25, -0.10, 0.15)
    base_yield_tons: Tuple[float, ...] = (0.0, 8.0, 5.0, 3.0)
    yield_sigma: float = 0.10  # Gaussian noise on harvest yield

    crop_names: Tuple[str, ...] = ("Fallow", "Corn", "Wheat", "Chickpea")
    crop_categories: Tuple[str, ...] = (
        "None",
        "Heavy Feeder",
        "Medium Feeder",
        "Legume",
    )

    # ── Costs (constant for MVP, future: may vary monthly) ─────
    cost_irrigate: float = 300.0
    cost_fertilize: float = 400.0
    fertilize_nitrogen_boost: float = 0.15  # how much N is added per fertilize

    # [Future Scope]
    cost_storage_monthly: float = 0.0
    cost_machinery_repair: float = 0.0

    # ── Financial ───────────────────────────────────────────────
    initial_cash: float = 10000.0
    initial_soil_nitrogen: float = 0.6
    loan_chunk: float = 5000.0
    max_debt: float = 20000.0  # ceiling, though 1-loan rule limits to loan_chunk
    base_interest_rate: float = 0.08  # 8% annual, applied monthly as rate/12

    # ── Spoilage ────────────────────────────────────────────────
    max_storage_age: int = 6  # months before rot

    # ── Soil ────────────────────────────────────────────────────
    natural_nitrogen_recovery: float = 0.01  # per month, passive recovery

    # ── Reward ──────────────────────────────────────────────────
    invalid_action_penalty: float = -50.0
    bankruptcy_penalty: float = -1000.0
    terminal_soil_bonus_factor: float = 10000.0

    # ── Weather ─────────────────────────────────────────────────
    weather_sigma: float = 0.15

    # Seasonal baselines: {months} → μ
    # Monsoon  {6,7,8,9}:    0.8
    # Winter   {10,11,12,1}: 0.2
    # Spring   {2,3}:        0.4
    # Summer   {4,5}:        0.05
    weather_seasonal_baselines: Tuple[Tuple[Tuple[int, ...], float], ...] = (
        ((6, 7, 8, 9), 0.8),
        ((10, 11, 12, 1), 0.2),
        ((2, 3), 0.4),
        ((4, 5), 0.05),
    )

    # ── Market Prices ───────────────────────────────────────────
    base_market_prices: Tuple[float, ...] = (0.0, 1200.0, 800.0, 500.0)
    market_price_sigma: float = 0.10

    # Seasonal multipliers for prices: {months} → multiplier
    # Post-harvest {10,11,12}: 0.85
    # Pre-monsoon  {4,5}:      1.15
    # Monsoon      {6,7,8,9}:  1.0
    # Winter       {1,2,3}:    0.95
    market_seasonal_multipliers: Tuple[Tuple[Tuple[int, ...], float], ...] = (
        ((10, 11, 12), 0.85),
        ((4, 5), 1.15),
        ((6, 7, 8, 9), 1.0),
        ((1, 2, 3), 0.95),
    )
    price_max_multiplier: float = 2.5  # ceiling clamp: price <= base × this

    # ── Price Autocorrelation (Random Walk) ─────────────────────
    enable_price_autocorrelation: bool = True
    price_reversion_speed: float = 0.3  # mean-reversion toward seasonal base

    # ── Demand Shocks ───────────────────────────────────────────
    demand_shock_probability: float = 0.08  # ~1 per year
    demand_shock_magnitude: Tuple[float, float] = (0.3, 0.6)  # uniform range

    # ── Observation Format ──────────────────────────────────────
    text_mode: bool = False  # True = text summary for LLMs; False = numeric for RL

    # ── Future Scope Flags ──────────────────────────────────────
    enable_machinery: bool = False
    enable_storage_cost: bool = False

    # ── Action Space ────────────────────────────────────────────
    num_actions: int = 11  # actions 0..10

    # Human-readable action names (for text_mode and logging)
    action_names: Tuple[str, ...] = (
        "Wait",
        "Plant Corn (Heavy Feeder)",
        "Plant Wheat (Medium Feeder)",
        "Plant Chickpea (Legume)",
        "Irrigate",
        "Fertilize",
        "Harvest & Store",
        "Harvest & Sell",
        "Sell Inventory",
        "Take Loan",
        "Repay Loan",
    )
