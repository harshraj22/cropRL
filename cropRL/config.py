"""
CropRL Environment Configuration.

Centralized dataclass holding every tunable parameter.
Task difficulty is controlled by overriding specific fields.
"""

from dataclasses import dataclass
from typing import Tuple

from .enums import ActionType, CropType, Season


@dataclass
class EnvConfig:
    """
    Configuration for the CropRL farm management environment.

    All numeric constants live here so that:
    - Tasks can override a subset for difficulty tuning
    - Tests can inject custom configs for reproducibility
    """

    # ── Multi-Agent ─────────────────────────────────────────────
    num_farmers: int = 2
    steps_per_agent_per_month: int = 5

    # ── Episode ─────────────────────────────────────────────────
    max_months: int = 60

    # ── Crop Definitions ────────────────────────────────────────
    num_crop_types: int = 4
    seed_costs: Tuple[float, ...] = (0.0, 800.0, 500.0, 200.0)
    growth_months: Tuple[int, ...] = (0, 4, 3, 3)
    base_yield_tons: Tuple[float, ...] = (0.0, 8.0, 5.0, 3.0)
    yield_sigma: float = 0.10
    crop_names: Tuple[str, ...] = ("Fallow", "Corn", "Wheat", "Chickpea")
    crop_categories: Tuple[str, ...] = ("None", "Heavy Feeder", "Medium Feeder", "Legume")

    # ── Water System ────────────────────────────────────────────
    optimal_water_level: Tuple[float, ...] = (1.0, 0.6, 0.4, 0.3)
    water_utilised_monthly: Tuple[float, ...] = (0.0, 0.15, 0.10, 0.08)
    irrigate_amount: Tuple[float, ...] = (0.0, 0.30, 0.20, 0.15)

    # ── Costs ───────────────────────────────────────────────────
    cost_irrigate: float = 300.0
    cost_fertilize: float = 400.0
    monthly_fixed_cost: float = 200.0
    fertilize_nitrogen_boost: float = 0.15
    cost_storage_monthly: float = 0.0
    cost_machinery_repair: float = 0.0

    # ── Soil ────────────────────────────────────────────────────
    monthly_nitrogen_impact: Tuple[float, ...] = (0.0, -0.08, -0.07, 0.05)
    natural_nitrogen_recovery: float = 0.01
    minimum_nitrogen_requirement: Tuple[float, ...] = (0.0, 0.3, 0.2, 0.1)

    # ── Seasonal Yield ──────────────────────────────────────────
    optimal_seasons_per_crop: Tuple[Tuple[Season, ...], ...] = (
        (),
        (Season.MONSOON,),
        (Season.WINTER,),
        (Season.WINTER, Season.SPRING),
    )
    non_optimal_season_multiplier: float = 0.4

    # ── Financial ───────────────────────────────────────────────
    initial_cash: float = 10000.0
    initial_soil_nitrogen: float = 0.6
    loan_chunk: float = 5000.0
    base_interest_rate: float = 0.08
    base_land_price: float = 15000.0

    # ── Spoilage ────────────────────────────────────────────────
    max_storage_age: int = 6

    # ── Reward ──────────────────────────────────────────────────
    invalid_action_penalty: float = -50.0
    bankruptcy_penalty: float = -1000.0

    # ── Weather ─────────────────────────────────────────────────
    weather_sigma: float = 0.15
    weather_sigma_realisation: float = 0.05
    weather_seasonal_baselines: Tuple[Tuple[Season, float], ...] = (
        (Season.MONSOON, 0.80),
        (Season.WINTER, 0.20),
        (Season.SPRING, 0.40),
        (Season.SUMMER, 0.05),
    )

    # ── Market Prices ───────────────────────────────────────────
    base_market_prices: Tuple[float, ...] = (0.0, 1200.0, 800.0, 500.0)
    market_price_sigma: float = 0.10
    market_seasonal_multipliers: Tuple[Tuple[Season, float], ...] = (
        (Season.WINTER, 0.95),
        (Season.SPRING, 0.95),
        (Season.SUMMER, 1.15),
        (Season.MONSOON, 1.00),
    )
    price_max_multiplier: float = 2.5
    price_min_multiplier: float = 0.5
    enable_price_autocorrelation: bool = True
    price_reversion_speed: float = 0.3
    demand_shock_probability: float = 0.08
    demand_shock_magnitude: Tuple[float, float] = (0.3, 0.6)

    # ── Supply-Aware Pricing (Multi-Agent) ──────────────────────
    supply_price_alpha: float = 0.15

    # ── Discussion Forum (Multi-Agent) ──────────────────────────
    forum_message_max_chars: int = 200

    # ── Observation Format ──────────────────────────────────────
    text_mode: bool = False

    # ── Future Scope Flags ──────────────────────────────────────
    enable_machinery: bool = False
    enable_storage_cost: bool = False

    # ── Action Space ────────────────────────────────────────────
    num_actions: int = 12
    action_names: Tuple[str, ...] = (
        "No-Op",
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
        "Post to Forum",
    )

    @property
    def max_steps(self) -> int:
        """Total step budget across all agents and months."""
        return self.max_months * self.steps_per_agent_per_month * self.num_farmers
