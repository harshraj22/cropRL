"""
CropRL Environment Configuration.

Centralized dataclass holding every tunable parameter.
Task difficulty is controlled by overriding specific fields.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .enums import ActionType, CropType, Season


@dataclass
class EnvConfig:
    """
    Configuration for the CropRL farm management environment.

    All numeric constants live here so that:
    - Tasks can override a subset for difficulty tuning
    - Tests can inject custom configs for reproducibility
    """

    # ── Episode ─────────────────────────────────────────────────
    max_steps: int = 300         # trajectory length limit (actions)
    max_months: int = 60         # calendar month horizon (5 years)

    # ── Crop Definitions ────────────────────────────────────────
    # Index 0 = Fallow (no crop)
    # Index 1 = Heavy Feeder (Corn)   — high cost, high profit, destroys N
    # Index 2 = Medium Feeder (Wheat) — moderate cost/profit, mild N drain
    # Index 3 = Legume (Chickpea)     — low cost, lower profit, restores N
    # Index 4 = Matcha  (Hype crop)   — niche, 5-month grow, 3.5× hype premium
    # Index 5 = Quinoa  (Hype crop)   — 3-month grow, 2.8× hype premium
    # Index 6 = Turmeric (Hype crop)  — 4-month grow, 2.2× hype premium
    num_crop_types: int = 7  # including fallow + 3 hype crops

    seed_costs: Tuple[float, ...] = (0.0, 800.0, 500.0, 200.0, 1500.0, 900.0, 600.0)
    growth_months: Tuple[int, ...] = (0, 4, 3, 3, 5, 3, 4)
    base_yield_tons: Tuple[float, ...] = (0.0, 8.0, 5.0, 3.0, 2.0, 3.0, 4.0)
    yield_sigma: float = 0.10  # Gaussian noise on harvest yield

    crop_names: Tuple[str, ...] = (
        "Fallow", "Corn", "Wheat", "Chickpea",
        "Matcha", "Quinoa", "Turmeric",
    )
    crop_categories: Tuple[str, ...] = (
        "None",
        "Heavy Feeder",
        "Medium Feeder",
        "Legume",
        "Hype Crop",
        "Hype Crop",
        "Hype Crop",
    )

    # ── Water System ────────────────────────────────────────────
    # Optimal water level (drainage cap) per crop.
    # Fallow = 1.0 (field can hold max water when empty).
    optimal_water_level: Tuple[float, ...] = (1.0, 0.6, 0.4, 0.3, 0.5, 0.35, 0.4)

    # Monthly water consumption per crop.
    water_utilised_monthly: Tuple[float, ...] = (0.0, 0.15, 0.10, 0.08, 0.12, 0.09, 0.10)

    # Instant water added to the field on irrigate, per crop.
    irrigate_amount: Tuple[float, ...] = (0.0, 0.30, 0.20, 0.15, 0.25, 0.18, 0.20)

    # ── Costs ───────────────────────────────────────────────────
    cost_irrigate: float = 300.0
    cost_fertilize: float = 400.0
    monthly_fixed_cost: float = 200.0    # overhead the agent bears each month
    fertilize_nitrogen_boost: float = 0.15

    # [Future Scope]
    cost_storage_monthly: float = 0.0
    cost_machinery_repair: float = 0.0

    # ── Soil ────────────────────────────────────────────────────
    # Monthly nitrogen impact while a crop is growing (applied each month tick).
    # Corn depletes fast, Wheat moderate, Chickpea restores.
    #
    # Derivation (see implementation plan):
    #   Corn:     N=1.0 → ~0 in 14 months ⇒ net drain 0.07/mo ⇒ raw = −0.08
    #   Wheat:    N=1.0 → ~0 in 16 months ⇒ net drain 0.06/mo ⇒ raw = −0.07
    #   Chickpea: restores ~0.5 in 9 months                    ⇒ raw = +0.05
    monthly_nitrogen_impact: Tuple[float, ...] = (0.0, -0.08, -0.07, 0.05, -0.12, -0.06, -0.04)

    natural_nitrogen_recovery: float = 0.01  # passive recovery per month

    # Minimum soil nitrogen for a crop to produce reasonable yield.
    minimum_nitrogen_requirement: Tuple[float, ...] = (0.0, 0.3, 0.2, 0.1, 0.25, 0.15, 0.15)

    # ── Seasonal Yield ──────────────────────────────────────────
    # Per-crop tuple of optimal seasons.
    optimal_seasons_per_crop: Tuple[Tuple[Season, ...], ...] = (
        (),                                       # Fallow
        (Season.MONSOON,),                        # Corn — Kharif
        (Season.WINTER,),                         # Wheat — Rabi
        (Season.WINTER, Season.SPRING),           # Chickpea — Rabi/Spring
        (Season.MONSOON, Season.SUMMER),          # Matcha — warm & wet
        (Season.SPRING, Season.SUMMER),           # Quinoa — mild climate
        (Season.MONSOON,),                        # Turmeric — tropical
    )

    # Yield multiplier when growing in a non-optimal season.
    non_optimal_season_multiplier: float = 0.4

    # ── Financial ───────────────────────────────────────────────
    initial_cash: float = 10000.0
    initial_soil_nitrogen: float = 0.6
    loan_chunk: float = 5000.0
    base_interest_rate: float = 0.08  # 8% annual, applied monthly as rate/12

    # Land value: current_land_price = base_land_price × soil_nitrogen
    base_land_price: float = 15000.0

    # Annual inflation rate (compounding, applied at the start of each year).
    inflation_rate: float = 0.03

    # ── Spoilage ────────────────────────────────────────────────
    max_storage_age: int = 6  # months before rot

    # ── Reward ──────────────────────────────────────────────────
    invalid_action_penalty: float = -50.0
    bankruptcy_penalty: float = -1000.0

    # ── Weather ─────────────────────────────────────────────────
    weather_sigma: float = 0.15  # noise on seasonal forecast
    weather_sigma_realisation: float = 0.05  # noise on actual vs expected

    # Seasonal rainfall baselines: Season → mean μ
    weather_seasonal_baselines: Tuple[Tuple[Season, float], ...] = (
        (Season.MONSOON, 0.80),
        (Season.WINTER, 0.20),
        (Season.SPRING, 0.40),
        (Season.SUMMER, 0.05),
    )

    # ── Market Prices ───────────────────────────────────────────
    base_market_prices: Tuple[float, ...] = (0.0, 1200.0, 800.0, 500.0, 2500.0, 1800.0, 1200.0)
    market_price_sigma: float = 0.10

    # Seasonal price multipliers: Season → multiplier
    market_seasonal_multipliers: Tuple[Tuple[Season, float], ...] = (
        (Season.WINTER, 0.95),       # Oct–Jan: post-harvest oversupply
        (Season.SPRING, 0.95),       # Feb–Mar
        (Season.SUMMER, 1.15),       # Apr–May: pre-monsoon scarcity
        (Season.MONSOON, 1.00),      # Jun–Sep
    )
    price_max_multiplier: float = 2.5   # ceiling = base × this
    price_min_multiplier: float = 0.5   # floor   = base × this

    # ── Price Autocorrelation (Random Walk) ─────────────────────
    enable_price_autocorrelation: bool = True
    price_reversion_speed: float = 0.3

    # ── Demand Shocks ───────────────────────────────────────────
    demand_shock_probability: float = 0.08
    demand_shock_magnitude: Tuple[float, float] = (0.3, 0.6)

    # ── Observation Format ──────────────────────────────────────
    text_mode: bool = False

    # ── Future Scope Flags ──────────────────────────────────────
    enable_machinery: bool = False
    enable_storage_cost: bool = False

    # ── Action Space ────────────────────────────────────────────
    num_actions: int = 15

    action_names: Tuple[str, ...] = (
        "End Turn / Wait",
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
        "Post Forum Message",
        "Plant Matcha (Hype Crop)",
        "Plant Quinoa (Hype Crop)",
        "Plant Turmeric (Hype Crop)",
    )


@dataclass
class MultiAgentConfig:
    """
    Configuration for the multi-agent extension of CropRL.

    Controls the number of agents, slot-based turn system, market
    supply-demand parameters, forum limits, and hype crop settings.
    """

    # ── Agent Setup ─────────────────────────────────────────────
    num_agents: int = 4
    objective_mode: str = "competitive"  # competitive | cooperative | mixed

    # ── Slot-Based Turn System ──────────────────────────────────
    action_slots_per_month: int = 4     # K — max actions per agent per month

    # ── Forum ───────────────────────────────────────────────────
    forum_messages_per_month: int = 2   # max public posts per agent per month

    # ── Market Supply-Demand ────────────────────────────────────
    # Market absorption capacity (tons) per crop at full price.
    # Uses CropType integer IDs as keys.
    market_capacity: Dict[int, float] = field(default_factory=lambda: {
        1: 50.0,   # Corn
        2: 40.0,   # Wheat
        3: 20.0,   # Chickpea
        4: 5.0,    # Matcha  — niche, saturates fast
        5: 10.0,   # Quinoa
        6: 15.0,   # Turmeric
    })
    price_impact_coeff: float = 1.0   # steepness of glut-induced price drop
    price_floor_mult:   float = 0.3   # price floor as fraction of base price

    # ── Hype Crops ──────────────────────────────────────────────
    enable_hype_crops: bool = True
    hype_trigger_prob: float = 0.05   # prob per month that a hype crop enters BUILDING
    hype_collapse_supply_threshold: float = 0.6  # fraction of market_capacity triggering early collapse
    # Maximum hype price premiums per crop (multiplier at hype_level=1.0)
    max_hype_premium: Dict[int, float] = field(default_factory=lambda: {
        4: 3.5,   # Matcha
        5: 2.8,   # Quinoa
        6: 2.2,   # Turmeric
    })

    # ── Mixed-Mode Bonus ────────────────────────────────────────
    mixed_mode_village_weight: float = 0.3  # weight of village avg in mixed mode
