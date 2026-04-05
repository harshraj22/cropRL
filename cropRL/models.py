"""
Data models for the CropRL Environment.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class CroprlAction(Action):
    """
    Agent selects a discrete action each step.
    """
    action_id: int = Field(
        ge=0,
        le=10,
        description="Discrete action index (0-10)",
        json_schema_extra={
            "enum": [
                ["0 - Wait (do nothing)", 0],
                ["1 - Plant Crop 1 (Corn / Heavy Feeder)", 1],
                ["2 - Plant Crop 2 (Wheat / Medium Feeder)", 2],
                ["3 - Plant Crop 3 (Chickpea / Legume)", 3],
                ["4 - Irrigate", 4],
                ["5 - Fertilize", 5],
                ["6 - Harvest & Store", 6],
                ["7 - Harvest & Sell", 7],
                ["8 - Sell Inventory", 8],
                ["9 - Take Loan", 9],
                ["10 - Repay Loan", 10],
            ]
        }
    )

class CroprlObservation(Observation):
    """
    Full farm dashboard the agent sees each step.
    """
    # ── Time & Weather ──────────────────────────────────────────
    current_month: int = Field(description="Calendar month 1-12")
    current_step: int = Field(description="Step index 0..max_steps-1")
    expected_rainfall: float = Field(description="Forecasted rainfall for this month, 0.0 to 1.0")

    # ── Biological & Soil ───────────────────────────────────────
    active_crop_type: int = Field(description="0=Fallow, 1=Corn, 2=Wheat, 3=Chickpea")
    crop_age_months: int = Field(description="Months since planting")
    expected_yield_potential: float = Field(description="Estimated yield if harvested now, normalized 0.0-1.0")
    soil_nitrogen: float = Field(description="Soil nitrogen level 0.0-1.0")
    current_water_level: float = Field(description="Current water level in the field 0.0-1.0")

    # ── Financial ───────────────────────────────────────────────
    cash_balance: float = Field(description="Current cash on hand")
    current_debt: float = Field(description="Outstanding loan debt")
    current_interest_rate: float = Field(description="Current annual interest rate")
    current_land_price: float = Field(description="Current land value = base_land_price × soil_nitrogen")
    market_price_crop_1: float = Field(description="Spot price for Corn")
    market_price_crop_2: float = Field(description="Spot price for Wheat")
    market_price_crop_3: float = Field(description="Spot price for Chickpea")

    # ── Costs ───────────────────────────────────────────────────
    cost_seed_1: float = Field(description="Cost to plant Corn")
    cost_seed_2: float = Field(description="Cost to plant Wheat")
    cost_seed_3: float = Field(description="Cost to plant Chickpea")
    cost_irrigate: float = Field(description="Cost to irrigate")
    cost_fertilize: float = Field(description="Cost to fertilize")

    # ── Inventory ───────────────────────────────────────────────
    stored_crop_type: int = Field(description="Type of crop in storage (0=empty)")
    stored_amount: float = Field(description="Tons of crop in storage")
    stored_age_months: int = Field(description="Months since stored crop was harvested")

    # ── Info ────────────────────────────────────────────────────
    message: str = Field(default="", description="Human-readable feedback about what happened this step")
    text_summary: str = Field(default="", description="Full text observation")

class CroprlState(State):
    """
    Internal environment state (superset of observation).
    """
    # ── Internal Bookkeeping ────────────────────────────────────
    irrigated_this_month: bool = False
    fertilized_this_month: bool = False
    previous_cash: float = 0.0
    has_active_loan: bool = False
    loan_interest_rate: float = 0.0    # locked rate at loan origination
    current_month_count: int = 0       # total months elapsed
    current_year: int = 1              # for inflation tracking
    task_id: str = "default"
