"""
CropRL Pydantic Models.

Defines the Action, Observation, and State types used by the OpenEnv framework.
These models provide type-safe serialization over WebSocket / HTTP.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class CropAction(Action):
    """
    Agent selects a discrete action each month.

    Action IDs:
        0  — Wait (do nothing)
        1  — Plant Crop 1 (Corn / Heavy Feeder)
        2  — Plant Crop 2 (Wheat / Medium Feeder)
        3  — Plant Crop 3 (Chickpea / Legume)
        4  — Irrigate
        5  — Fertilize
        6  — Harvest & Store
        7  — Harvest & Sell
        8  — Sell Inventory
        9  — Take Loan
        10 — Repay Loan
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


class CropObservation(Observation):
    """
    Full farm dashboard the agent sees each month.

    Inherits from Observation:
        done: bool       — whether the episode has terminated
        reward: float    — reward signal from the last action
        metadata: dict   — additional metadata
    """

    # ── Time & Weather ──────────────────────────────────────────
    current_month: int = Field(description="Calendar month 1-12")
    current_step: int = Field(description="Step index 0..max_steps-1")
    expected_rainfall: float = Field(
        description="Generated rainfall for this month, 0.0 to 1.0"
    )

    # ── Biological & Soil ───────────────────────────────────────
    active_crop_type: int = Field(
        description="0=Fallow, 1=Corn, 2=Wheat, 3=Chickpea"
    )
    crop_age_months: int = Field(description="Months since planting")
    expected_yield_potential: float = Field(
        description="Estimated yield if harvested now, normalized 0.0-1.0"
    )
    soil_nitrogen: float = Field(description="Soil nitrogen level 0.0-1.0")

    # ── Financial ───────────────────────────────────────────────
    cash_balance: float = Field(description="Current cash on hand")
    current_debt: float = Field(description="Outstanding loan debt")
    current_interest_rate: float = Field(
        description="Current annual interest rate"
    )
    market_price_crop_1: float = Field(description="Spot price for Corn (₹/ton)")
    market_price_crop_2: float = Field(description="Spot price for Wheat (₹/ton)")
    market_price_crop_3: float = Field(
        description="Spot price for Chickpea (₹/ton)"
    )

    # ── Costs (visible to agent) ────────────────────────────────
    cost_seed_1: float = Field(description="Cost to plant Corn")
    cost_seed_2: float = Field(description="Cost to plant Wheat")
    cost_seed_3: float = Field(description="Cost to plant Chickpea")
    cost_irrigate: float = Field(description="Cost to irrigate")
    cost_fertilize: float = Field(description="Cost to fertilize")

    # ── Inventory ───────────────────────────────────────────────
    stored_crop_type: int = Field(
        description="Type of crop in storage (0=empty)"
    )
    stored_amount: float = Field(description="Tons of crop in storage")
    stored_age_months: int = Field(description="Months since stored crop was harvested")

    # ── Info ────────────────────────────────────────────────────
    message: str = Field(
        default="",
        description="Human-readable feedback about what happened this step",
    )
    text_summary: str = Field(
        default="",
        description="Full text observation (only populated when text_mode=True)",
    )


class CropState(State):
    """
    Internal environment state (superset of observation).

    Inherits from State:
        episode_id: str  — unique episode identifier
        step_count: int  — number of steps taken
    """

    # ── Internal Bookkeeping ────────────────────────────────────
    irrigated_this_month: bool = False
    fertilized_this_month: bool = False
    previous_cash: float = 0.0
    has_active_loan: bool = False
    task_id: str = "default"
