"""
Data models for the CropRL Environment.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State

from .enums import ForumMsgType, HypePhase, LedgerEventType


class CroprlAction(Action):
    """
    Agent selects a discrete action each step.
    """
    action_id: int = Field(
        ge=0,
        le=14,
        description="Discrete action index (0-14)",
        json_schema_extra={
            "enum": [
                ["0 - Wait / No-Op", 0],
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
                ["11 - Post Forum Message", 11],
                ["12 - Plant Matcha (Hype Crop)", 12],
                ["13 - Plant Quinoa (Hype Crop)", 13],
                ["14 - Plant Turmeric (Hype Crop)", 14],
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
    active_crop_type: int = Field(description="0=Fallow, 1=Corn, 2=Wheat, 3=Chickpea, 4=Matcha, 5=Quinoa, 6=Turmeric")
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
    market_price_crop_4: float = Field(description="Spot price for Matcha (Hype Crop)")
    market_price_crop_5: float = Field(description="Spot price for Quinoa (Hype Crop)")
    market_price_crop_6: float = Field(description="Spot price for Turmeric (Hype Crop)")

    # ── Costs ───────────────────────────────────────────────────
    cost_seed_1: float = Field(description="Cost to plant Corn")
    cost_seed_2: float = Field(description="Cost to plant Wheat")
    cost_seed_3: float = Field(description="Cost to plant Chickpea")
    cost_seed_4: float = Field(description="Cost to plant Matcha (Hype)")
    cost_seed_5: float = Field(description="Cost to plant Quinoa (Hype)")
    cost_seed_6: float = Field(description="Cost to plant Turmeric (Hype)")
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-Agent Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LedgerEvent(BaseModel):
    """A single action event visible on the public ledger."""
    agent_id:   int
    month:      int
    slot:       int            # which slot within the month this happened
    event_type: LedgerEventType
    payload:    Dict[str, Any] = Field(default_factory=dict)


class ForumMessage(BaseModel):
    """A public message posted by an agent to the shared forum."""
    agent_id: int
    month:    int
    slot:     int
    text:     str
    msg_type: ForumMsgType = ForumMsgType.INTENT


class HypeCropStatus(BaseModel):
    """Current hype-cycle state for a single hype crop."""
    crop_type:       int    # CropType integer ID (4, 5, or 6)
    crop_name:       str
    hype_level:      float  # 0.0 – 1.0
    phase:           HypePhase
    months_in_phase: int


class MultiAgentObservation(CroprlObservation):
    """
    Extended observation wrapping CroprlObservation with multi-agent context.
    """
    # ── Identity & Turn Info ────────────────────────────────────
    agent_id:              int
    month_slot:            int   # current slot (0..K-1)
    slots_remaining:       int
    forum_posts_remaining: int

    # ── Shared World State ──────────────────────────────────────
    other_agents_crops:         Dict[int, int]         = Field(default_factory=dict)  # agent_id → CropType (0=unknown)
    ledger_this_month:          List[LedgerEvent]      = Field(default_factory=list)
    forum_this_month:           List[ForumMessage]     = Field(default_factory=list)
    last_month_realised_prices: Tuple[float, ...]      = Field(default_factory=tuple)

    # ── Hype Crop Statuses ──────────────────────────────────────
    hype_crop_statuses: List[HypeCropStatus] = Field(default_factory=list)


class MultiAgentAction(CroprlAction):
    """
    Extended action for multi-agent mode.
    Adds agent identity and optional forum message payload.
    """
    agent_id:      int = 0
    forum_message: Optional[str] = None   # only used when action_id == 11


class MultiAgentResult(BaseModel):
    """Final scoring result for a completed multi-agent episode."""
    agent_scores:      Dict[int, float]  # 0.01 – 0.99 per agent
    aggregate_score:   float
    winner_agent_id:   Optional[int]     # None in cooperative mode
    gini_coefficient:  float             # 0=equal, 1=monopoly
    total_village_nw:  float             # sum of all final net worths


class MultiAgentState(State):
    """Internal state for the multi-agent environment."""
    num_agents: int = 0
    current_month: int = 1
    month_count: int = 0
    episode_id: str = ""
    task_id: str = "default"


