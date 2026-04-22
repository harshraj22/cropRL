"""
CropRL Enumerations and Constants.

Central location for all domain enums and mappings used across the codebase.
"""

from enum import IntEnum, Enum
from typing import Dict


class Season(str, Enum):
    """Calendar seasons matching Indian agricultural cycles."""

    MONSOON = "Monsoon"   # Jun–Sep: heavy rains, Kharif planting
    WINTER = "Winter"     # Oct–Jan: Rabi season, cool/dry
    SPRING = "Spring"     # Feb–Mar: transition, some Rabi harvest
    SUMMER = "Summer"     # Apr–May: hot, pre-monsoon drought


class CropType(IntEnum):
    """Crop types available in the environment."""

    FALLOW = 0       # No crop planted
    CORN = 1         # Heavy Feeder — high yield, destroys nitrogen
    WHEAT = 2        # Medium Feeder — moderate yield, mild nitrogen drain
    CHICKPEA = 3     # Legume — lower yield, restores nitrogen
    # — Hype crops —
    MATCHA = 4       # Specialty: niche market, high hype premium
    QUINOA = 5       # Specialty: medium hype premium
    TURMERIC = 6     # Specialty: moderate hype premium


class ActionType(IntEnum):
    """All discrete actions the agent can take."""

    # In multi-agent mode, action 0 is END_TURN (replaces WAIT semantics)
    WAIT = 0
    END_TURN = 0   # alias for WAIT in multi-agent contexts
    PLANT_CORN = 1
    PLANT_WHEAT = 2
    PLANT_CHICKPEA = 3
    IRRIGATE = 4
    FERTILIZE = 5
    HARVEST_STORE = 6
    HARVEST_SELL = 7
    SELL_INVENTORY = 8
    TAKE_LOAN = 9
    REPAY_LOAN = 10
    POST_MESSAGE = 11    # Multi-agent: post to the public forum
    PLANT_MATCHA = 12    # Hype crop
    PLANT_QUINOA = 13    # Hype crop
    PLANT_TURMERIC = 14  # Hype crop


class HypePhase(str, Enum):
    """Phases of a hype-crop demand cycle."""

    DORMANT    = "dormant"
    BUILDING   = "building"
    PEAK       = "peak"
    COLLAPSING = "collapsing"


class LedgerEventType(str, Enum):
    """Types of events recorded on the public ledger."""

    PLANTED          = "planted"
    IRRIGATED        = "irrigated"
    FERTILIZED       = "fertilized"
    HARVESTED_STORED = "harvest_store"
    HARVESTED_SOLD   = "harvest_sell"
    SOLD_INVENTORY   = "sell_inventory"
    LOAN_TAKEN       = "loan_taken"
    LOAN_REPAID      = "loan_repaid"
    END_TURN         = "end_turn"
    FORUM_MESSAGE    = "forum_message"


class ForumMsgType(str, Enum):
    """Structured message intent tags for forum posts."""

    INTENT    = "intent"     # "I plan to…"
    PROPOSAL  = "proposal"   # "Let's coordinate…"
    WARNING   = "warning"    # "Avoid…"
    BOAST     = "boast"      # "I just made…"
    DECEPTION = "deception"  # self-tagged deception


# ── Month ↔ Season mapping ────────────────────────────────────────

MONTH_TO_SEASON: Dict[int, Season] = {
    1: Season.WINTER,
    2: Season.SPRING,
    3: Season.SPRING,
    4: Season.SUMMER,
    5: Season.SUMMER,
    6: Season.MONSOON,
    7: Season.MONSOON,
    8: Season.MONSOON,
    9: Season.MONSOON,
    10: Season.WINTER,
    11: Season.WINTER,
    12: Season.WINTER,
}

MONTH_NAMES = (
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


def get_season(month: int) -> Season:
    """Return the Season for a given calendar month (1–12)."""
    return MONTH_TO_SEASON[month]
