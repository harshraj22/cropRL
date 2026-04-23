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


class ActionType(IntEnum):
    """All discrete actions the agent can take."""

    NO_OP = 0
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
    POST_FORUM = 11

# Backward-compatible alias
ActionType.WAIT = ActionType.NO_OP


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
