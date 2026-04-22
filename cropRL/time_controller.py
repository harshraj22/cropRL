"""
TimeController — Slot-based turn synchronisation for the multi-agent CropRL env.

Each calendar month is divided into K action slots.  Every agent starts the
month with a budget of K slots.  The month only advances when **every** agent
has either exhausted their budget or explicitly called End Turn (action 0).

Design choices aligned with the implementation plan:
- Fixed rotation: the "first agent" rotates each month to avoid positional bias.
- Slot ordering within a month is fixed (agent 0, 1, … N-1 take turns), but
  the first active slot is awarded to a different agent each month.
- Agents that call End Turn early simply wait (blocked) while others finish.
"""

from __future__ import annotations

from typing import Dict, Optional


class TurnOverError(Exception):
    """Raised when an agent tries to act after calling End Turn this month."""


class TimeController:
    """
    Manages the shared calendar month and per-agent action budgets.

    Attributes
    ----------
    month : int
        Current calendar month (1-12).
    year : int
        Current year (1-based).
    month_count : int
        Total months elapsed since episode start.
    """

    def __init__(self, num_agents: int, action_slots_per_month: int) -> None:
        self.num_agents = num_agents
        self.action_slots_per_month = action_slots_per_month

        self.month: int = 1
        self.year: int = 1
        self.month_count: int = 0

        # Per-agent bookkeeping
        self._slots_used: Dict[int, int] = {i: 0 for i in range(num_agents)}
        self._turn_done: Dict[int, bool] = {i: False for i in range(num_agents)}

        # Rotating first-agent index (changes each month, fair rotation)
        self._first_agent_offset: int = 0

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the controller for a new episode."""
        self.month = 1
        self.year = 1
        self.month_count = 0
        self._first_agent_offset = 0
        self._reset_month()

    def slots_remaining(self, agent_id: int) -> int:
        """Return how many action slots agent *agent_id* has left this month."""
        if self._turn_done[agent_id]:
            return 0
        return self.action_slots_per_month - self._slots_used[agent_id]

    def is_turn_done(self, agent_id: int) -> bool:
        """Return True if the agent has signalled End Turn for this month."""
        return self._turn_done[agent_id]

    def consume_slot(self, agent_id: int) -> None:
        """
        Consume one action slot for *agent_id*.

        Raises
        ------
        TurnOverError
            If the agent has already called End Turn this month.
        ValueError
            If the agent has no slots remaining (budget exhausted).
        """
        if self._turn_done[agent_id]:
            raise TurnOverError(
                f"Agent {agent_id} already signalled End Turn this month."
            )
        if self._slots_used[agent_id] >= self.action_slots_per_month:
            # Auto-end turn when budget is exhausted
            self._turn_done[agent_id] = True
            return

        self._slots_used[agent_id] += 1

        # Auto-end turn when budget is now exhausted
        if self._slots_used[agent_id] >= self.action_slots_per_month:
            self._turn_done[agent_id] = True

    def submit_turn_end(self, agent_id: int) -> bool:
        """
        Mark *agent_id* as done for this month.

        Returns
        -------
        bool
            True if ALL agents have now signalled End Turn (month can advance).
        """
        self._turn_done[agent_id] = True
        return self.all_done()

    def all_done(self) -> bool:
        """Return True when every agent has signalled End Turn."""
        return all(self._turn_done.values())

    def advance_month(self) -> bool:
        """
        Advance the calendar by one month.  Returns True when a year boundary
        is crossed (useful for triggering inflation in the outer env).
        """
        old_month = self.month
        self.month = (self.month % 12) + 1
        self.month_count += 1

        year_advanced = False
        if self.month == 1 and old_month == 12:
            self.year += 1
            year_advanced = True

        # Rotate the first-agent offset
        self._first_agent_offset = (self._first_agent_offset + 1) % self.num_agents
        self._reset_month()
        return year_advanced

    def current_slot_for(self, agent_id: int) -> int:
        """Return the slot index (0-based) the agent is *currently* on."""
        return self._slots_used[agent_id]

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _reset_month(self) -> None:
        """Reset per-agent slot budgets and turn-done flags for a new month."""
        for i in range(self.num_agents):
            self._slots_used[i] = 0
            self._turn_done[i] = False
