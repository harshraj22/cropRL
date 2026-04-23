"""
PublicLedger & Forum — shared information layer for multi-agent CropRL.

The PublicLedger records every action taken by every agent within a month.
Agents see events that happened *before* their current slot (post-planting
reveal rule from the design plan).

The Forum is a capped message board co-located here because it also
emits LedgerEvents for observability.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from cropRL.enums import ForumMsgType, LedgerEventType
from cropRL.models import ForumMessage, LedgerEvent


class PublicLedger:
    """
    Records all public actions within an episode.

    Each month's events are tracked separately so agents can query
    "everything that happened in the current month up to (but not including)
    my current slot".
    """

    def __init__(self) -> None:
        # All events ever recorded, keyed by (month, slot)
        self._all_events: List[LedgerEvent] = []
        # Events in the current month only (reset on advance_month())
        self._month_events: List[LedgerEvent] = []

    # ──────────────────────────────────────────────────────────────
    # Recording
    # ──────────────────────────────────────────────────────────────

    def record(self, event: LedgerEvent) -> None:
        """Append an event to the ledger."""
        self._all_events.append(event)
        self._month_events.append(event)

    def reset_month(self) -> None:
        """Called at the start of each new month — clears the month window."""
        self._month_events = []

    # ──────────────────────────────────────────────────────────────
    # Querying
    # ──────────────────────────────────────────────────────────────

    def events_this_month(self, since_slot: int = 0) -> List[LedgerEvent]:
        """
        Return events in the current month with slot >= since_slot.

        Args:
            since_slot: Only return events from this slot onwards.
                        Pass 0 to get all events this month.
        """
        return [e for e in self._month_events if e.slot >= since_slot]

    def events_before_slot(self, slot: int) -> List[LedgerEvent]:
        """
        Return all events this month that happened *before* the given slot.
        Used to populate an agent's observation: they can see what happened
        before their current turn in this month.
        """
        return [e for e in self._month_events if e.slot < slot]

    def planted_crops_this_month(self, before_slot: int) -> Dict[int, int]:
        """
        Return a mapping of agent_id → crop_type for all PLANTED events
        that occurred before *before_slot* this month.
        Agents at slot 0 see nothing; agents at later slots can see early planters.
        """
        result: Dict[int, int] = {}
        for e in self._month_events:
            if e.slot < before_slot and e.event_type == LedgerEventType.PLANTED:
                result[e.agent_id] = e.payload.get("crop_type", 0)
        return result

    def all_events(self) -> List[LedgerEvent]:
        """Return the complete event log for the episode."""
        return list(self._all_events)


class Forum:
    """
    Shared message board.  Each agent may post at most
    *messages_per_month* messages per month.  Messages cost 1 action slot.

    Messages are visible to all agents at the moment they are posted
    (they appear in subsequent agents' observations during the same month).
    """

    def __init__(
        self,
        num_agents: int,
        messages_per_month: int,
        ledger: PublicLedger,
    ) -> None:
        self._num_agents = num_agents
        self._messages_per_month = messages_per_month
        self._ledger = ledger

        # Per-agent post count this month
        self._post_count: Dict[int, int] = {i: 0 for i in range(num_agents)}
        # All messages this month
        self._month_messages: List[ForumMessage] = []

    # ──────────────────────────────────────────────────────────────
    # Posting
    # ──────────────────────────────────────────────────────────────

    def post(
        self,
        agent_id: int,
        month: int,
        slot: int,
        text: str,
        msg_type: ForumMsgType = ForumMsgType.INTENT,
    ) -> tuple[bool, str]:
        """
        Post a message to the forum.

        Returns
        -------
        (success, message_text)
            success=False when this agent has exceeded its monthly cap.
        """
        if self._post_count[agent_id] >= self._messages_per_month:
            return False, (
                f"INVALID: Forum post limit reached "
                f"({self._messages_per_month} per month)."
            )

        text = text[:150]
        msg = ForumMessage(
            agent_id=agent_id,
            month=month,
            slot=slot,
            text=text,
            msg_type=msg_type,
        )
        self._month_messages.append(msg)
        self._post_count[agent_id] += 1

        # Also record on the public ledger for full observability
        self._ledger.record(
            LedgerEvent(
                agent_id=agent_id,
                month=month,
                slot=slot,
                event_type=LedgerEventType.FORUM_MESSAGE,
                payload={"text": text, "msg_type": msg_type.value},
            )
        )
        return True, f"[Forum] Agent {agent_id}: {text}"

    # ──────────────────────────────────────────────────────────────
    # Querying
    # ──────────────────────────────────────────────────────────────

    def messages_this_month(self) -> List[ForumMessage]:
        """Return all messages posted this month."""
        return list(self._month_messages)

    def posts_remaining(self, agent_id: int) -> int:
        """How many posts can this agent still make this month?"""
        return max(0, self._messages_per_month - self._post_count[agent_id])

    # ──────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────

    def reset_month(self) -> None:
        """Clear per-month state.  Called when the month advances."""
        self._month_messages = []
        for i in range(self._num_agents):
            self._post_count[i] = 0
