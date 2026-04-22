# Multi-Agent CropRL — Implementation Plan

> Transforming the single-farm CropRL benchmark into a competitive/cooperative multi-agent economy with emergent price dynamics, social signalling, and trending "hype crops".

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Month Advancement & Action Budget](#2-month-advancement--action-budget)
3. [Per-Agent Land & Independent Farms](#3-per-agent-land--independent-farms)
4. [Supply-Demand Market Engine](#4-supply-demand-market-engine)
5. [Public Ledger & Information Sharing](#5-public-ledger--information-sharing)
6. [Social Media / Hype Crops](#6-social-media--hype-crops)
7. [Public Forum & Agent Communication](#7-public-forum--agent-communication)
8. [Objective & Grading](#8-objective--grading)
9. [New & Modified Files](#9-new--modified-files)
10. [Data Model Changes](#10-data-model-changes)
11. [Step-by-Step Implementation Order](#11-step-by-step-implementation-order)
12. [Open Design Questions](#12-open-design-questions)

---

## 1. Architecture Overview

```
MultiAgentCroprlEnvironment
├── agents[N]  →  each wraps a private FarmState  (land, cash, crops…)
├── MarketEngine              ←  shared, owns supply/demand curves + hype engine
├── PublicLedger              ←  shared, records broadcasts after each action
├── Forum                     ←  shared, message board (capped N msgs/month)
└── TimeController            ←  shared, owns the calendar month & action-budget gate
```

All per-agent farm logic (yield maths, nitrogen, irrigation, soil) stays in `dynamics.py` — **unchanged**. The multi-agent layer sits on top as an orchestrator, not a rewrite.

---

## 2. Month Advancement & Action Budget

### The Core Problem

Currently, only `Wait` advances the month. In a multi-agent setting this breaks in two ways:

- If each agent has its own `Wait`, months advance asynchronously → agents drift out of sync.
- Requiring `Wait` before `Irrigate` means a freshly sown crop could die waiting.

### Proposed Strategy: **Slot-Based Turn System**

Each calendar month is divided into **action slots**. Every agent starts the month with a fixed budget of `K` slots (default `K = 4`). The month only advances when **every agent** has either exhausted their slots or explicitly called `End Turn`.

```
Month M begins
  ┌─ Agent A: Plant Corn  (slot 1/4)
  │           Irrigate    (slot 2/4)
  │           Fertilize   (slot 3/4)
  │           End Turn    (burns slots 4, month signal sent)
  │
  ├─ Agent B: Plant Wheat (slot 1/4)
  │           End Turn    (burns slots 2-4, month signal sent)
  │
  └─ Agent C: End Turn    (burns all 4 slots immediately)
                ↓
  TimeController detects all agents have signalled → advances to Month M+1
```

Key rules:

- `End Turn` (action ID `0`, replaces `Wait`) signals the TimeController that this agent is done for the month. It can be called at any slot.
- Actions that previously advanced the month (`Wait`) now only send the "done" signal; the month advances only when all agents agree.
- Any action taken after `End Turn` in the same month is rejected as invalid.
- Agents that call `End Turn` early simply wait while others finish — they receive updated ledger/forum state in the meantime, but cannot act.

### Why not simultaneous submission?

Simultaneous submission (all agents send all actions at once) makes communication and forum interactions meaningless — there is no point in reading what another agent planted if you submitted your planting decision at the same time. The slot-based model lets agents react to ledger events that happened **earlier in the same month**, which is strategically meaningful.

### Config additions

```python
@dataclass
class MultiAgentConfig:
    num_agents: int = 4
    action_slots_per_month: int = 4       # K — max actions before forced end-turn
    forum_messages_per_month: int = 2     # max public broadcasts per agent per month
```

---

## 3. Per-Agent Land & Independent Farms

### Farm Isolation

Each agent `i` owns a completely private `FarmState_i`:

- Its own `soil_nitrogen`, `water_level`, `active_crop_type`, `stored_*`, `cash`, `debt`.
- Private `rng_i` seeded deterministically from `global_seed + agent_id` → reproducible, but uncorrelated weather realisations per farm.
- Private inflated cost vectors (inflation still advances in lockstep with the shared calendar).

### Shared vs Private State

| Variable | Scope | Reason |
|----------|-------|--------|
| `soil_nitrogen`, `water_level`, `cash`, `debt` | Private | Independent farms |
| `calendar_month`, `year` | Shared | All farms experience the same time |
| `base_market_prices` (pre-supply-shock) | Shared | Same commodity baseline |
| `realised_market_prices` (post-supply-shock) | Shared | Emerges from collective sell actions |
| `inflation_factor` | Shared | Macro-economy is the same for all |

### Final Objective

The environment exposes **three objective modes** selectable via config:

| Mode | Description | Score |
|------|-------------|-------|
| `competitive` | Pure self-interest; each agent maximises own terminal net worth | Per-agent score via existing `grader()` |
| `cooperative` | Maximise sum of all net worths (e.g. village welfare) | `mean(grader(agent_i))` |
| `mixed` | Each agent scored on own net worth, but a bonus if village average > threshold | `0.7 * own_score + 0.3 * village_avg` |

The default for the benchmark is `competitive`. This incentivises agents to genuinely compete on crop selection and market timing.

---

## 4. Supply-Demand Market Engine

### Design Goals

- When many agents harvest and sell the same crop in the same month, the price they all receive is lower.
- Buyers are modelled implicitly via an inverse demand curve — no explicit buyer agents needed.
- The effect is proportional: one extra seller lowers price modestly; a glut lowers it dramatically.

### Price Impact Model

```
realised_price(crop) = base_price(crop) × seasonal_mult × hype_mult
                       × demand_response(total_sold_volume, crop)
```

where:

```
demand_response(V, crop) = max(price_floor_mult,
                               1 − price_impact_coeff × (V / market_capacity))
```

Parameters (in `MultiAgentConfig`):

```python
market_capacity: dict[CropType, float] = {
    CORN:     50.0,   # tons — market can absorb this much at full price
    WHEAT:    40.0,
    CHICKPEA: 20.0,
    MATCHA:   5.0,    # niche market, saturates fast
}
price_impact_coeff: float = 1.0   # controls how steeply glut depresses price
price_floor_mult:   float = 0.3   # price can never fall below 30% of base
```

### Order of Operations Each Month

1. All agents complete their action slots. Sell actions are queued.
2. `MarketEngine.resolve_sales()` is called once — it aggregates total sold volume per crop across all sell-queue entries, computes `demand_response`, and updates `realised_price`.
3. Revenue for each sell action = `units_sold × realised_price` (same price for everyone who sold that month).
4. `realised_price` is broadcast on the `PublicLedger` so agents can learn the market depth.

### Storage & Deferred Selling

Agents who harvest-and-store avoid the current month's glut. This creates a genuine strategic choice: sell now vs. wait for the glut to clear. The `MarketEngine` does not apply demand-response to `Sell Inventory` that happens in a different month from the bulk harvest (each month is a fresh resolve cycle).

### Price Autocorrelation with Supply-Demand

The existing mean-reverting random walk is preserved. Supply-demand response is a **multiplicative overlay**, not a replacement:

```
P_t = P_rw_t × demand_response(V_t)
```

This means the base random walk still provides price memory and trends; collective agent actions create additional shocks on top.

---

## 5. Public Ledger & Information Sharing

### Philosophy

Agents should gain information about others' actions **only after those actions are taken** in the current month's slot sequence. This mirrors real-world signals (a farmer sees a neighbour planting corn; they can still react this month if they have slots left).

### What Gets Broadcast

```python
class LedgerEvent:
    agent_id:    int
    month:       int
    slot:        int          # which slot within the month
    event_type:  LedgerEventType
    payload:     dict         # type-specific detail

class LedgerEventType(str, Enum):
    PLANTED       = "planted"       # crop_type
    IRRIGATED     = "irrigated"     # (no extra detail)
    FERTILIZED    = "fertilized"    # (no extra detail)
    HARVESTED_STORED = "harvest_store"  # crop_type, amount
    HARVESTED_SOLD   = "harvest_sell"   # crop_type, amount
    SOLD_INVENTORY   = "sell_inventory" # crop_type, amount
    LOAN_TAKEN    = "loan_taken"    # (no amount — privacy preserved)
    LOAN_REPAID   = "loan_repaid"
    END_TURN      = "end_turn"
    FORUM_MESSAGE = "forum_message" # text
```

### What Is Kept Private

- Exact cash balance and debt — no agent sees another's finances.
- Soil nitrogen and water levels — private farm conditions.
- Whether a specific agent can afford something.

### Observation Augmentation

`MultiAgentObservation` wraps the existing `CroprlObservation` and adds:

```python
class MultiAgentObservation(CroprlObservation):
    agent_id:            int
    month_slot:          int          # current slot (0..K-1)
    slots_remaining:     int
    
    # Ledger: everything that happened this month so far (all agents)
    ledger_this_month:   list[LedgerEvent]
    
    # Forum: messages posted this month (all agents, capped)
    forum_this_month:    list[ForumMessage]
    
    # Summary of what each other agent has planted (post-planting reveal)
    other_agents_crops:  dict[int, int]  # agent_id → CropType (0 if unknown/fallow)
    
    # Realised prices from last month's resolve (for learning market depth)
    last_month_realised_prices: tuple[float, float, float]
```

---

## 6. Social Media / Hype Crops

### Concept

Hype crops are specialty commodities whose prices are driven by a demand trend cycle independent of seasonal patterns. Examples: matcha, quinoa, turmeric, moringa, purple yam. Their prices spike during a "hype wave" and collapse when the wave passes — often triggered (or accelerated) by many agents piling in.

### Hype Engine

```python
class HypeCycle:
    crop_type:    CropType
    hype_level:   float        # 0.0 → 1.0
    phase:        HypePhase    # BUILDING, PEAK, COLLAPSING, DORMANT
    duration:     int          # months remaining in current phase
```

```
DORMANT  ─(random trigger)→ BUILDING ─(threshold)→ PEAK ─(timeout or glut)→ COLLAPSING → DORMANT
```

Transition logic:

- `DORMANT → BUILDING`: triggered randomly (~5% per month) or by a global demand_shock event.
- `BUILDING`: `hype_level += 0.08/month`. Transition to `PEAK` when `hype_level ≥ 0.9`.
- `PEAK`: `hype_level = 1.0`. Lasts 2-4 months. Transition to `COLLAPSING` if total sold volume > `market_capacity × 0.6` (farmers over-supply and the hype collapses early) OR after fixed duration.
- `COLLAPSING`: `hype_level -= 0.20/month`. Returns to `DORMANT` at 0.
- `DORMANT`: `hype_level = 0.0`.

### Price Multiplier

```
hype_mult(crop) = 1.0 + hype_level × (max_hype_premium − 1.0)
```

Where `max_hype_premium = 3.5` for matcha (3.5× base price at peak hype). Combined with demand-response and seasonality, a well-timed hype harvest can be enormously profitable — or devastating if the wave collapses mid-grow.

### Hype Crop Properties

| Crop | Base Price | Max Hype Premium | Seed Cost | Growth | Soil Impact | Risk |
|------|-----------|-----------------|-----------|--------|-------------|------|
| Matcha | ₹2,500/ton | 3.5× | ₹1,500 | 5 months | −0.12 N | High |
| Quinoa | ₹1,800/ton | 2.8× | ₹900 | 3 months | −0.06 N | Medium |
| Turmeric | ₹1,200/ton | 2.2× | ₹600 | 4 months | −0.04 N | Medium-low |

### Agent Visibility

The `hype_level` for each crop is **observable** — agents see it in their dashboard. The `phase` is also shown, so they can judge how much runway remains. The collapse trigger threshold is not shown (adds genuine risk). In text mode, a "Social Media Trends" block is appended to the observation.

### Supply-Demand Interaction

Hype crops have a **much smaller `market_capacity`** (e.g., matcha: 5 tons). This means even 2-3 agents simultaneously selling matcha causes a severe price impact. The combination of demand-response + collapsing hype creates a classic coordination dilemma: everyone wants to be the first to sell, not the last.

---

## 7. Public Forum & Agent Communication

### Design

The `Forum` is a shared message board. Each agent may post up to `forum_messages_per_month` messages per month (default: 2). Messages are broadcast to all agents at the moment they are posted (they appear in subsequent agents' observations during the same month's slot sequence).

### Message Structure

```python
class ForumMessage:
    agent_id:   int
    month:      int
    slot:       int
    text:       str           # free-form for LLM agents
    msg_type:   ForumMsgType  # INTENT, PROPOSAL, WARNING, BOAST, DECEPTION
```

`msg_type` is optional metadata that structured agents can set. LLM agents use free-form `text`.

### Action

A new action `POST_MESSAGE` (action ID `11`) is added:

```
11: Post Forum Message — Broadcast a text message to all agents (costs 1 slot, capped per month)
```

The action payload includes the message text. For discrete-action RL agents, structured message tokens can be used (e.g., "I will plant CORN", "AVOID WHEAT this month").

### Strategic Use Cases

- **Coordination**: "I'm planting Chickpea this month — who takes Corn?" → reduces glut risk.
- **Deception**: "I heard Wheat prices are going to spike" → trick others into planting wheat while you plant corn.
- **Warning**: "Matcha hype is collapsing — sell before next month!"
- **Cooperation on loans**: "I'm taking a loan to plant Matcha — if price stays up we can both profit."

### Forum Costs & Limits

- Posting costs 1 action slot (same as any other action). This creates a real opportunity cost.
- Cap of `forum_messages_per_month` enforced per agent per month. Exceeding it returns an invalid-action penalty.
- Messages cannot be edited or deleted.

### Verifiability

The ledger is ground truth. Agents can cross-check forum claims against the ledger: if an agent claimed they would plant Chickpea but the ledger shows `PLANTED: CORN`, that agent has a provably broken promise — relevant for reputation tracking in future extensions.

---

## 8. Objective & Grading

### Per-Agent Score

Each agent is scored independently using an augmented version of the existing `grader()`. The oracle upper bound is recalculated per agent using their private trajectory (their prices experienced may differ slightly due to individual sell timing within the month).

### Multi-Agent Score

The environment returns both individual scores and an aggregate:

```python
class MultiAgentResult:
    agent_scores:       dict[int, float]     # 0.01 – 0.99 per agent
    aggregate_score:    float                 # mode-dependent (see §3)
    winner_agent_id:    int | None            # None in cooperative mode
    gini_coefficient:   float                 # inequality measure (0=equal, 1=monopoly)
    total_village_nw:   float                 # sum of final net worths
```

The `gini_coefficient` measures how unequally wealth is distributed at episode end — useful for evaluating whether the multi-agent dynamics produce realistic economic stratification.

---

## 9. New & Modified Files

```
cropRL/
├── config.py                  ← ADD MultiAgentConfig dataclass
├── enums.py                   ← ADD HypePhase, LedgerEventType, ForumMsgType
├── models.py                  ← ADD MultiAgentObservation, LedgerEvent, ForumMessage
├── dynamics.py                ← NO CHANGE (all physics stay pure functions)
├── tasks.py                   ← ADD multi-agent task variants (easy/medium/hard × 2/4/8 agents)
│
├── market_engine.py           ← NEW: MarketEngine (supply-demand + hype cycles)
├── public_ledger.py           ← NEW: PublicLedger + Forum
├── time_controller.py         ← NEW: TimeController (slot-based month gating)
│
├── multi_agent_environment.py ← NEW: MultiAgentCroprlEnvironment (orchestrator)
│
└── server/
    ├── app.py                 ← MODIFY: expose multi-agent endpoints
    └── multi_agent_app.py     ← NEW: WebSocket hub for N concurrent agents
```

---

## 10. Data Model Changes

### `enums.py` additions

```python
class HypePhase(str, Enum):
    DORMANT    = "dormant"
    BUILDING   = "building"
    PEAK       = "peak"
    COLLAPSING = "collapsing"

class LedgerEventType(str, Enum):
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
    INTENT   = "intent"     # "I plan to…"
    PROPOSAL = "proposal"   # "Let's coordinate…"
    WARNING  = "warning"    # "Avoid…"
    BOAST    = "boast"      # "I just made…"
    DECEPTION = "deception" # (agents can self-tag; adds deception tracking)
```

### `config.py` addition

```python
@dataclass
class MultiAgentConfig:
    # Agent setup
    num_agents:               int   = 4
    objective_mode:           str   = "competitive"   # competitive|cooperative|mixed

    # Slot-based time
    action_slots_per_month:   int   = 4
    
    # Forum
    forum_messages_per_month: int   = 2
    
    # Market
    market_capacity:          dict  = field(default_factory=lambda: {
        1: 50.0,   # Corn
        2: 40.0,   # Wheat
        3: 20.0,   # Chickpea
        4: 5.0,    # Matcha
        5: 10.0,   # Quinoa
        6: 15.0,   # Turmeric
    })
    price_impact_coeff:       float = 1.0
    price_floor_mult:         float = 0.3

    # Hype crops
    enable_hype_crops:        bool  = True
    hype_trigger_prob:        float = 0.05     # per month, per hype crop
    hype_collapse_supply_threshold: float = 0.6

    # Mixed-mode bonus
    mixed_mode_village_weight: float = 0.3
```

### `models.py` additions

```python
class LedgerEvent(BaseModel):
    agent_id:   int
    month:      int
    slot:       int
    event_type: LedgerEventType
    payload:    dict = {}

class ForumMessage(BaseModel):
    agent_id:  int
    month:     int
    slot:      int
    text:      str
    msg_type:  ForumMsgType = ForumMsgType.INTENT

class HypeCropStatus(BaseModel):
    crop_type:  int
    crop_name:  str
    hype_level: float   # 0.0 – 1.0
    phase:      HypePhase
    months_in_phase: int

class MultiAgentObservation(CroprlObservation):
    # Identity & turn info
    agent_id:              int
    month_slot:            int
    slots_remaining:       int
    forum_posts_remaining: int

    # Shared world state
    other_agents_crops:    dict[int, int]         # post-planting reveal
    ledger_this_month:     list[LedgerEvent]
    forum_this_month:      list[ForumMessage]
    last_month_realised_prices: tuple[float, float, float]

    # Hype
    hype_crop_statuses:    list[HypeCropStatus]

class MultiAgentAction(CroprlAction):
    agent_id:      int
    forum_message: str | None = None   # only used with action_id == 11

class MultiAgentResult(BaseModel):
    agent_scores:      dict[int, float]
    aggregate_score:   float
    winner_agent_id:   int | None
    gini_coefficient:  float
    total_village_nw:  float
```

---

## 11. Step-by-Step Implementation Order

### Phase 1 — Core Multi-Agent Scaffolding (no market changes yet)

**Goal**: N isolated farms, slot-based sync, no interaction.

1. Add `MultiAgentConfig` to `config.py`.
2. Add `HypePhase`, `LedgerEventType`, `ForumMsgType` to `enums.py`.
3. Add `MultiAgentObservation`, `MultiAgentAction` to `models.py`.
4. Implement `time_controller.py`:
   - Tracks `month`, `year`, and per-agent `slots_used` and `turn_done` flags.
   - `submit_turn_end(agent_id)` → marks agent as done; triggers `advance_month()` when all done.
   - `slots_remaining(agent_id)` → raises `TurnOverError` if agent already called End Turn.
5. Implement `multi_agent_environment.py` (skeleton):
   - Holds `List[CroprlEnvironment]` (one per agent) + shared `TimeController`.
   - `reset()` → resets all inner envs, seeds all rngs.
   - `step(agent_id, action)` → routes to correct inner env; calls `time_controller.submit_turn_end` when action is `END_TURN`; blocks early if agent already ended turn.
6. Wire `advance_month()` so it calls `_advance_month()` on all inner envs in lockstep.

**Test**: 4 agents, easy config, all playing `rule_based_agent`. Verify month advances only when all 4 call End Turn. Verify independent farm states.

---

### Phase 2 — Market Engine & Supply-Demand

**Goal**: Selling price depends on collective sell volume.

1. Implement `market_engine.py`:
   - `MarketEngine` holds the authoritative `realised_prices` dict.
   - `queue_sell(agent_id, crop_type, volume)` → appends to pending sell queue.
   - `resolve_month()` → aggregates volume per crop, computes demand-response, returns per-agent revenues, broadcasts `realised_prices` to ledger.
   - Existing random-walk price generation moves to `MarketEngine.generate_base_prices()`.
2. Modify `multi_agent_environment.py`:
   - `_do_harvest_sell` and `_do_sell_inventory` no longer immediately credit cash.
   - Instead, they call `market_engine.queue_sell()` and return a "queued" message.
   - Revenue is credited during `advance_month()` after `resolve_month()`.
3. Update `MultiAgentObservation` with `last_month_realised_prices`.

**Design choice**: agents see each other's `PLANTED` events before the month resolves (via ledger), but the clearing price is only known after `resolve_month()`. This is realistic: you can see your neighbour planting corn but you don't know the clearing price until all selling is done.

**Test**: 4 agents all sell Corn in the same month → price drops vs. baseline. Only 1 agent sells Corn → price near baseline.

---

### Phase 3 — Public Ledger & Information Sharing

**Goal**: Agents observe each other's actions in real time within the month.

1. Implement `public_ledger.py`:
   - `PublicLedger.record(event: LedgerEvent)`.
   - `PublicLedger.events_this_month(since_slot: int)` → returns events after a given slot (so agents only see events that happened before their current slot).
   - `PublicLedger.reset_month()` → called at start of each month.
2. Each action handler in the inner `CroprlEnvironment` emits a `LedgerEvent` via a callback.
   - Easiest approach: `CroprlEnvironment` accepts an optional `on_action` callback that `MultiAgentCroprlEnvironment` passes in.
3. Populate `MultiAgentObservation.ledger_this_month` and `other_agents_crops` from ledger.
4. Update `format_text_observation()` in `dynamics.py` to include a "Neighbours" block when ledger events are non-empty.

**Planting reveal timing**: `other_agents_crops[i]` is 0 (unknown) until agent `i` fires a `PLANTED` event, after which all agents with a later slot that month can see it.

---

### Phase 4 — Forum & Agent Communication

**Goal**: Agents can post and read text messages within the month.

1. Add `Forum` class to `public_ledger.py`:
   - `Forum.post(agent_id, slot, text, msg_type)` → validates monthly cap, appends to log, appends `LedgerEvent(FORUM_MESSAGE)`.
   - `Forum.messages_this_month()` → returns all messages so far.
2. Add action `POST_MESSAGE` (ID `11`) to `ActionType` and `action_names`.
3. In `MultiAgentCroprlEnvironment.step()`, handle `POST_MESSAGE`:
   - Extract `forum_message` from `MultiAgentAction.forum_message`.
   - Call `forum.post()`, deduct 1 slot.
4. Populate `MultiAgentObservation.forum_this_month`.
5. Update `format_text_observation()` to include a "Forum" block.
6. Update `num_actions` to 12.

---

### Phase 5 — Hype Crops

**Goal**: Add 3 hype crops with price trend cycles.

1. Add hype crop entries to `EnvConfig`:
   - `crop_names`, `seed_costs`, `growth_months`, `base_yield_tons`, `monthly_nitrogen_impact`, etc. extended with indices 4 (Matcha), 5 (Quinoa), 6 (Turmeric).
   - `num_crop_types = 7`.
2. Add `HypeCropStatus` fields to `MultiAgentConfig`.
3. Implement `HypeEngine` inside `market_engine.py`:
   - `HypeEngine.tick_month()` → transitions hype phases, returns updated `HypeCropStatus` list.
   - `HypeEngine.get_hype_mult(crop_type)` → returns current price multiplier.
   - Integrates with `resolve_month()`: if sell volume of a hype crop exceeds `hype_collapse_supply_threshold × market_capacity`, trigger early `PEAK → COLLAPSING` transition.
4. Add plant actions for hype crops (IDs `12`, `13`, `14`) or reuse IDs `1–3` with an extended crop enum and keep `ActionType.PLANT_CROP(crop_id)` as a parameterised action.
   - **Recommendation**: keep discrete IDs for simplicity; add `PLANT_MATCHA = 12`, `PLANT_QUINOA = 13`, `PLANT_TURMERIC = 14`. `num_actions = 15`.
5. Add a "Social Media Trends" section to `format_text_observation()`.

---

### Phase 6 — Multi-Agent API & Server

**Goal**: Expose the multi-agent env over HTTP/WebSocket.

1. Implement `server/multi_agent_app.py`:
   - WebSocket endpoint `/ws/{session_id}/{agent_id}` — each agent connects on their own socket.
   - `SessionManager` maps `session_id → MultiAgentCroprlEnvironment` (shared across agent connections).
   - When an agent sends a step, it's routed to `env.step(agent_id, action)`. The response is that agent's `MultiAgentObservation`.
   - When the month advances (all agents done), the server broadcasts a `MONTH_ADVANCED` event to all connected sockets.
2. HTTP fallback: `POST /step` accepts `{session_id, agent_id, action_id, forum_message}`.
3. Update `openenv.yaml` with multi-agent task entries.

---

### Phase 7 — Grading & Evaluation

**Goal**: Clean scoring for benchmarking.

1. Implement `MultiAgentGrader` in `tasks.py`:
   - Calls per-agent `grader()` with per-agent trajectory.
   - Computes Gini coefficient.
   - Returns `MultiAgentResult`.
2. Add multi-agent task entries to `TASKS` dict:
   ```
   "easy_4agent", "medium_4agent", "hard_4agent",
   "easy_8agent", "medium_8agent", "hard_8agent"
   ```
3. Update `inference.py` with a `run_multi_agent_episode()` that spins up N LLM/rule-based agents in threads, each connected via their WebSocket.

---

## 12. Open Design Questions

These are intentional design decisions left for you to resolve before implementation:

**Q1: Sell queue vs. instant sell**
Option A (recommended above): all sells in a month are batched and cleared at month-end. Option B: sells are cleared immediately; first agent to sell gets full price, later ones get depressed price (first-mover advantage). Option B adds urgency but makes slot ordering very important (fairness concern in benchmarking).

**Q2: Slot ordering**
Should agents act in a fixed order (agent 0 always gets slot 1 first) or random? Fixed order introduces positional advantage. Random order is fairer but harder to reproduce. A rotating "first agent" (changes each month) is a good middle ground.

**Q3: Forum message length**
For LLM agents, free-form text is natural. For RL agents, you need a structured discrete message space. Consider supporting both via a `message_vocab` of N structured tokens (e.g., "WILL_PLANT_CORN", "MATCHA_HYPE_HIGH", "AVOID_WHEAT") alongside free-form text.

**Q4: Partial observability of hype**
Should the hype phase be shown directly, or should agents only see the current price? Hiding the phase forces agents to infer it from price trends — harder but more realistic. A middle ground: show `hype_level` but not the phase transition thresholds.

**Q5: Cooperative forum incentives**
In competitive mode, agents have no incentive to post truthful forum messages. If you want genuine coordination to emerge, consider a small "coordination bonus" when agents demonstrably diversify crops (e.g., at least K different crop types planted across all agents in a month).

**Q6: Inflation synchrony**
Inflation currently advances on year boundaries. With multi-agent, do all agents experience inflation simultaneously (recommended — shared macro economy) or independently (no, this creates asymmetry that is hard to reason about)?
