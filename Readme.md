# CropRL — Agricultural Decision-Making Environment

CropRL simulates the core decision-making challenges of a small Indian farm over a 5-year (60-month) horizon. An agent manages a single plot of land, choosing what to plant, when to harvest, how to manage soil health, and when to take financial risks — all under stochastic weather and fluctuating commodity prices.

The goal is to **maximize net worth** (cash + asset value − debt) by the end of the episode, while keeping the soil healthy for long-term productivity.

---

## The Domain

### Why Agriculture?

Farming is one of the oldest sequential decision-making problems. Every month, a farmer faces choices with **delayed, uncertain outcomes**:

- Do I plant an expensive crop that could yield 10× profit, or a safe one that at least won't bankrupt me?
- Should I sell now at a low price, or store the harvest and gamble on prices rising — knowing it might rot?
- Rain has been poor this year. Do I take a loan to irrigate, adding debt with interest?

These are the same exploration-exploitation, risk-reward, and long-horizon planning tradeoffs that make RL interesting — grounded in a domain people can intuitively understand.

### The Indian Context

The environment draws inspiration from Indian agricultural patterns:

- **Monsoon dependency** — the June–September monsoon delivers ~80% of the year's rainfall. Timing decisions around it is critical.
- **Seasonal price cycles** — prices dip post-harvest (Oct–Dec) when supply floods the market, and peak pre-monsoon (Apr–May) when stocks thin out.
- **Crop rotation** — corn depletes soil nitrogen while chickpea (a legume) restores it, mirroring real crop science.
- **Rural credit** — loans with dynamic interest rates that spike during droughts and planting seasons.

---

## The Crops

The agent manages three crop types that form a strategic triangle:

| Crop | Category | Seed Cost | Growth | Base Yield | Soil Impact | Base Price |
|------|----------|-----------|--------|------------|-------------|------------|
| **Corn** | Heavy Feeder | ₹800 | 4 months | 8 tons | **−0.25 N** (depletes) | ₹1,200/ton |
| **Wheat** | Medium Feeder | ₹500 | 3 months | 5 tons | −0.10 N (mild drain) | ₹800/ton |
| **Chickpea** | Legume | ₹200 | 3 months | 3 tons | **+0.15 N** (restores) | ₹500/ton |

**The core tension:** Corn is the most profitable but destroys the soil. Chickpea restores it but earns less. A model that learns Corn monoculture will see yields collapse within a few cycles as nitrogen depletes. The optimal policy involves **crop rotation** — a concept the agent must discover on its own.

---

## What the Agent Observes

Each month, the agent receives a full dashboard:

- **Time & Weather** — current month, season, and this month's rainfall (0.0–1.0)
- **Farm Status** — what's planted, crop age, soil nitrogen level (0.0–1.0), expected yield potential
- **Finances** — cash balance, debt, current interest rate
- **Market Prices** — current spot prices for all three crops
- **Storage** — what's in the warehouse and how old it is
- **Available Actions** — which of the 11 actions are currently valid

## What the Agent Can Do

Each month, the agent picks one of 11 actions:

| ID | Action | Effect |
|----|--------|--------|
| 0 | Wait | Do nothing this month |
| 1–3 | Plant Corn / Wheat / Chickpea | Spend seed cost, occupy land |
| 4 | Irrigate | Spend ₹300, mitigate 70% of drought impact on crops |
| 5 | Fertilize | Spend ₹400, boost soil nitrogen by +0.15 |
| 6 | Harvest & Store | Clear land, put harvest in warehouse |
| 7 | Harvest & Sell | Clear land, sell immediately at current market price |
| 8 | Sell Inventory | Sell whatever is in storage |
| 9 | Take Loan | Get ₹5,000 cash, start accumulating interest |
| 10 | Repay Loan | Pay off full debt if you have enough cash |

Invalid actions (e.g., planting when a crop is already growing) are penalized (−50 reward) and no-op'd.

---

## Sources of Uncertainty

Four stochastic processes drive the environment. All Gaussian samples are clamped to ±3σ, and all outputs have hard floor/ceiling clamps to prevent runaway values.

| Source | When | Formula | Clamp |
|--------|------|---------|-------|
| **Rainfall** | Every month | `clip(μ(month) + N(0, 0.15²), 0, 1)` | [0, 1] |
| **Market Prices** | Every month | Mean-reverting random walk with seasonal targets | [1, base × 2.5] |
| **Yield Noise** | At harvest | `deterministic_yield × (1 + N(0, 0.10²))` | [0, ∞) with ±3σ noise clamp |
| **Demand Shocks** | ~8%/month | One random crop gets ±30-60% price shock | Respects price ceiling |

### How Prices Work

Prices follow a **mean-reverting random walk** — each month's price is anchored to the previous month's, pulled toward the seasonal target:

```
target = base_price × seasonal_multiplier
drift  = 0.3 × (target − P_prev) / target
P_new  = P_prev × (1 + drift + noise)
```

This means price trends emerge (a rising corn market may keep rising), making storage speculation a learnable strategy — unlike independent draws where past prices carry no information.

Occasionally (~1 per year), a **demand shock** spikes or dips one crop's price by 30-60%, simulating export orders, festival demand, or import gluts.

### Seasonal Weather Baselines

| Season | Months | Baseline Rainfall |
|--------|--------|-------------------|
| **Monsoon** | Jun–Sep | 0.80 (heavy rains) |
| **Spring** | Feb–Mar | 0.40 |
| **Winter** | Oct–Jan | 0.20 |
| **Summer** | Apr–May | 0.05 (drought) |

### Seasonal Price Multipliers

| Period | Months | Price Multiplier |
|--------|--------|-----------------|
| Pre-monsoon | Apr–May | 1.15× (scarce supply) |
| Monsoon | Jun–Sep | 1.00× |
| Winter | Jan–Mar | 0.95× |
| Post-harvest | Oct–Dec | 0.85× (oversupply) |

### What's Deterministic

Everything else is a fixed function of state: nitrogen drain/recovery, costs, interest accumulation, spoilage countdown, expected yield potential (shown in the observation as a planning aid — uses the deterministic yield formula without noise).

---

## What a Good Policy Learns

A well-trained agent should discover:

1. **Crop rotation** — alternate nitrogen-depleting crops (corn) with legumes (chickpea) to sustain soil
2. **Seasonal timing** — plant before the monsoon (June) to exploit free water, sell before the post-harvest dip
3. **Irrigation calculus** — irrigate during summer droughts when it saves a high-value crop, skip when monsoon rains are sufficient
4. **Storage speculation** — store harvests during price troughs and sell during spikes; price autocorrelation makes trends learnable
5. **Shock exploitation** — maintain inventory or ready-to-harvest crops to capitalize on demand shocks
6. **Financial prudence** — avoid loans unless the expected return exceeds interest costs; repay before interest compounds
7. **Soil stewardship** — a terminal bonus of `nitrogen × 10,000` rewards agents that leave the farm in good shape, not just strip-mined

---

## Simplifying Assumptions

These are deliberate simplifications to keep the environment tractable for RL while preserving meaningful decision-making:

| Assumption | Reality | Why We Simplified |
|------------|---------|-------------------|
| Single plot of land | Farms have multiple fields | Multi-field = huge state/action space |
| One action per step | Farmers do multiple tasks daily | Action-space explosion (2^11 combos) |
| Monthly time steps | Decisions are daily/weekly | Keeps episodes at 60 steps |
| 3 crop types | Hundreds exist | Enough for the rotation tradeoff |
| Instant harvest | Harvest takes weeks | Avoids mid-action state complexity |
| No labor constraints | Labor is a real bottleneck | Would need another resource dimension |
| No pests or disease | Major risk in real farming | Future scope (see IDEAS.md) |
| Fixed costs | Costs fluctuate seasonally | Future scope (see IDEAS.md) |
| No government policy | Subsidies, MSP affect decisions | Out of scope for MVP |
| Deterministic spoilage | Real spoilage is probabilistic | Future scope (see IDEAS.md) |

---

## Difficulty Tiers

The environment supports three difficulty presets:

| Parameter | Easy | Medium | Hard |
|-----------|------|--------|------|
| Starting cash | ₹15,000 | ₹10,000 | ₹7,000 |
| Interest rate | 0% | 8% | 12% |
| Weather noise | Standard | Standard | Standard |
| Max steps | 60 | 60 | 60 |

**Easy** removes loan interest entirely and gives generous starting capital — the agent just needs to learn crop rotation basics. **Hard** starts with less cash and punishing interest rates, requiring tight financial management.

---

## Reward Signal

Each step's reward = `Δcash + penalties + terminal_bonus`

- **Cash delta** — the change in cash balance from the previous step (revenue − costs)
- **Invalid action penalty** — −50 for attempting unavailable actions
- **Bankruptcy penalty** — −1,000 if cash goes negative with active debt
- **Terminal soil bonus** — on the final step: `soil_nitrogen × 10,000` (incentivizes long-term sustainability over pure extraction)

---

## Future Improvements

See [IDEAS.md](./IDEAS.md) for planned extensions including:

- **Pest / disease events** — random crop damage creating early-harvest risk calculus
- **Stochastic spoilage** — probabilistic rot replacing deterministic countdown
- **Correlated crop prices** — modeling substitution effects between crops
- **Machinery system** with degradation and repair costs
- **Storage costs** creating sell-vs-hold pressure
- **Dynamic costs** for seeds and inputs
- **Multi-field farming** for more complex spatial planning
- **Action masking** for efficient RL training

