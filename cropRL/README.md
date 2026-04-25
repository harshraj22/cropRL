---
title: Croprl Environment Server
emoji: ⌨️
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# CropRL — Multi-Agent Agricultural Decision-Making Environment

CropRL simulates the core decision-making challenges of a modern farm ecosystem over a 5-year (60-month) horizon. In this **multi-agent** environment, multiple farmers (agents) operate simultaneously, each managing their own plot of land, finances, and inventory. They choose what to plant, when to harvest, how to manage soil health, and when to take financial risks — all under stochastic weather and fluctuating commodity prices.

Because agents operate in a shared economy, their actions impact one another. If every agent plants the same high-profit crop, market supply will flood at harvest time, crashing the price at the monthly clearing forum. 

The goal for each agent is to **maximize terminal profit** (change in net worth: cash + land value + inventory + growing crops − debt) by the end of the episode, while navigating competition, cooperation, and market dynamics.

---

## Motivation & Real-World Utility

### Why CropRL is a Strong RL Benchmark

Farming is one of the oldest sequential decision-making problems. Every month, a farmer faces choices with **delayed, uncertain outcomes**:

- Do I plant an expensive "hype" crop that could yield massive profit, or a safe one that at least won't bankrupt me?
- Should I sell now at a low price, or store the harvest and gamble on prices rising — knowing it might rot?
- Rain has been poor this year. Do I take a loan to irrigate, adding debt with interest?
- What are my neighbors planting? If they all plant Quinoa, the market will crash. Should I pivot to Chickpea?

These dynamics make CropRL an excellent benchmark for modern Reinforcement Learning algorithms:

- **Multi-Agent Market Dynamics:** Agents must learn Theory of Mind. They communicate via a public forum, form cartels, coordinate planting schedules, or bluff to mislead competitors.
- **Single Unifying Objective:** Despite the multi-domain complexity, the agent is trained on a single primary goal: maximizing terminal profit. It must balance short-term cash vs. preserving soil nitrogen, without hand-crafted, multi-objective reward shaping.
- **Harsh Real-World Constraints:** The environment imposes strict physical and financial bounds. Inflation and compounding interest erode idle cash or loan value, crops have strict physiological development times, and unmitigated weather shocks permanently limit yield potential.
- **Delayed Consequences:** Actions have incredibly long-tail outcomes. Planting corn today ensures a large harvest in 4 months but destroys the soil's yield capacity for the following year. Taking a loan today applies compounding financial pressure for seasons to come.

---

## The Crops

Agents manage 6 crop types (3 standard, 3 hype crops) that form a strategic balance:

| Crop | Category | Seed Cost | Growth | Base Yield | Soil Impact | Base Price |
|------|----------|-----------|--------|------------|-------------|------------|
| **Corn** | Heavy Feeder | ₹800 | 4 months | 8 tons | **−0.08 N/mo** | ₹1,200/ton |
| **Wheat** | Medium Feeder | ₹500 | 3 months | 5 tons | **−0.07 N/mo** | ₹800/ton |
| **Chickpea** | Legume | ₹200 | 3 months | 3 tons | **+0.05 N/mo** | ₹500/ton |
| **Matcha** | Hype Crop | ₹1,500 | 5 months | 2 tons | **−0.12 N/mo** | ₹2,500/ton |
| **Quinoa** | Hype Crop | ₹900 | 3 months | 3 tons | **−0.06 N/mo** | ₹1,800/ton |
| **Turmeric**| Hype Crop | ₹600 | 4 months | 4 tons | **−0.04 N/mo** | ₹1,200/ton |

*Note: Soil impacts are applied monthly during the growth period.*

**The core tension:** Corn is profitable but destroys the soil. Chickpea restores it but earns less. Hype crops (Matcha, Quinoa, Turmeric) offer massive margins but have expensive seed costs, meaning a bad harvest can bankrupt an agent. A model that learns Corn monoculture will see yields collapse within a few cycles as nitrogen depletes. The optimal policy involves **crop rotation** and **market diversification**.

---

## What the Agent Observes

Each month, the agent receives a full dashboard text observation:

- **Time & Weather** — current month, season, and expected rainfall (0.0–1.0)
- **Farm Status** — what's planted, crop age, soil nitrogen level (0.0–1.0), water level, expected yield potential
- **Finances** — cash balance, debt, current interest rate
- **Market Prices** — current spot prices for all 6 crops
- **Storage** — what's in the warehouse and how old it is
- **Public Ledger** — messages posted by other agents in the communication forum

## What the Agent Can Do

Each month, the agent picks one of 15 actions:

| ID | Action | Effect |
|----|--------|--------|
| 0 | Wait / No-Op | Do nothing but consume 1 action slot |
| 1–3 | Plant Corn / Wheat / Chickpea | Spend seed cost, occupy land |
| 4 | Irrigate | Spend ₹300, dynamically boost soil water level depending on the crop |
| 5 | Fertilize | Spend ₹400, boost soil nitrogen by +0.15 |
| 6 | Harvest & Store | Clear land, put harvest in warehouse |
| 7 | Harvest & Sell | Clear land, queue sale for month-end clearing |
| 8 | Sell Inventory | Queue stored crops for month-end sale |
| 9 | Take Loan | Get a lump sum of cash, start accumulating interest |
| 10 | Repay Loan | Pay off full debt if you have enough cash |
| 11 | Post Forum Message | Post a public message to the ledger for all agents to see (Theory of Mind/Coordination) |
| 12–14 | Plant Matcha / Quinoa / Turmeric | Plant a high-risk, high-reward hype crop |

*Note: Invalid actions are penalized (−50 reward) and no-op'd.*

---

## The Market Clearing Forum

Because multiple agents operate in the same economy, CropRL implements a **Clearing Forum** at the end of every month. 
1. Agents queue their crops for sale using actions `7` or `8`.
2. At the end of the month, the environment tallies the total supply of each crop entering the market.
3. If the supply is unusually high, the clearing price for that crop is depressed (slippage). If supply is zero, the price remains at the generated market rate.
4. Agents are paid out at the final clearing price, not the spot price they observed during their turn.

This mechanic heavily penalizes homogeneous agent behavior and strictly requires market coordination or contrarian strategies.

---

## Sources of Uncertainty

Four stochastic processes drive the environment. All Gaussian samples are clamped to ±3σ, and all outputs have hard floor/ceiling clamps to prevent runaway values.

| Source | When | Formula | Clamp |
|--------|------|---------|-------|
| **Rainfall** | Every month | `clip(μ(month) + N(0, 0.15²), 0, 1)` | [0, 1] |
| **Market Prices** | Every month | Mean-reverting random walk with seasonal targets | `[base × 0.5, base × 2.5]` |
| **Yield Noise** | At harvest | `deterministic_yield × (1 + N(0, 0.10²))` | [0, ∞) with ±3σ noise clamp |
| **Demand Shocks** | ~8%/month | One random crop gets ±30-60% price shock | Respects price ceiling |

### How Prices Work

Prices follow a **mean-reverting random walk** — each month's price is anchored to the previous month's, pulled toward the seasonal target:

```
target = base_price × seasonal_multiplier
drift  = 0.3 × (target − P_prev) / target
P_new  = P_prev × (1 + drift + noise)
```

This means price trends emerge (a rising corn market may keep rising), making storage speculation a learnable strategy. 

### Seasonal Price Multipliers

| Period | Months | Price Multiplier |
|--------|--------|-----------------|
| Pre-monsoon | Apr–May | 1.15× (scarce supply) |
| Monsoon | Jun–Sep | 1.00× |
| Winter | Jan–Mar | 0.95× |
| Post-harvest | Oct–Dec | 0.95× |

---

## Reward Signal

Each step's reward is calculated as the **Delta of Net Worth**:

`Reward = Δ(Net Worth) + Penalties`

Where Net Worth is evaluated as:
`Net Worth = Cash + Land Value + Stored Inventory Value + Growing Crop Expected Value − Debt`

- **Cash delta**: Direct revenue minus costs.
- **Asset value delta**: Changes in the value of the land (driven by soil nitrogen `nitrogen × base_land_price`) and growing crops. This implicitly rewards the agent for keeping soil healthy without needing a separate hardcoded "terminal soil bonus".
- **Invalid action penalty**: −50 for attempting unavailable actions.
- **Terminal Profit**: Computed at the end of the game as `Final Net Worth - Initial Net Worth`.

---

## What a Good Policy Learns

A well-trained agent should discover:

1. **Market Coordination** — using the Public Ledger (Action 11) to coordinate planting schedules and avoid market gluts.
2. **Crop rotation** — alternate nitrogen-depleting crops (corn) with legumes (chickpea) to sustain soil and land value.
3. **Seasonal timing** — plant before the monsoon (June) to exploit free water, sell before the post-harvest dip.
4. **Storage speculation** — store harvests during price troughs and sell during spikes.
5. **Financial prudence** — avoid loans unless the expected return exceeds interest costs; repay before interest compounds.

---

## Difficulty Tiers

The environment supports three difficulty presets:

| Parameter | Easy | Medium | Hard |
|-----------|------|--------|------|
| Starting cash | ₹15,000 | ₹10,000 | ₹7,000 |
| Interest rate | 0% | 8% | 12% |
| Weather noise | Standard | Standard | Standard |
| Max steps | 60 | 60 | 60 |