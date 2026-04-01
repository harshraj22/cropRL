# CropRL: Future Ideas & Enhancements

Ideas captured during implementation planning. These are not in scope for the MVP but the codebase is designed to accommodate them.

---

## Task Design Ideas

### Grader-Based Scoring (Post-MVP)
- **Grader as separate script** that accepts a trajectory JSON and returns 0.0–1.0
- **Grader as reward function** baked into the environment
- Need to decide which pattern fits OpenEnv hackathon expectations better

### Task Variations (Beyond Easy/Medium/Hard Env Complexity)
- **Survival Task**: Score = 1.0 if survived all 60 steps, else steps_survived/60
- **Profit Task**: Score = clip(net_worth / target, 0, 1)
- **Sustainability Task**: Composite of profit (50%) + soil health maintenance (50%)
- **Drought Challenge**: High weather variance, agent must learn irrigation timing
- **Debt Trap**: Start with debt, agent must escape negative cash flow spiral

### Task Difficulty Levers
| Lever | Easy | Medium | Hard |
|-------|------|--------|------|
| Interest rate | 0% | 8% | 12% |
| Initial cash | ₹15,000 | ₹10,000 | ₹7,000 |
| Initial soil N | 0.8 | 0.6 | 0.35 |
| Weather variance | σ=0.05 | σ=0.15 | σ=0.25 |
| Market volatility | σ=0.05 | σ=0.10 | σ=0.20 |
| Spoilage age | 12 months | 6 months | 3 months |
| Storage cost | ₹0/month | ₹0/month | ₹50/month |

---

## Market Price Enhancements

### Random Walk Model ✅ IMPLEMENTED
```
P_t = P_{t-1} * (1 + drift + ε)
drift = reversion_speed * (target - P_{t-1}) / target
target = base_price * seasonal_multiplier
ε ~ N(0, σ²), clamped to ±3σ
```
- Prices now autocorrelate month-to-month with mean reversion toward seasonal targets
- Configurable via `enable_price_autocorrelation` and `price_reversion_speed`
- All prices clamped to `[1.0, base × price_max_multiplier]`

### Demand Shocks ✅ IMPLEMENTED
- With probability `demand_shock_probability` (~8% per month ≈ 1/year), one random crop gets a price spike or dip of 30-60%
- Direction is random (positive or negative)
- Creates opportunities for agents that maintain storage/optionality

### Correlated Crop Prices
- When Corn price is high, Wheat tends to be mid, Chickpea low (substitute goods)
- Could model with a correlation matrix on the noise terms

---

## Environment Extensions (Future Scope from RoughPlan)

### Machinery System
- `Machinery_Health`: 0.0–1.0, degrades with each physical action (Plant/Harvest/Irrigate)
- `Repair_Machinery` action (ID 11): costs `cost_machinery_repair`, restores health
- If machinery_health < 0.2: yield penalty (equipment breaking down)
- If machinery_health = 0: cannot perform physical actions

### Storage Costs
- Monthly cost proportional to stored amount
- Creates pressure to sell rather than hoard indefinitely

### Dynamic Costs
- Seed/irrigation/fertilizer costs vary by month (supply chain effects)
- Could follow similar seasonal model as market prices

### Action Masking
- Provide `valid_actions: list[int]` in observation
- Prevents agent from wasting steps on invalid actions
- Important for RL training efficiency (mask logits)

### Multi-Field Farming
- Multiple land plots, each with independent crop/soil state
- Dramatically increases state/action space complexity

---

## Observation Formats

### Text-Friendly (for LLM agents)
```
Month: June (6) | Step: 12/60
Weather: Expected rainfall 0.72 (monsoon)
Farm: Growing Corn (age 3/4 months) | Soil N: 0.45
Finance: Cash ₹8,200 | Debt ₹5,000 | Interest: 11%
Markets: Corn ₹1,150/ton | Wheat ₹820/ton | Chickpea ₹490/ton
Storage: 5.2 tons of Wheat (age 2 months)
```

### Numeric Vector (for RL policies)
```python
[month_sin, month_cos, rainfall, crop_type_onehot(4), crop_age/6, 
 yield_potential, soil_n, cash/50000, debt/20000, interest_rate,
 price_1/2000, price_2/2000, price_3/1000,
 stored_type_onehot(4), stored_amount/10, stored_age/6]
```

---

## Additional Stochastic Features (Future Scope)

### Pest / Disease Events
- Each month, a growing crop has a small probability (~5%) of a pest event
- Damage: lose 20-80% of yield potential (uniform random)
- Creates "insurance" calculus: harvest early to lock in value vs gamble on maturity
- Could be modeled as a `crop_health_multiplier` in state
- Irrigated/fertilized crops could have lower pest probability
- Implementation: one `rng.random()` check per month + one float in state

### Stochastic Spoilage
- Instead of deterministic rot at max_storage_age, spoilage probability increases with age:
  ```
  spoilage_prob = 0 if age <= safe_age else (age - safe_age) / (max_age - safe_age)
  ```
- Removes the "safe to store for exactly N months" certainty
- Makes storage timing a genuine risk-reward tradeoff
- Implementation: change 3 lines in `apply_spoilage()`, zero new state
