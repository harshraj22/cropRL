
---

### 1. The Observation Space (State)
The monthly dashboard the PyTorch agent receives.

**Time & Weather**
* `Current_Month` (Integer: 1-12)
* `Expected_Rainfall` (Continuous: 0.0 to 1.0, normalized)

**Biological & Soil State**
* `Active_Crop_Type` (Categorical: 0 = Fallow, 1 = Heavy Feeder/Corn, 2 = Medium Feeder/Wheat, 3 = Legume/Chickpea)
* `Crop_Age_Months` (Integer: 0 to 6)
* `Expected_Yield_Potential` (Continuous: 0.0 to 1.0)
* `Soil_Nitrogen` (Continuous: 0.0 to 1.0)
* *[Future Scope]* `Machinery_Health` (Continuous: 0.0 to 1.0)

**Financial Market & Explicit Costs**
* `Cash_Balance` (Continuous)
* `Current_Debt` (Continuous)
* `Current_Interest_Rate` (Continuous) 
* `Market_Price_Crop_1`, `Market_Price_Crop_2`, `Market_Price_Crop_3` (Continuous)
* `Cost_Seed_1`, `Cost_Seed_2`, `Cost_Seed_3`, `Cost_Irrigate`, `Cost_Fertilize` (Constants) [future scope to update it monthly]
* *[Future Scope]* `Cost_Storage_Monthly` (Constant), `Cost_Machinery_Repair` (Constant)

**Inventory State (Unified Single Slot)**
* `Stored_Crop_Type` (Categorical: 0, 1, 2, or 3)
* `Stored_Amount` (Continuous: Tons)
* `Stored_Age_Months` (Integer: Months since harvest)

[FutureScope] Action mask to allow the agents take valid actions

---

### 2. The Action Space (Discrete: Size 11)
Expanded to accommodate the three crop types.

* **`0: Wait`**: Do nothing.
* **`1: Plant_Crop_1 (Heavy)`**: High seed cost, high profit, destroys soil nitrogen.
* **`2: Plant_Crop_2 (Medium)`**: Moderate seed cost, moderate profit, mild nitrogen drain.
* **`3: Plant_Crop_3 (Legume)`**: Low seed cost, lower profit, restores soil nitrogen.
* **`4: Irrigate`**: Spends `Cost_Irrigate`, mitigates drought penalty.
* **`5: Fertilize`**: Spends `Cost_Fertilize`, boosts `Soil_Nitrogen`.
* **`6: Harvest_and_Store`**: Calculates yield. Auto-sells existing inventory, overwrites with new yield, resets land.
* **`7: Harvest_and_Sell`**: Calculates yield, immediately converts to cash at spot price, resets land.
* **`8: Sell_Inventory`**: Empties the storage slot for cash at current market price.
* **`9: Take_Loan`**: Injects fixed cash chunk, adds to `Current_Debt`.
* **`10: Repay_Loan`**: Uses cash to pay down `Current_Debt`.
* *[Future Scope]* `11: Repair_Machinery`.

---

### 3. Environment Dynamics (The Physics & Economics Engine)

This is the core logic calculated inside the `step()` function.

#### A. Weather Generation Engine
Rainfall isn't purely random; it is seasonal with stochastic noise.
* **The Math:** Let $m$ be the current month. The expected rainfall $W_t$ is calculated using a seasonal baseline $\mu(m)$ plus Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
    $$W_t = \max(0, \min(1, \mu(m) + \epsilon))$$
* **Logic:**
    * If $m \in \{6, 7, 8, 9\}$ (Monsoon): $\mu(m) = 0.8$ (Heavy rain likely).
    * If $m \in \{10, 11, 12, 1\}$ (Winter): $\mu(m) = 0.2$ (Dry, but stable).
    * If $m \in \{4, 5\}$ (Summer): $\mu(m) = 0.05$ (Extreme dry heat).

#### B. Dynamic Interest Rate Calculation
The bank reacts to both the calendar (farmer demand) and the sky (default risk).
* **The Math:** The current interest rate $R_{interest}$ is updated every month:
    $$R_{interest} = R_{base} + \Delta_{liquidity}(m) + \Delta_{risk}(W_{deficit})$$
* **Logic:**
    * $R_{base}$: Standard bank rate (e.g., $0.08$ or $8\%$).
    * $\Delta_{liquidity}$: If $m \in \{6, 7\}$ (Planting season), add $0.03$ because everyone wants cash. If $m \in \{10, 11\}$ (Harvest), subtract $0.02$.
    * $\Delta_{risk}$: If the generated $W_t$ is significantly lower than the `Optimal_Rainfall` for the active crop, lenders panic. $W_{deficit} = \text{Optimal} - W_t$. If $W_{deficit} > 0.3$, add $0.05$ to the interest rate to price in the drought risk.

#### C. The Spoilage & Overwrite Rules
* Every month, `Stored_Age_Months` increases by 1. If it exceeds 6, the crop rots (`Stored_Amount = 0`).
* If `Harvest_and_Store` is chosen while the silo is full, the old crop is automatically liquidated at this month's market price, and the newly harvested crop takes its place.

#### D. Trajectory Horizon (Configurable)
* **Hackathon MVP:** Episode terminates at $T = 60$ steps (5 years).
* **Future Scope:** Increase to $T = 120$ or $T = 240$ (10–20 years). Longer trajectories strictly force the agent to learn the value of the Legume crop, as synthetic fertilizer costs will eventually outpace profits if soil is continually abused over a decade.

---

### 4. The Reward Function

The reward calculates the financial delta, heavily penalizes impossible actions, and provides a terminal payout.

* **Monthly Cash Flow:**
    $$R_t = \text{Cash}_t - \text{Cash}_{t-1}$$
* **Rule Enforcement Penalties:** $-50$ for invalid actions (e.g., planting on occupied land, harvesting empty land).
* **Bankruptcy Termination:** $-1000$ and `done = True` if $\text{Cash} < 0$ and borrowing limit is reached.
* **Terminal Liquidation (Step 60 ONLY):**
    $$R_{final} = (\text{Soil\_Nitrogen} \times 10,000) + (\text{Active\_Crop\_Value}) + (\text{Stored\_Crop\_Value})$$
* *[Future Scope Penalties]*:
    * Subtract a fixed $C_{storage} \times \text{Stored\_Amount}$ every step.
    * Subtract $C_{depreciation}$ every time a physical action (Plant, Harvest, Irrigate) is taken, requiring the agent to eventually spend cash to repair.

---
