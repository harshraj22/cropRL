1. Weather and Climate Parameters
These variables control the environmental conditions that affect crop growth and irrigation needs.

| Variable | Source | Description |
| :--- | :--- | :--- |
| `current_month` | Observation | The current calendar month (1-12). |
| `expected_rainfall` | Observation | Forecasted rainfall for the current month, normalized between **0.0** and **1.0**. |
| `optimal_water_level` | Config | The ideal water level for each crop type (Corn: 0.6, Wheat: 0.4, Chickpea: 0.3). |
| `weather_sigma` | Config | Gaussian noise (**σ**) applied to seasonal rainfall baselines (default: 0.15). |
| `weather_seasonal_baselines` | Config | Mean rainfall per season: Monsoon (0.8), Winter (0.2), Spring (0.4), Summer (0.05). |
| `weather_sigma_realisation` | Config | Sigma for the noise to be added to the realisation of the rainfall. |
| `water_utilised` | Config | For each crop type, this is the amount of water utilised by the crop per month. |
| `irrigate_amount` | Config | For each crop type, this is the amount of water added to the field when the `irrigate` action is taken. |
| `current_water_level` | Observation | Maintains the level of water in the field currently which is the function of `realised_rainfall`, `irrigate`action and `water_utilised`. |


NOTES:
1. `current_month`:
    * Only when the agent takes a `wait` action, only then increment the `current_month` by 1.
    * As per the user's note, current_month (and by extension crop ageing and seasonal updates) will ONLY advance when the agent chooses the wait (ID 0) action. Other actions (Plant, Irrigate, etc.) will occur "instantly" within the current month.
2. `expected_rainfall`:
    * This is calculated from `weather_seasonal_baselines` and `weather_sigma`.
    * this is provided to the agent as observation stream.
    * Once the agent executes the action and the `current_month` is incremented, then actual realisation of the rainfall should also happen.
    * For actual realisation, `expected_rainfall` should be used as mean and `weather_sigma_realisation` should be used as standard deviation. This should be kept small as we want the actual realised rainfall to be close to the `expected_rainfall`.

3. `current_water_level`:
    * This is the amount of water in the field currently.
    * `current_water_level` = `realised_rainfall` + `irrigate_amount` - `water_utilised`
    * If there is extra water than the `optimal_water_level`, assume that the farm has good drainage and the extra water will be drained. So the water level can never be greater than the `optimal_water_level`.
    * The current water level should be sent as observation stream to the agent in the next step.
    * If the `irrigate` action is performed, immediately increment the water level and do not wait for the `current_month` to increment.

---

## 2. Soil & Bio-Dynamics
These variables track the health of the land and the progress of the active crop.

| Variable | Source | Description |
| :--- | :--- | :--- |
| `soil_nitrogen` | Observation | Current soil fertility level (**0.0 to 1.0**). Affects yield directly. |
| `nitrogen_impact` | Config | Monthly nitrogen change per crop: Corn (-0.25), Wheat (-0.10), Chickpea (+0.15). |
| `natural_nitrogen_recovery` | Config | Passive soil nitrogen recovery rate per month (+0.01). |
| `fertilize_nitrogen_boost` | Config | Instant nitrogen increase when the **Fertilize** action is taken (+0.15). |
| `active_crop_type` | Observation | 0: Fallow, 1: Corn (Heavy Feeder), 2: Wheat (Medium), 3: Chickpea (Legume). |
| `crop_age_months` | Observation | Months elapsed since planting. |
| `growth_months` | Config | Months required for maturity: Corn (4), Wheat (3), Chickpea (3). |
| `expected_yield_potential` | Observation | Normalized estimate of harvest yield (**0.0 to 1.0**). Logic: `raw_yield / max_possible_yield`. |
| `base_yield_tons` | Config | Theoretical max yield in tons: Corn (8.0), Wheat (5.0), Chickpea (3.0). |
| `yield_sigma` | Config | Stochastic noise applied at the moment of harvest (default: 0.10). |
| `minimum_nitrogen_requirement` | Config | Minimum nitrogen required for each crop type. |
| `optimal_seasons_per_crop` | Config | For each crop, there is an optimal set of seasons in which it can be grown. If the crop is grown in a season which is not optimal for it, then the yield should be low. Note that there can be multiple optimal seasons for each crop. |

1. `nitrogen_impact`:
    * The soil nitrogen increment or decrement should happen every month rather than happening once during the harvest.
    * The values of `nitrogen_impact` should be chosen carefully such that changes happen very slowly. Currently, for example, for corn the `nitrogen_impact` is -0.25 => if I initially begin with soil nitrogen = 1.0, then after 4 months (maturity time for corn), the soil nitrogen will become 0.0. This is a very drastic change and should be avoided.
    * The soil nitrogen generally depletes very slowly. Like for corn it should be reach close to 0 after growing constantly for 3-4 times where each time it grows for 4 months.
    * Similarly, for wheat it should be reach close to 0 after growing constantly for 5-6 times where each time it grows for 3 months.
    * Only when the `current_month` is incremented, then the `soil_nitrogen` should be updated.

2. `fertilize_nitrogen_boost`:
    * This should immediately increase the soil_nitrogen level and it should not wait for the `current_month` to increase.

3. `expected_yield_potential`:
    * Normalized estimate of harvest yield (0.0 to 1.0). Logic: raw_yield / max_possible_yield
    * The `raw_yield` is a function of the following:
        * `crop_age_month` and `growth_months`:
            * 0–1 Month: The yield is always 0 in the first month because crop_age starts at 0.
            * Growth Phase: The yield remains low early on and increases rapidly as it approaches maturity.
            * Peak: The yield is maximized exactly at growth_months.
            * Rotting: Once maturity is reached, the crop rots very quickly—it will reach 0.0 yield just 2 months after its peak maturity.
        * `soil_nitrogen` and `minimum_nitrogen_requirement`:
            * If `soil_nitrogen` is below the minimum required level, yield should be low.
            * If `soil_nitrogen` is above the minimum required level, yield should be high and plateau after some time.
        * `current_water_level` and `optimal_water_level`:
            * If current_water_level is less than the optimal required water level, then the yield should be low.
        * `optimal_seasons_per_crop` and `current_season`:
            * if the crop is grown in a non-optimal season then the yield should be low.
            * Note that the current season should be a function of current_month.
    * Like rainfall, the actual realised yield can be slightly noisy from the `expected_yield_potential`. The actual realised yield should only be calculated if the agent takes the `harvest` action.
    * The conversion of `realised_yield` (a number between [0, 1]) to money happens as follows:
        * We get the raw_realised_yield = `realised_yield` * `base_yield_tons`
        * Then we get the actual money = `raw_realised_yield` * `market_price`

---

## 3. Market & Economics
These variables define the financial environment, including price volatility and lending.

| Variable | Source | Description |
| :--- | :--- | :--- |
| `market_price_crop_[1-3]` | Observation | Spot price (₹/ton) for Corn, Wheat, and Chickpea. |
| `base_market_prices` | Config | Baseline prices: Corn (1200), Wheat (800), Chickpea (500). |
| `market_seasonal_multipliers` | Config | Price fluctuations by season: Post-harvest (0.85), Pre-monsoon (1.15), Monsoon (1.0), Winter (0.95). |
| `market_price_sigma` | Config | Volatility noise applied to market prices (default: 0.10). |
| `price_reversion_speed` | Config | Speed at which prices return to seasonal targets in autocorrelation mode (0.3). |
| `demand_shock_probability` | Config | Probability (0.08) of a rare demand shock event (price spike/dip of 30-60%). |
| `current_interest_rate` | Observation | Calculated annual interest rate. Includes liquidity premiums (planting/harvest seasons) and drought risk premiums. |
| `base_interest_rate` | Config | Standard annual interest rate for loans (default: 0.08 / 8%). |
| `base_land_price` | Config | The maximum price of the land which can be achieved if sold. |
| `monthly_fixed_cost` | Config | Monthly fixed cost that the agent has to bear |

1. The spot price of a crop should not go below 0.5 * the base price of the crop. Currently, the lower bound is ₹1.0. This is too low.
2. The `current_land_price` = `base_land_price` * `soil_nitrogen`.
3.. Add another factor called `inflation`:
    * Inflation should be a factor that increases the price of everything every year. This means the following:
        * The `base_market_prices` for each crop increases by the inflation factor.
        * The cost of seed for each crop increases by the inflation factor.
        * The cost of irrigation increases by the inflation factor.
        * The cost of fertilizer increases by the inflation factor.
        * The fixed amount that the agent gets when takes the loan should also increase by the inflation factor.
        * The `base_land_price` should also be increased by the inflation factor.
        * `monthly_fixed_cost` should increase by the inflation factor.
    * Inflation should be applied at the begining of every year.
4. Debt uses the interest rate at which the loan was taken and not the current interest rate.
5. New loan should not be given if existing loan is not paid off.

---
## 4. Farm Operations & Logistics
These variables represent the costs and inventory management aspects of the farm.

| Variable | Source | Description |
| :--- | :--- | :--- |
| `cost_seed_[1-3]` | Observation | Cost to plant: Corn (800), Wheat (500), Chickpea (200). |
| `cost_irrigate` | Observation | Cost to apply irrigation for the current month (₹300). |
| `cost_fertilize` | Observation | Cost to apply fertilizer for the current month (₹400). |
| `stored_crop_type` | Observation | Type of crop currently in storage (0 if empty). |
| `stored_amount` | Observation | Quantity of crop in storage in tons. |
| `stored_age_months` | Observation | Months since the stored crop was harvested. |
| `max_storage_age` | Config | Shelf life before the stored crop spoils and value drops to zero (default: 6 months). |

---

## 5. Task & Episode Control
Internal bookkeeping and performance tracking variables.

| Variable | Source | Description |
| :--- | :--- | :--- |
| `current_step` | Observation | Current month index in the 60-month episode (0-59). |
| `cash_balance` | Observation | Liquid cash available for operations. |
| `current_debt` | Observation | Total outstanding loan principal. |
| `loan_chunk` | Config | Single loan amount provided when taking the **Take Loan** action (₹5000). |
| `max_debt` | Config | Ceiling for total allowable debt (₹20,000). |
| `done` | Observation | Boolean flag indicating if the episode has ended (max steps or bankruptcy). |
| `reward` | Observation | The incremental reward/penalty received from the last action. |
| `invalid_action_penalty` | Config | Penalty applied for attempting illegal actions (e.g., planting on occupied land) (-50). |
| `bankruptcy_penalty` | Config | Heavy penalty applied if cash balance remains negative (-1000). |
| `terminal_soil_bonus_factor` | Config | Bonus applied at the end of the episode based on final soil nitrogen (factor: 10,000). |

1. `max_steps` is the maximum allowed length of the episode.This can be set very high since we have separated `current_month` from `current_step`, i.e., only when the agents calls the `wait` action then we increment the `current_month`. However, any action taken increases the `current_step`.
2. Remove `max_debt` since the agent can take only one active loan at a time. Until it's paid off it can't take another loan.
3. Remove `terminal_soil_bonus_factor` as it is equiavlent to the `base_land_price`.
4. Terminate the episode on the following conditions:
    * `current_step` is greater than `max_steps`.
    * bankruptcy has happened
    * if more than 60 months have elaspsed since the begining of the episode.
4. The overall objective of the environment is to maximise profit after episode terminates. The profit is calculated as follows:
    * It is the difference between the final value vs initial value.
    * Final Value is the sum of the cash balance, current_land_price, the value of the stored crop minus the outstanding debt. If there is any crop on the land which is not harvested it should on termination, it can be harvested and sold at the current market price and the revenue from selling this should be added to the final value.
    * Initial value is the sum of the cash_balance that the agent starts with and the land_price at the beginning.
5. This overall objective is a sparse reward. Break it down into meaningful and constructive intermediate rewards for the agent so that it can learn using RL.
