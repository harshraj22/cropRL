# CropRL — Quickstart

CropRL is an open-source, multi-agent agricultural management environment. Over a 60-month (5-year) horizon, agents must balance short-term profits against long-term soil health, manage loan debt, store crops for market speculation, and utilize the chat forum to avoid crop gluts.

For complete details on market modeling, soil dynamics, stochastic physics, and task balancing, please see the **[Detailed Documentation in `cropRL/README.md`](./cropRL/README.md)**.

## Core Concepts

- **Goal**: Maximize your final net worth (Cash + Asset Value - Debt) + Terminal Nitrogen Bonus.
- **Action Slots**: Every month, agents take actions up to their slot capacity (default is 4). 
- **Wait / No-Op**: If you have no meaningful action to take, you must pass Action `0` (`WAIT` / `NO_OP`) to consume your slot.
- **Time Advance**: A calendar month only resolves and advances when *all* agents have consumed their capacity of action slots. 
- **Global Market**: Sell actions are deferred. Crop prices resolve using a batch-clearing mechanism at the close of every month based on **global supply dynamics**. Each competing seller of the same crop drops the clearing price for everyone.
- **Hype Crops**: 3 extreme high-risk/high-reward crops (Matcha, Quinoa, Turmeric) follow boom/bust pricing trends heavily linked to "Social Media Hype" indicators. Flooding the market with a hype crop (selling >60% of market capacity) during its peak instantly triggers a price crash.

## Action Space

| ID | Action | Description |
|----|--------|-------------|
| 0 | **Wait / No-Op** | Consumes 1 action slot. Required to end your turn if no other acts are needed. |
| 1–3 | **Plant Corn/Wheat/Chickpea** | Standard crops: Corn (depletes N), Wheat (medium N drain), Chickpea (restores N). |
| 4 | **Irrigate** | Costs cash, boosts current water levels for yields. |
| 5 | **Fertilize** | Costs cash, restores missing soil Nitrogen. |
| 6 | **Harvest & Store** | Harvest mature crop into personal warehouse storage (rots sequentially). |
| 7 | **Harvest & Sell** | Harvest and queue for month-end sale on the global market. |
| 8 | **Sell Inventory** | Queue stored crops for month-end sale on the global market. |
| 9 | **Take Loan** | Instantly acquire ₹5,000 cash; accumulates harsh compounding interest. |
| 10 | **Repay Loan** | Completely pay off the existing principal + interest balance. |
| 11 | **Post Forum Message** | Broadcast a text message readable by all other farmers (e.g. negotiation/threats). |
| 12–14 | **Plant Matcha/Quinoa/Turmeric**| High variance cash-crops meant to play the "hype" market booms. |

## Inference Loop Example

The basic single/multi-agent step loop works by polling the environment's `get_obs(agent_id)` per agent until all agents reach a `done` state. Because actions are gated by slots, doing nothing gracefully consumes slots using `WAIT (0)`.

```python
from cropRL.tasks import create_env_for_task
from cropRL.models import MultiAgentAction

# Start a predefined task (e.g., "easy_4agent", "medium_4agent", "hard_8agent")
env = create_env_for_task("medium_4agent", text_mode=True)
env.reset(seed=42)

n = env._ma_cfg.num_agents

done_agents = set()
total_steps = 0
max_steps = env._env_cfg.max_steps * n

while len(done_agents) < n and total_steps < max_steps:
    for agent_id in range(n):
        obs = env.get_obs(agent_id)
        
        if obs.done:
            done_agents.add(agent_id)
            # Send a 0 action (WAIT) so dead/done agents don't freeze the environment clock
            action = MultiAgentAction(action_id=0, agent_id=agent_id)
            env.step(action)
            continue
            
        # --- Your agent logic goes here ---
        # e.g., action_id, forum_message = policy(obs)
        action_id = 0
        forum_message = None 
        # ----------------------------------
        
        action = MultiAgentAction(action_id=action_id, agent_id=agent_id, forum_message=forum_message)
        new_obs = env.step(action)
        
        total_steps += 1
        
        if new_obs.done:
            done_agents.add(agent_id)

# Evaluate outputs
result = env.compute_result({})
print(f"Winner: Agent {result.winner_agent_id}")
print(f"Final Scores: {result.agent_scores}")
```

> **Note**: An agent who hits a terminal bankruptcy condition (defined strictly as having **Cash < 0 AND an active loan**) before the 60-month horizon ends must still "Wait" (`action_id=0`) through the remaining simulation slots so the active farmers' calendar months keep incrementing!

## Running the LLM Baseline

We provide a fully rigged inference script for benchmarking local models (via Ollama) against CropRL environments.

```bash
# Evaluate Llama 3 on an easy 4-agent environment
python cropRL/inference.py --task easy_4agent --model llama3
```