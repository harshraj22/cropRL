# CropRL: Farm Management RL Environment

A farm management reinforcement learning environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An AI agent manages a small Indian farm over 60 months (5 years), balancing crop rotation, soil health, financial management, and weather risk.

## Quick Start

```bash
# Install dependencies
pip install openenv-core numpy pydantic fastapi uvicorn

# Run locally
python -c "
from crop_env import create_env_for_task, CropAction

env = create_env_for_task('medium')
obs = env.reset(seed=42)
print(f'Cash: {obs.cash_balance}, Soil: {obs.soil_nitrogen}')

# Plant corn
obs = env.step(CropAction(action_id=1))
print(f'Planted! Cash: {obs.cash_balance}')
"
```

## Environment Details

### Observation Space
Each month the agent receives:
- **Time & Weather**: Month (1-12), rainfall (0-1)
- **Farm**: Crop type, age, yield potential, soil nitrogen
- **Finance**: Cash, debt, interest rate, market prices (3 crops)
- **Storage**: Stored crop type, amount, age

### Action Space (Discrete: 11)
| ID | Action | Description |
|----|--------|-------------|
| 0 | Wait | Do nothing |
| 1-3 | Plant | Corn/Wheat/Chickpea |
| 4 | Irrigate | Mitigate drought |
| 5 | Fertilize | Boost soil nitrogen |
| 6 | Harvest & Store | Store crop for later sale |
| 7 | Harvest & Sell | Sell at market price |
| 8 | Sell Inventory | Sell stored crops |
| 9 | Take Loan | Borrow ₹5,000 |
| 10 | Repay Loan | Pay off full debt |

### Tasks
Three difficulty levels, same objective (maximize net worth):
- **Easy**: No interest, stable weather, generous cash
- **Medium**: Standard conditions
- **Hard**: High interest, volatile weather, poor soil

## Running Inference

```bash
# OpenAI-compatible API
API_BASE_URL="..." MODEL_NAME="..." HF_TOKEN="..." python inference.py

# Ollama (local)
python inference_ollama.py --model llama3.2 --task medium
```

## Running Tests

```bash
pytest tests/ -v
```
