# CropRL REST API

Local HTTP interface to the CropRL farming environment. **Not required for the hackathon submission** — this is for local testing and experimentation.

## Quick Start

```bash
# From project root
uvicorn extras.api.app:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | List available task difficulties |
| `GET` | `/actions` | List all action IDs and names |
| `POST` | `/reset` | Start a new episode |
| `GET` | `/state` | Get current observation |
| `POST` | `/step` | Take an action |

## Example Usage (curl)

```bash
# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'

# Check state
curl http://localhost:8000/state

# Plant chickpea (action 3)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_id": 3}'

# Wait (action 0)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_id": 0}'

# Harvest & sell (action 7)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_id": 7}'
```

## Example Usage (Python)

```python
import requests

BASE = "http://localhost:8000"

# Reset
r = requests.post(f"{BASE}/reset", json={"task": "easy", "seed": 42})
obs = r.json()["observation"]

# Play a few steps
for action in [3, 0, 7, 1, 0, 0, 0, 7]:
    r = requests.post(f"{BASE}/step", json={"action_id": action})
    data = r.json()
    print(f"Step {data['step']}: {data['action_name']} → R={data['reward']:.0f}, Cash=₹{data['observation']['cash_balance']:,.0f}")
```
