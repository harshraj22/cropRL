"""
Multi-Agent WebSocket Hub for CropRL.

Exposes two endpoints:

  WebSocket  /ws/{session_id}/{agent_id}
      Each agent opens its own persistent socket.  Actions are forwarded
      to the shared MultiAgentCroprlEnvironment for that session.
      When the month advances (all agents done), a MONTH_ADVANCED event
      is broadcast to every connected socket in the session.

  HTTP POST  /multi/reset
      Create or reset a multi-agent session.
      Body: { "session_id": str, "num_agents": int, "seed": int (opt) }

  HTTP POST  /multi/step
      Fallback HTTP endpoint (e.g. for agents without WebSocket support).
      Body: {
          "session_id": str,
          "agent_id":   int,
          "action_id":  int,
          "forum_message": str | null
      }

  HTTP GET   /multi/result/{session_id}
      Compute and return MultiAgentResult for a finished session.

  HTTP GET   /multi/sessions
      List active session IDs.

Usage:
    uvicorn cropRL.server.multi_agent_app:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fastapi is required for the multi-agent server. "
        "Install it with: uv sync"
    ) from exc

from cropRL.config import EnvConfig, MultiAgentConfig
from cropRL.models import MultiAgentAction
from cropRL.multi_agent_environment import MultiAgentCroprlEnvironment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session Manager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _Session:
    """Holds one MultiAgentCroprlEnvironment + all connected WebSockets."""

    def __init__(self, env: MultiAgentCroprlEnvironment) -> None:
        self.env = env
        # agent_id → active WebSocket (or None for HTTP-only agents)
        self.sockets: Dict[int, Optional[WebSocket]] = {}
        self.trajectories: Dict[int, list] = {}

    def register_socket(self, agent_id: int, ws: WebSocket) -> None:
        self.sockets[agent_id] = ws

    def unregister_socket(self, agent_id: int) -> None:
        self.sockets[agent_id] = None

    async def broadcast(self, payload: dict, exclude: Optional[int] = None) -> None:
        """Send a JSON message to every connected socket except *exclude*."""
        data = json.dumps(payload)
        for aid, ws in self.sockets.items():
            if ws is not None and aid != exclude:
                try:
                    await ws.send_text(data)
                except Exception:
                    pass  # socket already closed

    async def send_to(self, agent_id: int, payload: dict) -> None:
        """Send a JSON message to a specific agent's socket."""
        ws = self.sockets.get(agent_id)
        if ws is not None:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                pass


class SessionManager:
    """Thread-safe (asyncio-level) registry of multi-agent sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _Session] = {}

    def create(
        self,
        session_id: str,
        env_config: Optional[EnvConfig] = None,
        ma_config: Optional[MultiAgentConfig] = None,
        seed: Optional[int] = None,
        task_id: str = "medium_4agent",
    ) -> _Session:
        env = MultiAgentCroprlEnvironment(
            env_config=env_config,
            ma_config=ma_config,
            task_id=task_id,
        )
        env.reset(seed=seed)
        session = _Session(env)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[_Session]:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> _Session:
        session = self.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return session

    def list_ids(self) -> List[str]:
        return list(self._sessions.keys())

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Global session registry
_manager = SessionManager()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="CropRL Multi-Agent Server",
    description=(
        "WebSocket hub for N agents interacting with a shared "
        "MultiAgentCroprlEnvironment."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── HTTP Schemas ──────────────────────────────────────────────

class ResetRequest(BaseModel):
    session_id: str = ""          # empty → auto-generate
    num_agents: int = 4
    seed: Optional[int] = None
    task_id: str = "medium"
    objective_mode: str = "competitive"
    action_slots_per_month: int = 4
    forum_messages_per_month: int = 2
    enable_hype_crops: bool = True


class StepRequest(BaseModel):
    session_id: str
    agent_id: int
    action_id: int
    forum_message: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    num_agents: int
    message: str


# ── HTTP Endpoints ────────────────────────────────────────────

@app.post("/multi/reset", response_model=ResetResponse)
async def reset_session(req: ResetRequest) -> ResetResponse:
    """Create or reset a multi-agent episode."""
    session_id = req.session_id or str(uuid4())

    ma_config = MultiAgentConfig(
        num_agents=req.num_agents,
        objective_mode=req.objective_mode,
        action_slots_per_month=req.action_slots_per_month,
        forum_messages_per_month=req.forum_messages_per_month,
        enable_hype_crops=req.enable_hype_crops,
    )

    # Adjust base EnvConfig for task difficulty
    from cropRL.tasks import TASKS
    base_task = req.task_id if req.task_id in TASKS else "medium"
    overrides = TASKS[base_task]["config_overrides"].copy()
    env_config = EnvConfig(**overrides)

    _manager.create(
        session_id=session_id,
        env_config=env_config,
        ma_config=ma_config,
        seed=req.seed,
        task_id=f"{base_task}_{req.num_agents}agent",
    )

    return ResetResponse(
        session_id=session_id,
        num_agents=req.num_agents,
        message=f"Session '{session_id}' created with {req.num_agents} agents.",
    )


@app.post("/multi/step")
async def http_step(req: StepRequest) -> Dict[str, Any]:
    """
    Synchronous HTTP fallback for agents without WebSocket support.
    Returns the agent's observation as a JSON dict.
    """
    session = _manager.require(req.session_id)

    action = MultiAgentAction(
        action_id=req.action_id,
        agent_id=req.agent_id,
        forum_message=req.forum_message,
    )

    obs = session.env.step(action)

    # Track trajectory for grading
    if req.agent_id not in session.trajectories:
        session.trajectories[req.agent_id] = []
    session.trajectories[req.agent_id].append({
        "prices": [
            obs.market_price_crop_1,
            obs.market_price_crop_2,
            obs.market_price_crop_3,
        ]
    })

    obs_dict = obs.model_dump()

    # After HTTP step, broadcast month-advance event if applicable
    if obs.done or "Month advanced" in obs.message:
        await session.broadcast({
            "event": "MONTH_ADVANCED",
            "session_id": req.session_id,
            "month": obs.current_month,
        })

    return {"observation": obs_dict, "done": obs.done}


@app.get("/multi/result/{session_id}")
async def get_result(session_id: str) -> Dict[str, Any]:
    """Compute and return the MultiAgentResult for a finished session."""
    session = _manager.require(session_id)
    result = session.env.compute_result(session.trajectories)
    return result.model_dump()


@app.get("/multi/sessions")
async def list_sessions() -> Dict[str, Any]:
    """List all active session IDs."""
    return {"sessions": _manager.list_ids()}


# ── WebSocket Endpoint ────────────────────────────────────────

@app.websocket("/ws/{session_id}/{agent_id}")
async def websocket_agent(
    websocket: WebSocket,
    session_id: str,
    agent_id: int,
) -> None:
    """
    Persistent WebSocket connection for one agent in a multi-agent session.

    Messages from client (JSON):
        { "action_id": int, "forum_message": str | null }

    Messages to client (JSON):
        {
          "event": "OBSERVATION",
          "observation": { ... MultiAgentObservation fields ... },
          "done": bool
        }
        OR
        {
          "event": "MONTH_ADVANCED",
          "month": int
        }
        OR
        {
          "event": "ERROR",
          "detail": str
        }
    """
    session = _manager.get(session_id)
    if session is None:
        await websocket.close(code=4004, reason=f"Session '{session_id}' not found.")
        return

    n = session.env._ma_cfg.num_agents
    if agent_id < 0 or agent_id >= n:
        await websocket.close(
            code=4001, reason=f"agent_id {agent_id} out of range 0..{n-1}."
        )
        return

    await websocket.accept()
    session.register_socket(agent_id, websocket)

    try:
        # Send initial observation on connect
        obs = session.env.get_obs(agent_id)
        await websocket.send_text(json.dumps({
            "event": "OBSERVATION",
            "observation": obs.model_dump(),
            "done": False,
        }))

        # Main receive loop
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "event": "ERROR",
                    "detail": "Invalid JSON.",
                }))
                continue

            action_id = data.get("action_id")
            if action_id is None:
                await websocket.send_text(json.dumps({
                    "event": "ERROR",
                    "detail": "Missing 'action_id' field.",
                }))
                continue

            action = MultiAgentAction(
                action_id=int(action_id),
                agent_id=agent_id,
                forum_message=data.get("forum_message"),
            )

            obs = session.env.step(action)

            # Track trajectory
            if agent_id not in session.trajectories:
                session.trajectories[agent_id] = []
            session.trajectories[agent_id].append({
                "prices": [
                    obs.market_price_crop_1,
                    obs.market_price_crop_2,
                    obs.market_price_crop_3,
                ]
            })

            # Send observation to this agent
            await websocket.send_text(json.dumps({
                "event": "OBSERVATION",
                "observation": obs.model_dump(),
                "done": obs.done,
            }))

            # Broadcast month-advance signal to all other agents
            if "Month advanced" in (obs.message or ""):
                await session.broadcast(
                    {
                        "event": "MONTH_ADVANCED",
                        "session_id": session_id,
                        "month": obs.current_month,
                    },
                    exclude=agent_id,
                )

            if obs.done:
                # Compute and broadcast final result
                result = session.env.compute_result(session.trajectories)
                await session.broadcast({
                    "event": "EPISODE_DONE",
                    "session_id": session_id,
                    "result": result.model_dump(),
                })
                break

    except WebSocketDisconnect:
        pass
    finally:
        session.unregister_socket(agent_id)


def main(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
