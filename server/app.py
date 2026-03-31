"""
FastAPI application for the LLM Inference Scheduling Environment.

Creates an HTTP/WebSocket server compatible with the OpenEnv client protocol.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    uv run --project . server
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.types import Action, Observation

    from .llm_inference_environment import LLMInferenceEnvironment

    # Create the app with OpenEnv's create_app factory
    app = create_app(
        LLMInferenceEnvironment,
        Action,
        Observation,
        env_name="llm_inference_env",
    )

except ImportError:
    # Standalone mode without openenv-core: create a minimal FastAPI app
    import json
    from typing import Any, Dict

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel

    from .llm_inference_environment import LLMInferenceEnvironment

    app = FastAPI(
        title="LLM Inference Scheduling Environment",
        description="RL environment for LLM inference request scheduling",
        version="0.1.0",
    )

    # Store environment instances per session
    _environments: Dict[str, LLMInferenceEnvironment] = {}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/")
    async def root():
        return {
            "name": "llm_inference_env",
            "description": "RL environment for LLM inference request scheduling",
            "version": "0.1.0",
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        env = LLMInferenceEnvironment()
        session_id = str(id(env))
        _environments[session_id] = env

        try:
            while True:
                raw = await websocket.receive_text()
                message = json.loads(raw)
                msg_type = message.get("type", "")

                if msg_type == "reset":
                    data = message.get("data", {})
                    obs = env.reset(**data)
                    response = {
                        "type": "observation",
                        "data": {
                            "observation": obs.model_dump(),
                            "reward": obs.reward,
                            "done": obs.done,
                        },
                    }
                    await websocket.send_text(json.dumps(response))

                elif msg_type == "step":
                    data = message.get("data", {})
                    from ..models import SchedulingAction
                    action = SchedulingAction(**{
                        k: v for k, v in data.items()
                        if k in SchedulingAction.model_fields
                    })
                    obs = env._step_impl(action)
                    response = {
                        "type": "observation",
                        "data": {
                            "observation": obs.model_dump(),
                            "reward": obs.reward,
                            "done": obs.done,
                        },
                    }
                    await websocket.send_text(json.dumps(response))

                elif msg_type == "state":
                    state = env.state
                    response = {
                        "type": "state",
                        "data": state.model_dump(),
                    }
                    await websocket.send_text(json.dumps(response))

                elif msg_type == "close":
                    break

                else:
                    response = {
                        "type": "error",
                        "data": {
                            "message": f"Unknown message type: {msg_type}",
                            "code": "UNKNOWN_TYPE",
                        },
                    }
                    await websocket.send_text(json.dumps(response))

        except WebSocketDisconnect:
            pass
        finally:
            _environments.pop(session_id, None)
            await websocket.close()


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
