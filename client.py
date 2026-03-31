"""
LLM Inference Scheduling Environment — OpenEnv Client.

Provides the `LLMInferenceEnv` client class for connecting to and interacting
with the LLM Inference Scheduling Environment server via WebSocket.

Example (async):
    >>> async with LLMInferenceEnv(base_url="http://localhost:8000") as env:
    ...     result = await env.reset()
    ...     while not result.done:
    ...         action = SchedulingAction(request_index=0, replica_index=0)
    ...         result = await env.step(action)

Example (sync):
    >>> with LLMInferenceEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset()
    ...     result = env.step(SchedulingAction(request_index=0, replica_index=0))
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult, StateT

    _HAS_OPENENV = True
except ImportError:
    # Standalone mode
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    ObsT = TypeVar("ObsT")
    StateT = TypeVar("StateT")

    @dataclass
    class StepResult(Generic[ObsT]):
        observation: ObsT
        reward: Optional[float] = None
        done: bool = False

    _HAS_OPENENV = False

from .models import SchedulingAction, SchedulingObservation


if _HAS_OPENENV:
    from openenv.core.env_server.types import State

    class LLMInferenceEnv(EnvClient):
        """
        OpenEnv client for the LLM Inference Scheduling Environment.

        Connects to a running environment server via WebSocket and provides
        methods for reset, step, and state queries.

        Example:
            >>> with LLMInferenceEnv(base_url="http://localhost:8000").sync() as env:
            ...     result = env.reset()
            ...     print(result.observation)
            ...     action = SchedulingAction(request_index=0, replica_index=0)
            ...     result = env.step(action)
        """

        def _step_payload(self, action: SchedulingAction) -> Dict[str, Any]:
            """Convert a SchedulingAction to the JSON payload for the server."""
            if isinstance(action, SchedulingAction):
                return action.model_dump(exclude={"metadata"})
            elif isinstance(action, dict):
                return action
            else:
                return {"metadata": {}}

        def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SchedulingObservation]:
            """Parse server response into StepResult with SchedulingObservation."""
            obs_data = payload.get("observation", {})

            # Extract the scheduling observation from metadata if nested
            if isinstance(obs_data, dict) and "metadata" in obs_data:
                inner = obs_data.get("metadata", {}).get("observation", {})
                if inner:
                    obs = SchedulingObservation(**inner)
                else:
                    obs = SchedulingObservation(**obs_data)
            else:
                obs = SchedulingObservation(**obs_data)

            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: Dict[str, Any]) -> State:
            """Parse server state response."""
            return State(**payload)

        @property
        def last_action_mask(self) -> Optional[Dict[str, list]]:
            """Get the action mask from the last observation (if stored)."""
            return None  # Retrieved via observation

else:
    # Standalone client (no OpenEnv installed)
    class LLMInferenceEnv:  # type: ignore[no-redef]
        """
        Standalone client for the LLM Inference Scheduling Environment.

        Uses the environment directly without a server, for local testing.
        """

        def __init__(self, config=None):
            from .server.llm_inference_environment import LLMInferenceEnvironment
            self._env = LLMInferenceEnvironment(config=config)

        def reset(self, **kwargs) -> StepResult:
            obs = self._env.reset(**kwargs)
            inner = obs.metadata.get("observation", {})
            scheduling_obs = SchedulingObservation(**inner) if inner else SchedulingObservation()
            return StepResult(
                observation=scheduling_obs,
                reward=obs.reward,
                done=obs.done,
            )

        def step(self, action: SchedulingAction) -> StepResult:
            obs = self._env._step_impl(action)
            inner = obs.metadata.get("observation", {})
            scheduling_obs = SchedulingObservation(**inner) if inner else SchedulingObservation()
            return StepResult(
                observation=scheduling_obs,
                reward=obs.reward,
                done=obs.done,
            )

        @property
        def state(self):
            return self._env.state

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass
