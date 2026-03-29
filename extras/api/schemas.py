"""Pydantic schemas for the CropRL REST API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────────────────


class ResetRequest(BaseModel):
    task: str = Field("medium", description="Task difficulty: easy, medium, hard")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    text_mode: bool = Field(True, description="Include text summaries in observations")


class ActionRequest(BaseModel):
    action_id: int = Field(..., ge=0, le=10, description="Action ID (0-10)")


# ── Response Models ────────────────────────────────────────────


class TaskInfo(BaseModel):
    task_id: str
    description: str
    max_steps: int


class ResetResponse(BaseModel):
    message: str
    task_id: str
    observation: dict


class StateResponse(BaseModel):
    step: int
    done: bool
    observation: dict


class ActionResponse(BaseModel):
    step: int
    action_id: int
    action_name: str
    reward: float
    done: bool
    message: str
    observation: dict
