"""Data models for conversation generation."""

from __future__ import annotations

from pydantic import BaseModel


class PersonaResponse(BaseModel):
    """Response model for persona utterance."""

    utterance: str


class PersonaParams(BaseModel):
    """Parameters for persona behavior."""

    profile: str
    base_prob: float
    max_prob: float
    decay: float
    recovery_step: float
