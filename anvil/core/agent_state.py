from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AgentTurn(BaseModel):
    agent_id: str
    session_id: UUID = Field(default_factory=uuid4)
    input_schema: dict[str, Any]
    context_token_count: int = 0
    kv_cache_handle: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentOutput(BaseModel):
    agent_id: str
    session_id: UUID
    output_schema: dict[str, Any]
    tokens_generated: int = 0
    kv_cache_handle: str
    next_agent: str | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None


class AgentDefinition(BaseModel):
    agent_id: str
    name: str
    description: str = ""
    system_prompt: str = ""
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    tools: list[str] = []
    max_turns: int = 5
    timeout_s: int = 120


class OrchestratorState(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    agents: dict[str, AgentDefinition] = {}
    active_agent: str | None = None
    queue: list[str] = []
    history: list[AgentOutput] = []
    any_error: bool = False
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {"extra": "forbid"}
