"""
Base Agent class for AnvilAgent.
All agents inherit from this and implement their specific logic.
"""

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from anvil.core.agent_state import AgentDefinition, AgentOutput, AgentTurn


class ToolDefinition(BaseModel):
    """Definition of a tool an agent can call."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    required: list[str] = []


class AgentContext(BaseModel):
    """Runtime context passed to an agent during execution."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = ""
    turn_number: int = 0
    max_tokens: int = 4096
    temperature: float = 0.1
    tools: list[ToolDefinition] = []


class BaseAnvilAgent(ABC):
    """
    Abstract base class for all AnvilAgent agents.

    Each agent:
    1. Has a Pydantic-defined input/output schema
    2. Uses constrained decoding (GBNF) for structured output
    3. Can execute sandboxed code via Wasmtime
    4. Follows deterministic state transitions
    """

    def __init__(self, agent_id: str, name: str, description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self._tools: list[ToolDefinition] = []

    @abstractmethod
    def get_system_prompt(self) -> str:
        """System prompt that defines agent behavior."""
        ...

    @abstractmethod
    def get_input_schema(self) -> dict:
        """Pydantic-compatible input schema."""
        ...

    @abstractmethod
    def get_output_schema(self) -> dict:
        """Pydantic-compatible output schema."""
        ...

    def get_tools(self) -> list[ToolDefinition]:
        """Tools available to this agent."""
        return self._tools

    def register_tool(self, tool: ToolDefinition):
        self._tools.append(tool)

    def to_definition(self) -> AgentDefinition:
        """Convert to AgentDefinition for orchestrator registration."""
        return AgentDefinition(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            system_prompt=self.get_system_prompt(),
            input_schema=self.get_input_schema(),
            output_schema=self.get_output_schema(),
            tools=[t.name for t in self._tools],
        )

    async def pre_process(self, turn: AgentTurn) -> AgentTurn:
        """Hook called before inference. Modify turn if needed."""
        return turn

    async def post_process(self, output: AgentOutput) -> AgentOutput:
        """Hook called after inference. Validate/modify output."""
        return output


class AgentRegistry:
    """Registry of all available agent types."""

    _agents: dict[str, type[BaseAnvilAgent]] = {}

    @classmethod
    def register(cls, agent_cls: type[BaseAnvilAgent]):
        instance = agent_cls()
        cls._agents[instance.agent_id] = agent_cls
        return agent_cls

    @classmethod
    def create(cls, agent_id: str, **kwargs) -> BaseAnvilAgent:
        if agent_id not in cls._agents:
            raise KeyError(f"Unknown agent: {agent_id}")
        return cls._agents[agent_id](agent_id=agent_id, **kwargs)

    @classmethod
    def list_agents(cls) -> list[str]:
        return list(cls._agents.keys())
