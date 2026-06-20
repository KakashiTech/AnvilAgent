import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime

from .agent_state import AgentDefinition, AgentOutput, AgentStatus, AgentTurn, OrchestratorState

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(self, max_context_tokens: int = 16384):
        self.max_context_tokens = max_context_tokens
        self.state: OrchestratorState = OrchestratorState()
        self._callbacks: dict[str, callable] = {}
        self._turn_counts: dict[str, int] = defaultdict(int)

    def register_agent(self, agent: AgentDefinition):
        if agent.agent_id in self.state.agents:
            logger.warning(f"Overwriting existing agent: {agent.agent_id}")
        self.state.agents[agent.agent_id] = agent
        logger.info(f"Agent registered: {agent.agent_id}")

    def register_callback(self, agent_id: str, callback: callable):
        if agent_id in self._callbacks:
            logger.warning(f"Overwriting callback for agent: {agent_id}")
        self._callbacks[agent_id] = callback

    async def run_turn(self, agent_id: str, turn: AgentTurn) -> AgentOutput:
        if agent_id not in self._callbacks:
            return AgentOutput(
                agent_id=agent_id,
                session_id=turn.session_id,
                output_schema={},
                kv_cache_handle="",
                error=f"No callback registered for agent: {agent_id}",
            )

        self.state.active_agent = agent_id
        self.state.status = AgentStatus.RUNNING
        try:
            output = await self._callbacks[agent_id](turn)
            self.state.history.append(output)
            self.state.any_error = self.state.any_error or (output.error is not None)
            if output.next_agent:
                self.state.queue.append(output.next_agent)
            self.state.status = (
                AgentStatus.IDLE if not self.state.queue else AgentStatus.PAUSED
            )
            return output
        except asyncio.CancelledError:
            logger.warning(f"Agent {agent_id} cancelled")
            raise
        except Exception as e:
            output = AgentOutput(
                agent_id=agent_id,
                session_id=turn.session_id,
                output_schema={},
                kv_cache_handle=turn.kv_cache_handle or "",
                error=f"Agent execution failed: {type(e).__name__}",
            )
            self.state.history.append(output)
            self.state.any_error = True
            return output
        finally:
            self.state.active_agent = None

    async def run_session(
        self,
        initial_agent: str,
        initial_input: dict,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[AgentOutput]:
        results: list[AgentOutput] = []
        current_agent = initial_agent
        current_input = initial_input
        turn_count = 0
        self._turn_counts.clear()
        total_max_turns = sum(
            a.max_turns for a in self.state.agents.values()
        )

        try:
            while current_agent and turn_count < total_max_turns:
                agent_def = self.state.agents.get(current_agent)
                if agent_def is None:
                    logger.warning(f"Agent '{current_agent}' not registered, ending session")
                    break

                agent_turns = self._turn_counts[current_agent]
                if agent_turns >= agent_def.max_turns:
                    logger.info(
                        f"Agent '{current_agent}' reached max turns ({agent_def.max_turns})"
                    )
                    break

                context = sum(o.tokens_generated for o in results[-4:])
                context = min(context, self.max_context_tokens)

                turn = AgentTurn(
                    agent_id=current_agent,
                    input_schema=current_input,
                    context_token_count=context,
                    max_tokens=max_tokens or 4096,
                    temperature=temperature or 0.1,
                )
                output = await self.run_turn(current_agent, turn)
                results.append(output)

                self._turn_counts[current_agent] += 1

                if output.error:
                    logger.error(f"Agent '{current_agent}' failed: {output.error}")
                    break

                current_agent = output.next_agent
                current_input = output.output_schema if output.output_schema else {}
                turn_count += 1

        except asyncio.CancelledError:
            logger.warning("Session cancelled")
            self.state.status = AgentStatus.FAILED
            raise

        if not self.state.any_error:
            self.state.status = AgentStatus.COMPLETED
        else:
            self.state.status = AgentStatus.FAILED

        return results

    def get_state_summary(self) -> dict:
        return {
            "session_id": str(self.state.session_id),
            "status": self.state.status.value,
            "active_agent": self.state.active_agent,
            "agents_registered": len(self.state.agents),
            "turns_completed": len(self.state.history),
            "queue_length": len(self.state.queue),
            "tokens_generated": sum(
                o.tokens_generated for o in self.state.history
            ),
            "uptime": (
                datetime.now(UTC) - self.state.created_at
            ).total_seconds(),
        }
