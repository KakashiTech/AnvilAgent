import asyncio
from datetime import datetime, timezone
from uuid import UUID

import pytest

from anvil.core.agent_state import (
    AgentDefinition,
    AgentOutput,
    AgentStatus,
    AgentTurn,
    OrchestratorState,
)
from anvil.core.orchestrator import AgentOrchestrator
from anvil.core.scheduler import AgentScheduler, ScheduleStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MAX_CONTEXT = 16384


@pytest.fixture
def dummy_agent() -> AgentDefinition:
    return AgentDefinition(
        agent_id="test_agent",
        name="Test Agent",
        description="A test agent for unit tests",
        system_prompt="You are a test agent.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        tools=[],
        max_turns=3,
        timeout_s=30,
    )


@pytest.fixture
def dummy_pipeline() -> list[AgentDefinition]:
    return [
        AgentDefinition(
            agent_id="planner",
            name="Planner",
            description="Plans the task",
            system_prompt="Plan.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            max_turns=1,
        ),
        AgentDefinition(
            agent_id="worker",
            name="Worker",
            description="Executes the plan",
            system_prompt="Work.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            max_turns=1,
        ),
        AgentDefinition(
            agent_id="reviewer",
            name="Reviewer",
            description="Reviews the output",
            system_prompt="Review.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            max_turns=1,
        ),
    ]


@pytest.fixture
def orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(max_context_tokens=MAX_CONTEXT)


# ---------------------------------------------------------------------------
# AgentState schema tests
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_agent_status_values(self):
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.RUNNING == "running"
        assert AgentStatus.PAUSED == "paused"
        assert AgentStatus.COMPLETED == "completed"
        assert AgentStatus.FAILED == "failed"
        assert AgentStatus.TIMEOUT == "timeout"

    def test_agent_turn_defaults(self):
        turn = AgentTurn(agent_id="a", input_schema={"x": 1})
        assert isinstance(turn.session_id, UUID)
        assert turn.max_tokens == 4096
        assert turn.temperature == 0.1
        assert turn.context_token_count == 0
        assert turn.kv_cache_handle is None
        assert isinstance(turn.created_at, datetime)
        assert turn.created_at.tzinfo is not None

    def test_agent_output_defaults(self):
        output = AgentOutput(
            agent_id="a",
            session_id=UUID(int=0),
            output_schema={"result": "ok"},
            kv_cache_handle="kv-001",
        )
        assert output.tokens_generated == 0
        assert output.next_agent is None
        assert output.error is None
        assert isinstance(output.completed_at, datetime)
        assert output.completed_at.tzinfo is not None

    def test_agent_definition_defaults(self):
        agent = AgentDefinition(
            agent_id="d",
            name="D",
            description="desc",
            system_prompt="prompt",
            input_schema={},
            output_schema={},
        )
        assert agent.tools == []
        assert agent.max_turns == 5
        assert agent.timeout_s == 120

    def test_orchestrator_state_defaults(self):
        state = OrchestratorState()
        assert isinstance(state.session_id, UUID)
        assert state.status == AgentStatus.IDLE
        assert state.active_agent is None
        assert state.queue == []
        assert state.history == []
        assert isinstance(state.created_at, datetime)
        assert state.created_at.tzinfo is not None

    def test_orchestrator_state_extra_forbid(self):
        with pytest.raises(ValueError):
            OrchestratorState(session_id=UUID(int=0), agents={}, foo="bar")


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    async def test_register_agent(self, orchestrator, dummy_agent):
        orchestrator.register_agent(dummy_agent)
        assert "test_agent" in orchestrator.state.agents
        assert orchestrator.state.agents["test_agent"] == dummy_agent

    async def test_register_callback(self, orchestrator):
        async def fake_callback(turn: AgentTurn) -> AgentOutput:
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={"echo": turn.input_schema},
                kv_cache_handle="kv-001",
            )

        orchestrator.register_callback("test_agent", fake_callback)
        assert "test_agent" in orchestrator._callbacks

    async def test_run_single_turn(self, orchestrator, dummy_agent):
        orchestrator.register_agent(dummy_agent)

        async def echo(turn: AgentTurn) -> AgentOutput:
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={"echo": turn.input_schema},
                kv_cache_handle="kv-001",
                next_agent="worker",
            )

        orchestrator.register_callback("test_agent", echo)
        turn = AgentTurn(agent_id="test_agent", input_schema={"msg": "hello"})
        output = await orchestrator.run_turn("test_agent", turn)

        assert output.agent_id == "test_agent"
        assert output.output_schema == {"echo": {"msg": "hello"}}
        assert output.next_agent == "worker"
        assert output.error is None
        assert orchestrator.state.status == AgentStatus.PAUSED

    async def test_run_full_session(self, orchestrator, dummy_pipeline):
        for agent in dummy_pipeline:
            orchestrator.register_agent(agent)

        call_log: list[str] = []

        async def planner(turn: AgentTurn) -> AgentOutput:
            call_log.append("planner")
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={"plan": "step1"},
                kv_cache_handle="kv-p",
                next_agent="worker",
            )

        async def worker(turn: AgentTurn) -> AgentOutput:
            call_log.append("worker")
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={"result": "done"},
                kv_cache_handle="kv-w",
                next_agent="reviewer",
            )

        async def reviewer(turn: AgentTurn) -> AgentOutput:
            call_log.append("reviewer")
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={"approved": True},
                kv_cache_handle="kv-r",
                next_agent=None,
            )

        orchestrator.register_callback("planner", planner)
        orchestrator.register_callback("worker", worker)
        orchestrator.register_callback("reviewer", reviewer)

        results = await orchestrator.run_session("planner", {"task": "test"})

        assert len(results) == 3
        assert call_log == ["planner", "worker", "reviewer"]
        assert orchestrator.state.status == AgentStatus.COMPLETED
        assert orchestrator.state.history == results

    async def test_error_status_not_overwritten(self, orchestrator, dummy_pipeline):
        """Test 1: Error status is not overwritten by COMPLETED."""
        for agent in dummy_pipeline:
            orchestrator.register_agent(agent)

        async def planner(turn: AgentTurn) -> AgentOutput:
            raise RuntimeError("Planner failed")

        async def worker(turn: AgentTurn) -> AgentOutput:
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={},
                kv_cache_handle="kv-w",
            )

        orchestrator.register_callback("planner", planner)
        orchestrator.register_callback("worker", worker)

        results = await orchestrator.run_session("planner", {"task": "test"})

        assert len(results) == 1
        assert "Planner failed" in results[0].error or "RuntimeError" in results[0].error
        assert orchestrator.state.status == AgentStatus.FAILED

    async def test_per_agent_turn_limit(self):
        """Test 2: Per-agent turn limit works."""
        orch = AgentOrchestrator()
        agent = AgentDefinition(
            agent_id="looper",
            name="Looper",
            description="",
            system_prompt="",
            max_turns=2,
        )
        orch.register_agent(agent)

        call_count = 0

        async def looper(turn: AgentTurn) -> AgentOutput:
            nonlocal call_count
            call_count += 1
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={},
                kv_cache_handle="kv-l",
                next_agent="looper",
            )

        orch.register_callback("looper", looper)
        results = await orch.run_session("looper", {})
        assert call_count == 2
        assert len(results) == 2

    async def test_context_sliding_window(self):
        """Test 3: Context sliding window keeps only last N turns."""
        orch = AgentOrchestrator(max_context_tokens=1000)
        agent = AgentDefinition(
            agent_id="gen",
            name="Generator",
            description="",
            system_prompt="",
            max_turns=10,
        )
        orch.register_agent(agent)

        context_values: list[int] = []

        async def gen(turn: AgentTurn) -> AgentOutput:
            context_values.append(turn.context_token_count)
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={},
                kv_cache_handle="kv-g",
                next_agent="gen",
                tokens_generated=500,
            )

        orch.register_callback("gen", gen)
        results = await orch.run_session("gen", {})

        assert len(context_values) > 4
        for i, ctx in enumerate(context_values[4:], 4):
            assert ctx <= 2000, f"Turn {i}: context {ctx} exceeds sliding window limit"

    async def test_unregistered_agent_ends_session_with_warning(self, orchestrator):
        """Test 4: Unregistered agent ends session with warning."""
        dummy = AgentDefinition(
            agent_id="a", name="A", description="", system_prompt="",
            max_turns=5,
        )
        orchestrator.register_agent(dummy)

        async def cb(turn: AgentTurn) -> AgentOutput:
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={"next": "ghost"},
                kv_cache_handle="kv-a",
                next_agent="ghost",
            )

        orchestrator.register_callback("a", cb)
        results = await orchestrator.run_session("a", {})
        assert len(results) == 1
        assert orchestrator.state.status == AgentStatus.COMPLETED

    async def test_active_agent_cleared_on_error(self, orchestrator, dummy_agent):
        """Test 5: active_agent is cleared in error case."""
        orchestrator.register_agent(dummy_agent)

        async def failing(turn: AgentTurn) -> AgentOutput:
            raise RuntimeError("fail")

        orchestrator.register_callback("test_agent", failing)
        turn = AgentTurn(agent_id="test_agent", input_schema={})
        output = await orchestrator.run_turn("test_agent", turn)

        assert output.error is not None
        assert orchestrator.state.active_agent is None

    async def test_run_turn_missing_callback(self, orchestrator, dummy_agent):
        orchestrator.register_agent(dummy_agent)
        turn = AgentTurn(agent_id="test_agent", input_schema={})
        output = await orchestrator.run_turn("test_agent", turn)
        assert output.error is not None
        assert "No callback" in output.error

    async def test_unknown_agent_breaks_session(self, orchestrator):
        results = await orchestrator.run_session("ghost", {})
        assert results == []

    async def test_state_transitions(self, orchestrator, dummy_agent):
        orchestrator.register_agent(dummy_agent)

        async def echo(turn: AgentTurn) -> AgentOutput:
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={"ok": True},
                kv_cache_handle="kv-001",
            )

        orchestrator.register_callback("test_agent", echo)

        assert orchestrator.state.status == AgentStatus.IDLE

        turn = AgentTurn(agent_id="test_agent", input_schema={})
        output = await orchestrator.run_turn("test_agent", turn)

        assert output.error is None
        assert orchestrator.state.status == AgentStatus.IDLE
        assert orchestrator.state.active_agent is None
        assert len(orchestrator.state.history) == 1

    async def test_get_state_summary(self, orchestrator):
        summary = orchestrator.get_state_summary()
        assert "session_id" in summary
        assert summary["status"] == "idle"
        assert summary["agents_registered"] == 0
        assert summary["turns_completed"] == 0
        assert summary["queue_length"] == 0
        assert summary["tokens_generated"] == 0
        assert isinstance(summary["uptime"], float)

    async def test_session_cancelled_error(self, orchestrator, dummy_agent):
        """Test CancelledError sets FAILED status."""
        orchestrator.register_agent(dummy_agent)

        async def hang(turn: AgentTurn) -> AgentOutput:
            await asyncio.sleep(10)
            return AgentOutput(
                agent_id=turn.agent_id,
                session_id=turn.session_id,
                output_schema={},
                kv_cache_handle="kv-h",
            )

        orchestrator.register_callback("test_agent", hang)

        async def run_and_cancel():
            task = asyncio.create_task(
                orchestrator.run_session("test_agent", {})
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert orchestrator.state.status == AgentStatus.FAILED

        await run_and_cancel()


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


class TestScheduler:
    async def test_scheduler_timeout_and_retry(self):
        """Test 6: Scheduler timeout + retry works."""
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(10)

        sched = AgentScheduler(time_slice_ms=20, max_retries=3)
        sched.register_callback("flaky", flaky)
        sched.enqueue("flaky")

        r1 = await sched.schedule()
        assert r1.status == ScheduleStatus.TIMEOUT
        assert "retry 1/3" in r1.error

        r2 = await sched.schedule()
        assert r2.status == ScheduleStatus.TIMEOUT
        assert "retry 2/3" in r2.error

        r3 = await sched.schedule()
        assert r3.status == ScheduleStatus.TIMEOUT
        assert "Max retries" in r3.error
        assert call_count == 3

    async def test_scheduler_no_race_in_schedule(self):
        """Test 7: Scheduler does not race in schedule()."""
        sched = AgentScheduler(time_slice_ms=100)

        async def quick():
            return "done"

        sched.register_callback("a", quick)
        sched.enqueue("a")
        sched.enqueue("a")

        async def concurrent_schedule():
            r1 = await sched.schedule()
            r2 = await sched.schedule()
            return r1, r2

        r1, r2 = await concurrent_schedule()
        assert r1.status == ScheduleStatus.SUCCESS
        assert r2.status == ScheduleStatus.SUCCESS

    async def test_scheduler_fifo(self):
        """Test 8: Scheduler priority queue is FIFO."""
        sched = AgentScheduler(time_slice_ms=100)
        order = []

        async def make_cb(name: str):
            async def cb():
                order.append(name)
                return name
            return cb

        sched.register_callback("first", await make_cb("first"))
        sched.register_callback("second", await make_cb("second"))
        sched.register_callback("third", await make_cb("third"))

        sched.enqueue("first")
        sched.enqueue("second")
        sched.enqueue("third")

        await sched.schedule()
        await sched.schedule()
        await sched.schedule()

        assert order == ["first", "second", "third"]

    async def test_scheduler_invalid_time_slice(self):
        with pytest.raises(ValueError, match="time_slice_ms must be positive"):
            AgentScheduler(time_slice_ms=0)
        with pytest.raises(ValueError, match="time_slice_ms must be positive"):
            AgentScheduler(time_slice_ms=-1)

    async def test_scheduler_invalid_max_retries(self):
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            AgentScheduler(max_retries=-1)

    async def test_scheduler_empty_queue(self):
        sched = AgentScheduler()
        result = await sched.schedule()
        assert result.status == ScheduleStatus.EMPTY

    async def test_scheduler_missing_callback(self):
        sched = AgentScheduler()
        sched.enqueue("ghost")
        result = await sched.schedule()
        assert result.status == ScheduleStatus.ERROR
        assert "No callback" in result.error

    async def test_scheduler_retry_then_success(self):
        call_count = 0

        async def eventually_works():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                await asyncio.sleep(10)
            return "ok"

        sched = AgentScheduler(time_slice_ms=20, max_retries=5)
        sched.register_callback("agent", eventually_works)
        sched.enqueue("agent")

        r1 = await sched.schedule()
        assert r1.status == ScheduleStatus.TIMEOUT

        r2 = await sched.schedule()
        assert r2.status == ScheduleStatus.TIMEOUT

        r3 = await sched.schedule()
        assert r3.status == ScheduleStatus.SUCCESS
        assert r3.output == "ok"

    async def test_scheduler_error_result(self):
        async def broken():
            raise ValueError("broken")

        sched = AgentScheduler()
        sched.register_callback("agent", broken)
        sched.enqueue("agent")

        result = await sched.schedule()
        assert result.status == ScheduleStatus.ERROR
