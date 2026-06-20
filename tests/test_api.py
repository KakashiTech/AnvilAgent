"""Tests for the AnvilAgent API layer."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from anvil.api.router import (
    HAS_FASTAPI,
    AgentCreateRequest,
    AgentRunRequest,
    AnvilAPI,
    ConnectionManager,
    SessionResponse,
    TelemetrySnapshot,
    create_app,
)

if HAS_FASTAPI:
    from fastapi import WebSocket
    from fastapi.testclient import TestClient

    from anvil.core.agent_state import (
        AgentDefinition,
        AgentOutput,
        AgentStatus,
    )


pytestmark = pytest.mark.skipif(
    not HAS_FASTAPI, reason="FastAPI is not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_orchestrator():
    orb = MagicMock()
    state = MagicMock()
    state.agents = {}
    state.session_id = uuid4()
    state.history = []
    state.status = AgentStatus.IDLE
    state.queue = []
    state.created_at = __import__("datetime").datetime.now()
    orb.state = state
    orb.get_state_summary.return_value = {
        "session_id": str(state.session_id),
        "status": "idle",
        "active_agent": None,
        "agents_registered": 0,
        "turns_completed": 0,
        "queue_length": 0,
        "tokens_generated": 0,
        "uptime": 0.0,
    }
    return orb


@pytest.fixture
def api(mock_orchestrator):
    inst = AnvilAPI()
    inst.set_orchestrator(mock_orchestrator)
    return inst


@pytest.fixture
def client(api):
    return TestClient(api.app)


@pytest.fixture
def sample_agent_def():
    return AgentDefinition(
        agent_id="test_agent",
        name="Test Agent",
        description="A test agent",
        system_prompt="You are a test agent.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        tools=[],
        max_turns=5,
        timeout_s=120,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_agent_create_request_defaults(self):
        req = AgentCreateRequest(agent_id="a", name="A")
        assert req.description == ""
        assert req.system_prompt == "You are a helpful assistant."
        assert req.input_schema == {"type": "object"}
        assert req.output_schema == {"type": "object"}
        assert req.tools == []
        assert req.max_turns == 5
        assert req.timeout_s == 120

    def test_agent_create_request_override(self):
        req = AgentCreateRequest(
            agent_id="x",
            name="X",
            tools=["search", "code"],
            max_turns=10,
        )
        assert req.tools == ["search", "code"]
        assert req.max_turns == 10

    def test_agent_run_request_defaults(self):
        req = AgentRunRequest(agent_id="a")
        assert req.input == {}
        assert req.max_tokens == 4096
        assert req.temperature == 0.1

    def test_session_response_defaults(self):
        resp = SessionResponse(session_id="s1", status="idle")
        assert resp.agents_registered == 0
        assert resp.turns_completed == 0
        assert resp.tokens_generated == 0

    def test_telemetry_snapshot_defaults(self):
        snap = TelemetrySnapshot(session_id="s1", status="running")
        assert snap.ram_used_gb == 0.0
        assert snap.vram_used_gb == 0.0
        assert snap.tokens_per_second == 0.0
        assert snap.kv_cache_mb == 0.0
        assert snap.agent_queue == []
        assert snap.uptime_s == 0.0


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_is_fast(self, client):
        resp = client.get("/health")
        assert resp.elapsed.total_seconds() < 5.0


# ---------------------------------------------------------------------------
# Agent Management
# ---------------------------------------------------------------------------


class TestAgentManagement:
    def test_register_agent(self, client, mock_orchestrator):
        payload = {
            "agent_id": "planner",
            "name": "Planner Agent",
            "description": "Planning agent",
            "system_prompt": "You are a planner.",
        }
        resp = client.post("/agents/register", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "planner"
        assert data["status"] == "registered"
        mock_orchestrator.register_agent.assert_called_once()

    def test_register_agent_validates_required_fields(self, client):
        resp = client.post("/agents/register", json={})
        assert resp.status_code == 422

    def test_register_agent_missing_agent_id(self, client):
        resp = client.post("/agents/register", json={"name": "No ID"})
        assert resp.status_code == 422

    def test_list_agents_empty(self, client, mock_orchestrator):
        mock_orchestrator.state.agents = {}
        resp = client.get("/agents")
        assert resp.status_code == 200
        assert resp.json() == {"agents": []}

    def test_list_agents_with_entries(self, client, mock_orchestrator, sample_agent_def):
        mock_orchestrator.state.agents = {
            "test_agent": sample_agent_def,
        }
        resp = client.get("/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["agents"]) == 1
        assert data["agents"][0]["agent_id"] == "test_agent"
        assert data["agents"][0]["name"] == "Test Agent"

    def test_list_agents_no_orchestrator(self):
        app = create_app(orchestrator=None)
        client = TestClient(app)
        resp = client.get("/agents")
        assert resp.status_code == 200
        assert resp.json() == {"agents": []}


# ---------------------------------------------------------------------------
# Session / Execution
# ---------------------------------------------------------------------------


class TestSession:
    async def test_run_session(self, client, mock_orchestrator):
        fake_output = AgentOutput(
            agent_id="planner",
            session_id=uuid4(),
            output_schema={"plan": "done"},
            kv_cache_handle="kv-001",
            next_agent=None,
        )
        mock_orchestrator.run_session = AsyncMock(return_value=[fake_output])

        payload = {"agent_id": "planner", "input": {"task": "test"}}
        resp = client.post("/session/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["turns"] == 1
        assert data["outputs"][0]["agent_id"] == "planner"
        assert data["outputs"][0]["error"] is None

    async def test_run_session_forwards_max_tokens_and_temperature(self, client, mock_orchestrator):
        fake_output = AgentOutput(
            agent_id="planner",
            session_id=uuid4(),
            output_schema={},
            kv_cache_handle="kv-001",
            next_agent=None,
        )
        mock_orchestrator.run_session = AsyncMock(return_value=[fake_output])

        payload = {
            "agent_id": "planner",
            "input": {"task": "test"},
            "max_tokens": 2048,
            "temperature": 0.5,
        }
        resp = client.post("/session/run", json=payload)
        assert resp.status_code == 200
        mock_orchestrator.run_session.assert_called_once_with(
            "planner", {"task": "test"}, max_tokens=2048, temperature=0.5
        )

    async def test_run_session_with_error(self, client, mock_orchestrator):
        fake_output = AgentOutput(
            agent_id="planner",
            session_id=uuid4(),
            output_schema={},
            kv_cache_handle="kv-001",
            error="Planner failed",
        )
        mock_orchestrator.run_session = AsyncMock(return_value=[fake_output])

        payload = {"agent_id": "planner", "input": {"task": "test"}}
        resp = client.post("/session/run", json=payload)
        assert resp.status_code == 200
        assert resp.json()["outputs"][0]["error"] == "Planner failed"

    async def test_run_session_missing_agent_id(self, client):
        resp = client.post("/session/run", json={})
        assert resp.status_code == 422

    def test_session_status(self, client, mock_orchestrator):
        resp = client.get("/session/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == str(mock_orchestrator.state.session_id)
        assert data["status"] == "idle"
        assert data["agents_registered"] == 0
        assert data["turns_completed"] == 0
        assert data["tokens_generated"] == 0

    def test_session_status_no_orchestrator(self):
        app = create_app(orchestrator=None)
        client = TestClient(app)
        resp = client.get("/session/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == ""
        assert data["status"] == "not_initialized"


# ---------------------------------------------------------------------------
# 503 when orchestrator not initialized
# ---------------------------------------------------------------------------


class TestOrchestratorUnavailable:
    def test_register_agent_503(self):
        app = create_app(orchestrator=None)
        client = TestClient(app)
        payload = {"agent_id": "x", "name": "X"}
        resp = client.post("/agents/register", json=payload)
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Orchestrator not initialized"

    def test_run_session_503(self):
        app = create_app(orchestrator=None)
        client = TestClient(app)
        payload = {"agent_id": "x", "input": {}}
        resp = client.post("/session/run", json=payload)
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Orchestrator not initialized"


# ---------------------------------------------------------------------------
# KV Cache endpoint
# ---------------------------------------------------------------------------


class TestMemory:
    def test_kv_stats_without_pool(self, client):
        resp = client.get("/memory/kv-stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_slots"] == 0
        assert data["pages_on_disk"] == 0
        assert data["total_compressed_mb"] == 0.0

    def test_kv_stats_with_mock_pool(self):
        mock_pool = MagicMock()
        mock_pool.list_blocks.return_value = [
            {"block_id": "b1", "size_bytes": 2 * 1024 * 1024},
            {"block_id": "b2", "size_bytes": 1 * 1024 * 1024},
        ]
        app = create_app(orchestrator=None, ssd_pool=mock_pool)
        client = TestClient(app)
        resp = client.get("/memory/kv-stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pages_on_disk"] == 2
        assert data["total_compressed_mb"] > 0

    def test_kv_stats_with_context_restorer(self):
        mock_restorer = MagicMock()
        mock_restorer._active_context = MagicMock()
        app = create_app(orchestrator=None, context_restorer=mock_restorer)
        client = TestClient(app)
        resp = client.get("/memory/kv-stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_slots"] == 1


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


class TestWebSocket:
    def test_websocket_connect_disconnect(self, client):
        with client.websocket_connect("/ws/telemetry") as ws:
            ws.send_text("ping")
            ws.close()

    def test_websocket_multiple_connections(self, client):
        with (
            client.websocket_connect("/ws/telemetry") as ws1,
            client.websocket_connect("/ws/telemetry") as ws2,
        ):
            ws1.send_text("ping")
            ws2.send_text("ping")
            ws1.close()
            ws2.close()

    def test_websocket_receives_telemetry(self, api, client):
        import asyncio

        snap = TelemetrySnapshot(
            session_id="test-session",
            status="running",
            ram_used_gb=1.5,
        )

        with client.websocket_connect("/ws/telemetry") as ws:
            asyncio.run(api._connection_manager.broadcast(snap.model_dump()))
            data = ws.receive_json()
            assert data["session_id"] == "test-session"
            assert data["status"] == "running"
            assert data["ram_used_gb"] == 1.5
            ws.close()


# ---------------------------------------------------------------------------
# ConnectionManager unit tests
# ---------------------------------------------------------------------------


class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        ws = MagicMock(spec=WebSocket)
        ws.client = "test-client"
        ws.accept = AsyncMock()

        cm = ConnectionManager()
        assert cm.active_connections == 0

        await cm.connect(ws)
        assert cm.active_connections == 1

        cm.disconnect(ws)
        assert cm.active_connections == 0

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple(self):
        ws1, ws2 = MagicMock(spec=WebSocket), MagicMock(spec=WebSocket)
        ws1.client = "c1"
        ws2.client = "c2"
        ws1.accept = AsyncMock()
        ws2.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2.send_json = AsyncMock()

        cm = ConnectionManager()
        await cm.connect(ws1)
        await cm.connect(ws2)

        await cm.broadcast({"msg": "hello"})
        ws1.send_json.assert_awaited_once_with({"msg": "hello"})
        ws2.send_json.assert_awaited_once_with({"msg": "hello"})

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self):
        ws_live = MagicMock(spec=WebSocket)
        ws_dead = MagicMock(spec=WebSocket)
        ws_live.client = "live"
        ws_dead.client = "dead"
        ws_live.accept = AsyncMock()
        ws_dead.accept = AsyncMock()
        ws_live.send_json = AsyncMock()
        ws_dead.send_json = AsyncMock()
        ws_dead.send_json = AsyncMock(side_effect=Exception("gone"))

        cm = ConnectionManager()
        await cm.connect(ws_live)
        await cm.connect(ws_dead)
        assert cm.active_connections == 2

        await cm.broadcast({"msg": "test"})
        assert cm.active_connections == 1
        ws_live.send_json.assert_awaited_once()


# ---------------------------------------------------------------------------
# create_app factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_app_without_orchestrator(self):
        app = create_app(orchestrator=None)
        assert app is not None
        assert app.title == "AnvilAgent API"

    def test_create_app_with_orchestrator(self, mock_orchestrator):
        app = create_app(orchestrator=mock_orchestrator)
        assert app is not None
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_create_app_with_memory_components(self):
        mock_pool = MagicMock()
        mock_restorer = MagicMock()
        app = create_app(
            orchestrator=None,
            ssd_pool=mock_pool,
            context_restorer=mock_restorer,
        )
        assert app is not None
        assert app.title == "AnvilAgent API"
