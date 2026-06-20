"""
AnvilAgent API Layer.

Provides REST endpoints for agent lifecycle management
and WebSocket for real-time telemetry streaming.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from anvil.core.agent_state import AgentDefinition

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None  # type: ignore[assignment]

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = object
    BaseModel = object

logger = logging.getLogger(__name__)


# ─── Request/Response Schemas ───────────────────────────────────────


class AgentCreateRequest(BaseModel):
    agent_id: str
    name: str
    description: str = ""
    system_prompt: str = "You are a helpful assistant."
    input_schema: dict = {"type": "object"}
    output_schema: dict = {"type": "object"}
    tools: list[str] = []
    max_turns: int = 5
    timeout_s: int = 120


class AgentRunRequest(BaseModel):
    agent_id: str
    input: dict = {}
    max_tokens: int = 4096
    temperature: float = 0.1


class SessionResponse(BaseModel):
    session_id: str
    status: str
    agents_registered: int = 0
    turns_completed: int = 0
    tokens_generated: int = 0


class TelemetrySnapshot(BaseModel):
    session_id: str
    active_agent: str | None = None
    status: str
    ram_used_gb: float = 0.0
    vram_used_gb: float = 0.0
    tokens_per_second: float = 0.0
    kv_cache_mb: float = 0.0
    agent_queue: list[str] = []
    uptime_s: float = 0.0


# ─── Connection Manager for WebSocket ──────────────────────────────


class ConnectionManager:
    """Manages WebSocket connections for telemetry streaming."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.append(ws)
        logger.info(f"WebSocket connected: {ws.client}")

    def disconnect(self, ws: WebSocket):
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def active_connections(self) -> int:
        return len(self._connections)


# ─── AnvilAgent API App ────────────────────────────────────────────


class AnvilAPI:
    """
    Main API application for AnvilAgent.
    Coordinates agent lifecycle, inference routing, and telemetry.
    """

    def __init__(self, ssd_pool=None, context_restorer=None):
        self._orchestrator = None
        self._connection_manager = ConnectionManager()
        self._app = None
        self._telemetry_task: asyncio.Task | None = None
        self._ssd_pool = ssd_pool
        self._context_restorer = context_restorer
        self._last_tokens = 0
        self._last_telemetry_time = 0.0
        self._vram_total_gb = 0.0
        self._detect_vram()
        self._create_app()

    def _detect_vram(self):
        try:
            from anvil.hardware.detector import detect as detect_hw

            hw = detect_hw()
            self._vram_total_gb = hw.vram_total_bytes / (1024**3)
        except Exception:
            self._vram_total_gb = 0.0

    def _create_app(self):
        if not HAS_FASTAPI:
            logger.warning("FastAPI not installed. API endpoints disabled.")
            return

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self._telemetry_task = asyncio.create_task(self._telemetry_loop())
            yield
            if self._telemetry_task:
                self._telemetry_task.cancel()

        app = FastAPI(
            title="AnvilAgent API",
            version="0.1.0",
            lifespan=lifespan,
        )

        # ─── Health ─────────────────────────────────────────────
        @app.get("/health")
        async def health():
            return {"status": "ok", "version": "0.1.0"}

        # ─── Agent Management ──────────────────────────────────
        @app.post("/agents/register")
        async def register_agent(req: AgentCreateRequest):
            if self._orchestrator is None:
                raise HTTPException(503, "Orchestrator not initialized")
            agent = AgentDefinition(**req.model_dump())
            self._orchestrator.register_agent(agent)
            return {"agent_id": req.agent_id, "status": "registered"}

        @app.get("/agents")
        async def list_agents():
            if self._orchestrator is None:
                return {"agents": []}
            return {
                "agents": [
                    {"agent_id": aid, "name": a.name}
                    for aid, a in self._orchestrator.state.agents.items()
                ]
            }

        # ─── Session / Execution ───────────────────────────────
        @app.post("/session/run")
        async def run_session(req: AgentRunRequest):
            if self._orchestrator is None:
                raise HTTPException(503, "Orchestrator not initialized")
            results = await self._orchestrator.run_session(
                req.agent_id,
                req.input,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            return {
                "session_id": str(self._orchestrator.state.session_id),
                "turns": len(results),
                "outputs": [
                    {
                        "agent_id": r.agent_id,
                        "tokens": r.tokens_generated,
                        "next": r.next_agent,
                        "error": r.error,
                    }
                    for r in results
                ],
            }

        @app.get("/session/status")
        async def session_status():
            if self._orchestrator is None:
                return SessionResponse(session_id="", status="not_initialized")
            summary = self._orchestrator.get_state_summary()
            return SessionResponse(
                session_id=summary["session_id"],
                status=summary["status"],
                agents_registered=summary["agents_registered"],
                turns_completed=summary["turns_completed"],
                tokens_generated=summary["tokens_generated"],
            )

        # ─── Telemetry WebSocket ───────────────────────────────
        @app.websocket("/ws/telemetry")
        async def telemetry_websocket(ws: WebSocket):
            await self._connection_manager.connect(ws)
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                self._connection_manager.disconnect(ws)

        # ─── KV Cache Management ───────────────────────────────
        @app.get("/memory/kv-stats")
        async def kv_stats():
            active = 0
            pages_on_disk = 0
            total_compressed_mb = 0.0

            if self._ssd_pool is not None:
                blocks = self._ssd_pool.list_blocks()
                pages_on_disk = len(blocks)
                total_compressed_mb = (
                    sum(b.get("size_bytes", 0) for b in blocks) / (1024**2)
                )

            if self._context_restorer is not None:
                active = 1 if self._context_restorer._active_context else 0

            return {
                "active_slots": active,
                "pages_on_disk": pages_on_disk,
                "total_compressed_mb": round(total_compressed_mb, 2),
            }

        self._app = app

    async def _telemetry_loop(self):
        """Broadcast telemetry snapshots every 1s via WebSocket."""
        self._last_telemetry_time = time.monotonic()
        while True:
            await asyncio.sleep(1.0)
            if self._orchestrator and self._connection_manager.active_connections > 0:
                summary = self._orchestrator.get_state_summary()

                ram_gb = 0.0
                if HAS_PSUTIL:
                    ram_gb = psutil.virtual_memory().used / (1024**3)

                vram_gb = 0.0
                try:
                    from anvil.hardware.detector import detect as detect_hw
                    hw = detect_hw()
                    vram_gb = (hw.vram_total_bytes - hw.vram_free_bytes) / (1024**3)
                except Exception:
                    vram_gb = self._vram_total_gb

                now = time.monotonic()
                dt = now - self._last_telemetry_time
                tokens_now = summary.get("tokens_generated", 0)
                tokens_per_s = (tokens_now - self._last_tokens) / dt if dt > 0 else 0.0
                self._last_tokens = tokens_now
                self._last_telemetry_time = now

                kv_cache_mb = 0.0
                if self._ssd_pool is not None:
                    blocks = self._ssd_pool.list_blocks()
                    kv_cache_mb = (
                        sum(b.get("size_bytes", 0) for b in blocks) / (1024**2)
                    )

                snap = TelemetrySnapshot(
                    session_id=summary["session_id"],
                    active_agent=summary["active_agent"],
                    status=summary["status"],
                    ram_used_gb=round(ram_gb, 2),
                    vram_used_gb=round(vram_gb, 2),
                    tokens_per_second=round(tokens_per_s, 2),
                    kv_cache_mb=round(kv_cache_mb, 2),
                    agent_queue=summary.get("queue", []),
                    uptime_s=summary["uptime"],
                )
                await self._connection_manager.broadcast(snap.model_dump())

    def set_orchestrator(self, orchestrator):
        """Inject orchestrator instance."""
        self._orchestrator = orchestrator

    @property
    def app(self):
        return self._app


# ─── Factory ───────────────────────────────────────────────────────


def create_app(orchestrator=None, ssd_pool=None, context_restorer=None,
               pipeline=None) -> FastAPI:
    """Factory to create and configure the FastAPI app."""
    if pipeline is not None:
        ssd_pool = ssd_pool or pipeline.ssd_pool
        context_restorer = context_restorer or getattr(
            pipeline.context_scheduler, 'restorer', None
        )
    api = AnvilAPI(ssd_pool=ssd_pool, context_restorer=context_restorer)
    if orchestrator:
        api.set_orchestrator(orchestrator)
    return api.app
