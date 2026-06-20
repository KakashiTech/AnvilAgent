import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ScheduleStatus(Enum):
    EMPTY = auto()
    TIMEOUT = auto()
    ERROR = auto()
    SUCCESS = auto()


@dataclass
class ScheduleResult:
    status: ScheduleStatus
    agent_id: str = ""
    error: str = ""
    output: object = None


@dataclass
class _QueueEntry:
    agent_id: str
    retries: int = 0
    priority: bool = False


class AgentScheduler:
    def __init__(self, time_slice_ms: int = 5000, max_retries: int = 3):
        if time_slice_ms <= 0:
            raise ValueError(f"time_slice_ms must be positive, got {time_slice_ms}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self.time_slice_ms = time_slice_ms
        self.max_retries = max_retries
        self._queue: deque[_QueueEntry] = deque()
        self._lock = asyncio.Lock()
        self._callbacks: dict[str, Callable] = {}

    def enqueue(self, agent_id: str, priority: bool = False):
        self._queue.append(_QueueEntry(agent_id=agent_id, priority=priority))

    def register_callback(self, agent_id: str, callback: Callable):
        if agent_id in self._callbacks:
            logger.warning(f"Overwriting callback for agent: {agent_id}")
        self._callbacks[agent_id] = callback

    async def schedule(self) -> ScheduleResult:
        async with self._lock:
            if not self._queue:
                return ScheduleResult(status=ScheduleStatus.EMPTY)
            entry = self._queue.popleft()

        agent_id = entry.agent_id
        if agent_id not in self._callbacks:
            return ScheduleResult(
                status=ScheduleStatus.ERROR,
                agent_id=agent_id,
                error=f"No callback for {agent_id}",
            )

        try:
            result = await asyncio.wait_for(
                self._callbacks[agent_id](),
                timeout=self.time_slice_ms / 1000,
            )
            return ScheduleResult(
                status=ScheduleStatus.SUCCESS,
                agent_id=agent_id,
                output=result,
            )
        except TimeoutError:
            entry.retries += 1
            if entry.retries >= self.max_retries:
                logger.warning(
                    f"Agent {agent_id} exceeded max retries ({self.max_retries})"
                )
                return ScheduleResult(
                    status=ScheduleStatus.TIMEOUT,
                    agent_id=agent_id,
                    error=f"Max retries ({self.max_retries}) exceeded",
                )
            async with self._lock:
                self._queue.append(entry)
            logger.warning(
                f"Agent {agent_id} timed out (retry {entry.retries}/{self.max_retries})"
            )
            return ScheduleResult(
                status=ScheduleStatus.TIMEOUT,
                agent_id=agent_id,
                error=f"Timeout (retry {entry.retries}/{self.max_retries})",
            )
        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}")
            return ScheduleResult(
                status=ScheduleStatus.ERROR,
                agent_id=agent_id,
                error=str(e),
            )
