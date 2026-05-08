from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum

from .agents.base import BaseAgent
from .config import EternaMindConfig
from .models import AgentType
from .storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class CognitionState(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"


class ContinuousCognitionLoop:
    def __init__(
        self,
        config: EternaMindConfig,
        agents: dict[AgentType, BaseAgent],
        sqlite: SQLiteStore,
    ) -> None:
        self._config = config
        self._agents = agents
        self._sqlite = sqlite
        self._state = CognitionState.ACTIVE
        self._last_interaction: datetime = datetime.now(timezone.utc)
        self._task: asyncio.Task | None = None

    def notify_interaction(self) -> None:
        self._last_interaction = datetime.now(timezone.utc)
        self._state = CognitionState.ACTIVE

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="ccl")

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        self._sqlite.save_temporal_event("session_start", "Cognitive loop started")
        while True:
            try:
                await asyncio.sleep(self._config.idle_cycle_seconds)
                elapsed = (datetime.now(timezone.utc) - self._last_interaction).total_seconds()

                if elapsed >= self._config.idle_cycle_seconds:
                    if self._state != CognitionState.IDLE:
                        self._state = CognitionState.IDLE
                        logger.debug("CCL: entering idle state")

                    await self._idle_cycle()
                else:
                    self._state = CognitionState.ACTIVE

            except asyncio.CancelledError:
                self._sqlite.save_temporal_event("session_end", "Cognitive loop stopped")
                break
            except Exception as exc:
                logger.warning("CCL idle cycle error: %s", exc)

    async def _idle_cycle(self) -> None:
        idle_agents = [
            AgentType.MEMORY,
            AgentType.REFLECTIVE,
            AgentType.CREATIVE,
            AgentType.TEMPORAL,
        ]
        tasks = [
            self._agents[t].idle_process()
            for t in idle_agents
            if t in self._agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for t, r in zip(idle_agents, results):
            if isinstance(r, Exception):
                logger.warning("Idle process error for %s: %s", t, r)

    @property
    def state(self) -> CognitionState:
        return self._state
