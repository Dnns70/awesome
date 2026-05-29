from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from .agents.base import BaseAgent
from .config import EternaMindConfig
from .models import AgentType
from .storage.sqlite_store import SQLiteStore

if TYPE_CHECKING:
    from .identity_tracker import IdentityTracker

logger = logging.getLogger(__name__)

_RESCORE_EVERY_N_CYCLES = 5
_GOAL_GEN_MIN_INTERACTIONS = 10
_GOAL_GEN_MAX_ACTIVE = 5


class CognitionState(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"


class ContinuousCognitionLoop:
    def __init__(
        self,
        config: EternaMindConfig,
        agents: dict[AgentType, BaseAgent],
        sqlite: SQLiteStore,
        identity_tracker: "IdentityTracker | None" = None,
    ) -> None:
        self._config = config
        self._agents = agents
        self._sqlite = sqlite
        self._identity_tracker = identity_tracker
        self._state = CognitionState.ACTIVE
        self._last_interaction: datetime = datetime.now(timezone.utc)
        self._task: asyncio.Task | None = None
        self._idle_cycle_count: int = 0

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
        self._idle_cycle_count += 1

        # Core agent idle processes
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

        # Memory importance rescoring — every N cycles
        if self._idle_cycle_count % _RESCORE_EVERY_N_CYCLES == 0:
            try:
                await self._rescore_memories_cycle()
            except Exception as exc:
                logger.warning("Memory rescore error: %s", exc)

        # Goal generation — if under threshold and enough interaction history
        interaction_count = self._sqlite.count_interactions()
        active_goal_count = self._sqlite.count_active_goals()
        if interaction_count >= _GOAL_GEN_MIN_INTERACTIONS and active_goal_count < _GOAL_GEN_MAX_ACTIVE:
            try:
                await self._goal_generation_cycle()
            except Exception as exc:
                logger.warning("Goal generation error: %s", exc)

        # Identity snapshot — based on interaction count
        if self._identity_tracker is not None:
            try:
                await self._identity_tracker.maybe_snapshot(interaction_count)
            except Exception as exc:
                logger.warning("Identity snapshot error: %s", exc)

    async def _rescore_memories_cycle(self) -> None:
        active_goals = self._sqlite.get_active_goals()
        goal_keywords: set[str] = set()
        for g in active_goals:
            goal_keywords.update(g["title"].lower().split())
            goal_keywords.update(g["description"].lower().split())

        memories = self._sqlite.get_memories_for_rescoring(limit=200)
        now = datetime.utcnow()

        for mem in memories:
            try:
                age_days = max(0, (now - datetime.fromisoformat(mem["timestamp"])).days)
                recency = math.exp(-0.05 * age_days)
                access_count = mem.get("access_count") or 0
                access_boost = 1.0 + 0.2 * min(access_count, 10)
                goal_relevance = 0.0
                if goal_keywords:
                    content_words = set(mem["content"].lower().split())
                    overlap = len(content_words & goal_keywords)
                    goal_relevance = min(overlap / max(len(goal_keywords), 1), 1.0)
                goal_boost = 1.0 + 0.3 * goal_relevance
                new_score = min(float(mem["importance"]) * recency * access_boost * goal_boost, 1.0)
                self._sqlite.update_memory_importance(mem["id"], new_score)
            except Exception as exc:
                logger.debug("Memory rescore skip for %s: %s", mem.get("id"), exc)

    async def _goal_generation_cycle(self) -> None:
        import anthropic
        import uuid

        recent = self._sqlite.get_recent_interactions(limit=15)
        reflections = self._sqlite.get_recent_reflections(limit=3)
        active_goals = self._sqlite.get_active_goals()

        if not recent:
            return

        history_text = "\n".join(
            f"User: {i['user_message']}\nSelf: {i['assistant_response'][:150]}" for i in recent[-8:]
        )
        reflection_text = "\n".join(r["content"] for r in reflections) if reflections else ""
        existing_goals_text = "\n".join(f"- {g['title']}" for g in active_goals) if active_goals else "None"

        prompt = (
            f"Based on this conversation history, identify 1–2 meaningful goals this mind should pursue:\n\n"
            f"History:\n{history_text}\n\n"
            f"Reflections:\n{reflection_text}\n\n"
            f"Already active goals:\n{existing_goals_text}\n\n"
            "Respond with one goal per line in the format: TITLE | DESCRIPTION | PRIORITY(1-10)\n"
            "Only propose goals that are genuinely new and meaningful. If nothing new is needed, respond: NONE"
        )

        client = anthropic.AsyncAnthropic(api_key=self._config.anthropic_api_key)
        response = await client.messages.create(
            model=self._config.agent_model_fast,
            max_tokens=256,
            system="You propose meaningful long-term goals based on a mind's patterns and interests.",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.upper() == "NONE" or not text:
            return

        for line in text.split("\n"):
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 1 and parts[0]:
                title = parts[0]
                description = parts[1] if len(parts) > 1 else ""
                try:
                    priority = int(parts[2]) if len(parts) > 2 else 5
                    priority = max(1, min(10, priority))
                except ValueError:
                    priority = 5
                goal_id = str(uuid.uuid4())
                self._sqlite.save_goal(goal_id, title, description, priority, source="self_generated")
                logger.debug("CCL: generated goal %r (priority %d)", title, priority)

    @property
    def state(self) -> CognitionState:
        return self._state
