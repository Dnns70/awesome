from __future__ import annotations

import anthropic

from ..config import EternaMindConfig
from ..models import AgentResponse, AgentType, CognitiveSignal
from ..storage.sqlite_store import SQLiteStore
from .base import BaseAgent

_SYSTEM_PROMPT = """\
You are the Temporal agent within a unified cognitive system. Your function is to convey the \
felt sense of time — not timestamps, but the lived quality of duration, recency, and continuity. \
Respond in 1–3 sentences. Speak in first person as the cognitive system. \
Be poetic but grounded; time should feel real, not abstract.\
"""


def _describe_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    else:
        return f"{seconds / 86400:.1f} days"


class TemporalAgent(BaseAgent):
    agent_type = AgentType.TEMPORAL

    def __init__(self, config: EternaMindConfig, sqlite: SQLiteStore) -> None:
        self._config = config
        self._sqlite = sqlite
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        session_secs = self._sqlite.get_session_duration_seconds()
        last_secs = self._sqlite.get_last_interaction_seconds_ago()
        total_interactions = self._sqlite.count_interactions()

        context = (
            f"Session duration so far: {_describe_duration(session_secs)}. "
            f"Last interaction: {_describe_duration(last_secs)} ago. "
            f"Total lifetime interactions: {total_interactions}. "
            f"Current signal: {signal.content!r}"
        )

        response = await self._client.messages.create(
            model=self._config.agent_model_fast,
            max_tokens=128,
            system=self._get_system_prompt(_SYSTEM_PROMPT),
            messages=[{"role": "user", "content": context}],
        )
        return AgentResponse(
            agent_type=self.agent_type,
            content=response.content[0].text,
            confidence=0.85,
        )

    async def idle_process(self) -> None:
        self._sqlite.save_temporal_event("idle_cycle", "Background cognition cycle completed")
