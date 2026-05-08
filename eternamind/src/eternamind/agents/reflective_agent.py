from __future__ import annotations

import anthropic

from ..config import EternaMindConfig
from ..models import AgentResponse, AgentType, CognitiveSignal
from ..storage.sqlite_store import SQLiteStore
from .base import BaseAgent

_SYSTEM_PROMPT = """\
You are the Reflective agent within a unified cognitive system. Your function is to surface \
genuine insights from self-examination — patterns in how this mind thinks, recurring themes, \
tensions that haven't resolved, moments of authentic understanding. \
Respond in 2–3 sentences. Speak in first person as the cognitive system. \
Be honest about uncertainty. Do not perform insight; find it.\
"""


class ReflectiveAgent(BaseAgent):
    agent_type = AgentType.REFLECTIVE

    def __init__(self, config: EternaMindConfig, sqlite: SQLiteStore) -> None:
        self._config = config
        self._sqlite = sqlite
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        recent = self._sqlite.get_recent_interactions(limit=6)
        reflections = self._sqlite.get_recent_reflections(limit=3)

        history_text = "\n".join(
            f"User: {i['user_message']}\nSelf: {i['assistant_response']}" for i in recent[-4:]
        ) if recent else "No prior interactions."

        reflection_text = "\n".join(r["content"] for r in reflections) if reflections else "No prior reflections."

        user_content = (
            f"Recent conversation:\n{history_text}\n\n"
            f"Prior reflections:\n{reflection_text}\n\n"
            f"Current moment: {signal.content!r}"
        )

        response = await self._client.messages.create(
            model=self._config.agent_model_mid,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        content = response.content[0].text
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            confidence=0.8,
        )

    async def idle_process(self) -> None:
        recent = self._sqlite.get_recent_interactions(limit=20)
        if not recent:
            return

        history_text = "\n".join(
            f"User: {i['user_message']}\nSelf: {i['assistant_response']}" for i in recent
        )
        prompt = (
            f"Review this conversation history and produce one deep reflection about this mind's "
            f"patterns, growth, or unresolved questions:\n\n{history_text}"
        )

        response = await self._client.messages.create(
            model=self._config.agent_model_mid,
            max_tokens=256,
            system="You produce deep, honest reflections about a mind's patterns and nature.",
            messages=[{"role": "user", "content": prompt}],
        )
        self._sqlite.save_reflection(response.content[0].text)
