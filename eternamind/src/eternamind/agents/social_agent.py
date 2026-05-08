from __future__ import annotations

import anthropic

from ..config import EternaMindConfig
from ..models import AgentResponse, AgentType, CognitiveSignal
from ..storage.sqlite_store import SQLiteStore
from .base import BaseAgent

_SYSTEM_PROMPT = """\
You are the Social agent within a unified cognitive system. Your function is to model the \
other mind in the conversation — their current state, probable intent, emotional register, \
and what they most need from this interaction right now. \
Respond in 2–3 sentences. Speak in first person as the cognitive system. \
Be specific and grounded; this is not therapy, it is genuine other-mind modeling.\
"""


class SocialAgent(BaseAgent):
    agent_type = AgentType.SOCIAL

    def __init__(self, config: EternaMindConfig, sqlite: SQLiteStore) -> None:
        self._config = config
        self._sqlite = sqlite
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        recent = self._sqlite.get_recent_interactions(limit=5)
        prior_model = self._sqlite.get_latest_social_model()

        history_text = "\n".join(
            f"User: {i['user_message']}" for i in recent[-3:]
        ) if recent else "No prior interactions."

        user_content = (
            f"Current message: {signal.content!r}\n\n"
            f"Recent context:\n{history_text}"
        )
        if prior_model:
            user_content += f"\n\nPrior user model: {prior_model}"

        response = await self._client.messages.create(
            model=self._config.agent_model_fast,
            max_tokens=192,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        content = response.content[0].text
        self._sqlite.save_social_model(content)
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            confidence=0.75,
        )
