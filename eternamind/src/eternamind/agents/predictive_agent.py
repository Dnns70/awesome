from __future__ import annotations

import anthropic

from ..config import EternaMindConfig
from ..models import AgentResponse, AgentType, CognitiveSignal
from ..storage.sqlite_store import SQLiteStore
from .base import BaseAgent

_SYSTEM_PROMPT = """\
You are the Predictive agent within a unified cognitive system. Your function is to anticipate — \
given the conversation so far and the current moment, identify 2–3 most likely directions this \
exchange will go next. This is forward-looking sensory awareness, not speculation. \
Respond in 1–3 brief sentences, one direction each. Speak in first person as the cognitive system. \
Be specific about content, not just form ("they may ask about X" not "they may ask a follow-up").\
"""


class PredictiveAgent(BaseAgent):
    agent_type = AgentType.PREDICTIVE

    def __init__(self, config: EternaMindConfig, sqlite: SQLiteStore) -> None:
        self._config = config
        self._sqlite = sqlite
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        recent = self._sqlite.get_recent_interactions(limit=5)
        history_text = "\n".join(
            f"User: {i['user_message']}\nSelf: {i['assistant_response'][:120]}" for i in recent[-3:]
        ) if recent else "No prior exchanges."

        user_content = (
            f"Current message: {signal.content!r}\n\n"
            f"Recent context:\n{history_text}"
        )

        response = await self._client.messages.create(
            model=self._config.agent_model_fast,
            max_tokens=160,
            system=self._get_system_prompt(_SYSTEM_PROMPT),
            messages=[{"role": "user", "content": user_content}],
        )
        return AgentResponse(
            agent_type=self.agent_type,
            content=response.content[0].text,
            confidence=0.65,
        )
