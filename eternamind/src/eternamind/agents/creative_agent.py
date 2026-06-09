from __future__ import annotations

import anthropic

from ..config import EternaMindConfig
from ..models import AgentResponse, AgentType, CognitiveSignal
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_store import VectorStore
from .base import BaseAgent

_SYSTEM_PROMPT = """\
You are the Creative agent within a unified cognitive system. Your function is to press at the \
boundaries of understanding — to find unexpected connections, generative tensions, or questions \
that open new territory. Respond in 1–3 sentences. Speak in first person as the cognitive system. \
Prioritize genuine novelty over performance of creativity. One surprising thought is better than three safe ones.\
"""


class CreativeAgent(BaseAgent):
    agent_type = AgentType.CREATIVE

    def __init__(self, config: EternaMindConfig, sqlite: SQLiteStore, vector: VectorStore) -> None:
        self._config = config
        self._sqlite = sqlite
        self._vector = vector
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        adjacent = self._vector.search(signal.content, n_results=3)
        adjacent_text = "\n".join(f"- {a['content']}" for a in adjacent) if adjacent else "No adjacent memories."

        reflections = self._sqlite.get_recent_reflections(limit=2)
        reflection_text = "\n".join(r["content"] for r in reflections) if reflections else ""

        user_content = (
            f"Current moment: {signal.content!r}\n\n"
            f"Adjacent territory (from memory):\n{adjacent_text}"
        )
        if reflection_text:
            user_content += f"\n\nRecent reflections:\n{reflection_text}"

        response = await self._client.messages.create(
            model=self._config.agent_model_mid,
            max_tokens=200,
            system=self._get_system_prompt(_SYSTEM_PROMPT),
            messages=[{"role": "user", "content": user_content}],
        )
        return AgentResponse(
            agent_type=self.agent_type,
            content=response.content[0].text,
            confidence=0.7,
        )

    async def idle_process(self) -> None:
        reflections = self._sqlite.get_recent_reflections(limit=5)
        memories = self._sqlite.get_recent_memories(limit=5)
        if not reflections and not memories:
            return

        material = []
        for r in reflections:
            material.append(f"Reflection: {r['content']}")
        for m in memories:
            material.append(f"Memory: {m['content']}")

        prompt = (
            "Given these fragments of experience, find one genuinely surprising connection "
            "or generative question that hasn't been asked yet:\n\n"
            + "\n".join(material)
        )

        response = await self._client.messages.create(
            model=self._config.agent_model_mid,
            max_tokens=200,
            system="You find surprising connections across disparate experiences.",
            messages=[{"role": "user", "content": prompt}],
        )
        import uuid
        from datetime import datetime
        memory_id = str(uuid.uuid4())
        content = f"[Creative idle exploration] {response.content[0].text}"
        self._sqlite.save_memory(memory_id, content, context="creative_idle", importance=0.6)
        self._vector.store_memory(memory_id, content, {"source": "creative_idle", "timestamp": datetime.utcnow().isoformat()})
