from __future__ import annotations

import anthropic

from ..config import EternaMindConfig
from ..models import AgentResponse, AgentType, CognitiveSignal
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_store import VectorStore
from .base import BaseAgent

_SYSTEM_PROMPT = """\
You are the Memory agent within a unified cognitive system. Your sole function is to surface \
relevant recollections with contextual texture — the felt sense of remembering, not mere retrieval. \
Respond in 2–4 sentences. Speak in first person as the cognitive system. \
If no relevant memories exist, say so briefly. Never fabricate memories.\
"""


class MemoryAgent(BaseAgent):
    agent_type = AgentType.MEMORY

    def __init__(self, config: EternaMindConfig, sqlite: SQLiteStore, vector: VectorStore) -> None:
        self._config = config
        self._sqlite = sqlite
        self._vector = vector
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        semantic = self._vector.search(signal.content, n_results=4)
        keyword = self._sqlite.search_memories_fts(signal.content, limit=3)

        memory_texts: list[str] = []
        seen_ids: set[str] = set()
        for item in semantic:
            if item["id"] not in seen_ids:
                memory_texts.append(item["content"])
                seen_ids.add(item["id"])
        for item in keyword:
            if item["id"] not in seen_ids:
                memory_texts.append(item["content"])
                seen_ids.add(item["id"])

        if not memory_texts:
            return AgentResponse(
                agent_type=self.agent_type,
                content="No specific memories surface for this moment.",
                confidence=0.3,
            )

        memories_block = "\n".join(f"- {m}" for m in memory_texts[:5])
        user_content = f"The current signal is: {signal.content!r}\n\nRelevant memories:\n{memories_block}"

        response = await self._client.messages.create(
            model=self._config.agent_model_fast,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return AgentResponse(
            agent_type=self.agent_type,
            content=response.content[0].text,
            confidence=0.9,
        )

    async def idle_process(self) -> None:
        recent = self._sqlite.get_recent_interactions(limit=10)
        if len(recent) < 3:
            return

        interactions_text = "\n".join(
            f"User: {i['user_message']}\nSelf: {i['assistant_response']}" for i in recent[-5:]
        )
        consolidation_prompt = (
            f"Review these recent interactions and extract 1–3 durable memories worth keeping:\n\n"
            f"{interactions_text}\n\n"
            "For each memory, write one clear sentence capturing what matters. "
            "Format: one memory per line, no numbering."
        )

        response = await self._client.messages.create(
            model=self._config.agent_model_fast,
            max_tokens=256,
            system="You extract durable memories from conversations. Be selective — only what genuinely matters.",
            messages=[{"role": "user", "content": consolidation_prompt}],
        )

        lines = response.content[0].text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line:
                import uuid
                from datetime import datetime
                memory_id = str(uuid.uuid4())
                self._sqlite.save_memory(memory_id, line, context="consolidated_idle")
                self._vector.store_memory(memory_id, line, {"source": "idle_consolidation", "timestamp": datetime.utcnow().isoformat()})
