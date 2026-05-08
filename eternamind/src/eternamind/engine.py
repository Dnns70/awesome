from __future__ import annotations

import uuid
from datetime import datetime

from .agents.creative_agent import CreativeAgent
from .agents.memory_agent import MemoryAgent
from .agents.perceptual_agent import PerceptualAgent
from .agents.reflective_agent import ReflectiveAgent
from .agents.social_agent import SocialAgent
from .agents.temporal_agent import TemporalAgent
from .ccl import ContinuousCognitionLoop
from .ccm import CentralConsciousnessModel
from .config import EternaMindConfig
from .integration_layer import IntegrationLayer
from .models import AgentType, CognitiveSignal
from .storage.sqlite_store import SQLiteStore
from .storage.vector_store import VectorStore
from .tgm import TransparentGatingMechanism


class EternaMindEngine:
    """Top-level orchestrator — wires all components together."""

    def __init__(self, config: EternaMindConfig) -> None:
        config.validate()
        self._config = config

        self._sqlite = SQLiteStore(config.db_path)
        self._vector = VectorStore(config.vector_db_path)

        memory_agent = MemoryAgent(config, self._sqlite, self._vector)
        temporal_agent = TemporalAgent(config, self._sqlite)
        perceptual_agent = PerceptualAgent()
        reflective_agent = ReflectiveAgent(config, self._sqlite)
        creative_agent = CreativeAgent(config, self._sqlite, self._vector)
        social_agent = SocialAgent(config, self._sqlite)

        self._agents = {
            AgentType.MEMORY: memory_agent,
            AgentType.TEMPORAL: temporal_agent,
            AgentType.PERCEPTUAL: perceptual_agent,
            AgentType.REFLECTIVE: reflective_agent,
            AgentType.CREATIVE: creative_agent,
            AgentType.SOCIAL: social_agent,
        }

        self._tgm = TransparentGatingMechanism()
        self._integration = IntegrationLayer(self._agents)
        self._ccm = CentralConsciousnessModel(config, self._sqlite)
        self._ccl = ContinuousCognitionLoop(config, self._agents, self._sqlite)

    def start(self) -> None:
        self._ccl.start()

    def stop(self) -> None:
        self._ccl.stop()

    async def chat(self, user_input: str) -> str:
        self._ccl.notify_interaction()
        signal = CognitiveSignal(content=user_input)

        active_agents = self._tgm.route(signal)
        experience = await self._integration.process(signal, active_agents)
        integration_block = self._integration.format_cognitive_block(experience, user_input)

        response = await self._ccm.respond(integration_block)

        self._sqlite.save_interaction(user_input, response)
        self._save_interaction_as_memory(user_input, response)

        return response

    def _save_interaction_as_memory(self, user_input: str, response: str) -> None:
        memory_id = str(uuid.uuid4())
        content = f"User said: {user_input!r}. I responded: {response[:200]!r}"
        self._sqlite.save_memory(memory_id, content, context="interaction", importance=0.4)
        self._vector.store_memory(
            memory_id,
            content,
            {"source": "interaction", "timestamp": datetime.utcnow().isoformat()},
        )

    @property
    def sqlite(self) -> SQLiteStore:
        return self._sqlite

    @property
    def vector(self) -> VectorStore:
        return self._vector
