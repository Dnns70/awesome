from __future__ import annotations

import asyncio
from typing import Callable, Coroutine, Any

from .agents.base import BaseAgent
from .models import AgentResponse, AgentType, CognitiveSignal, UnifiedExperience

_AGENT_LABELS: dict[AgentType, str] = {
    AgentType.MEMORY: "MEMORY",
    AgentType.TEMPORAL: "TEMPORAL",
    AgentType.PERCEPTUAL: "PERCEPTUAL",
    AgentType.REFLECTIVE: "REFLECTIVE",
    AgentType.CREATIVE: "CREATIVE",
    AgentType.SOCIAL: "SOCIAL",
}

_ORDERED_TYPES = [
    AgentType.MEMORY,
    AgentType.TEMPORAL,
    AgentType.PERCEPTUAL,
    AgentType.REFLECTIVE,
    AgentType.CREATIVE,
    AgentType.SOCIAL,
]


class IntegrationLayer:
    def __init__(self, agents: dict[AgentType, BaseAgent]) -> None:
        self._agents = agents

    async def process(
        self,
        signal: CognitiveSignal,
        active_agents: frozenset[AgentType],
    ) -> UnifiedExperience:
        tasks: dict[AgentType, Coroutine[Any, Any, AgentResponse]] = {
            agent_type: self._agents[agent_type].process(signal)
            for agent_type in active_agents
            if agent_type in self._agents
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        contributions: list[AgentResponse] = []
        for agent_type, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                contributions.append(AgentResponse(
                    agent_type=agent_type,
                    content=f"[agent error: {type(result).__name__}]",
                    confidence=0.0,
                ))
            else:
                contributions.append(result)

        temporal = next((r.content for r in contributions if r.agent_type == AgentType.TEMPORAL), "")
        perceptual = next((r.content for r in contributions if r.agent_type == AgentType.PERCEPTUAL), "")

        return UnifiedExperience(
            primary_content=signal.content,
            agent_contributions=contributions,
            temporal_context=temporal,
            perceptual_context=perceptual,
        )

    def format_cognitive_block(self, experience: UnifiedExperience, user_input: str) -> str:
        lines = ["[COGNITIVE INTEGRATION — CURRENT AWARENESS]"]

        for agent_type in _ORDERED_TYPES:
            contribution = experience.get_contribution(agent_type)
            if contribution:
                label = _AGENT_LABELS[agent_type]
                lines.append(f"[{label}] {contribution}")

        lines.append("[END OF INTEGRATION — WHAT FOLLOWS IS WHAT THE PERSON IS SAYING]")
        lines.append(user_input)
        return "\n".join(lines)
