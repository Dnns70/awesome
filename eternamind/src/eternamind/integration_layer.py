from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Coroutine

from .agents.base import BaseAgent
from .models import AgentResponse, AgentType, CognitiveSignal, UnifiedExperience

if TYPE_CHECKING:
    from .agents.executive_agent import ExecutiveAgent
    from .storage.sqlite_store import SQLiteStore

_AGENT_LABELS: dict[AgentType, str] = {
    AgentType.MEMORY: "MEMORY",
    AgentType.TEMPORAL: "TEMPORAL",
    AgentType.PERCEPTUAL: "PERCEPTUAL",
    AgentType.REFLECTIVE: "REFLECTIVE",
    AgentType.CREATIVE: "CREATIVE",
    AgentType.SOCIAL: "SOCIAL",
    AgentType.PREDICTIVE: "PREDICTIVE",
}

_ORDERED_TYPES = [
    AgentType.MEMORY,
    AgentType.TEMPORAL,
    AgentType.PERCEPTUAL,
    AgentType.REFLECTIVE,
    AgentType.CREATIVE,
    AgentType.SOCIAL,
    AgentType.PREDICTIVE,
]


class IntegrationLayer:
    def __init__(
        self,
        agents: dict[AgentType, BaseAgent],
        executive_agent: "ExecutiveAgent | None" = None,
        sqlite: "SQLiteStore | None" = None,
    ) -> None:
        self._agents = agents
        self._executive_agent = executive_agent
        self._sqlite = sqlite

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

        experience = UnifiedExperience(
            primary_content=signal.content,
            agent_contributions=contributions,
            temporal_context=temporal,
            perceptual_context=perceptual,
        )

        if self._executive_agent is not None:
            evaluation = await self._executive_agent.evaluate(experience, signal)
            experience.executive_evaluation = evaluation

        return experience

    def format_cognitive_block(self, experience: UnifiedExperience, user_input: str) -> str:
        lines = ["[COGNITIVE INTEGRATION — CURRENT AWARENESS]"]

        # Determine display order from executive evaluation (if available)
        if experience.executive_evaluation and experience.executive_evaluation.emphasis_order:
            order = experience.executive_evaluation.emphasis_order
            weights = experience.executive_evaluation.weights
        else:
            order = _ORDERED_TYPES
            weights = {}

        for agent_type in order:
            contribution = experience.get_contribution(agent_type)
            if not contribution or contribution.startswith("[agent error"):
                continue
            label = _AGENT_LABELS.get(agent_type, agent_type.value.upper())
            weight = weights.get(agent_type, 1.0)
            prefix = "[PRIMARY] " if weight >= 0.8 else ""
            lines.append(f"[{label}] {prefix}{contribution}")

        # Inject active goals if storage is available
        if self._sqlite is not None:
            active_goals = self._sqlite.get_active_goals(limit=5)
            if active_goals:
                goal_lines = []
                for g in active_goals:
                    last_note = g["progress_notes"].split("\n")[-1] if g["progress_notes"] else ""
                    note_suffix = f" — {last_note}" if last_note else ""
                    goal_lines.append(f"  • [P{g['priority']}] {g['title']}{note_suffix}")
                lines.append("[ACTIVE GOALS]\n" + "\n".join(goal_lines))

        # Executive synthesis just before the handoff marker
        if experience.executive_evaluation and experience.executive_evaluation.summary:
            lines.append(f"[EXECUTIVE SYNTHESIS] {experience.executive_evaluation.summary}")

        lines.append("[END OF INTEGRATION — WHAT FOLLOWS IS WHAT THE PERSON IS SAYING]")
        lines.append(user_input)
        return "\n".join(lines)
