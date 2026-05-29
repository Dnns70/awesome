from __future__ import annotations

import re

from .models import AgentType, CognitiveSignal

# Patterns that activate the conditional agents
_REFLECTIVE_PATTERNS = re.compile(
    r"\b(why|feel|think|believe|understand|wonder|curious|confus|struggle|meaning|purpose|pattern|myself|self|mind|conscious|aware)\b",
    re.IGNORECASE,
)
_CREATIVE_PATTERNS = re.compile(
    r"\b(imagine|what if|could|possible|idea|create|design|invent|explore|novel|new|different|alternative|suppose|hypothet)\b",
    re.IGNORECASE,
)
_SOCIAL_PATTERNS = re.compile(
    r"\b(you|your|we|us|together|help|want|need|feel|think|like|love|hate|frustrat|excit|happy|sad|angry|nervous|worry)\b",
    re.IGNORECASE,
)

# Core agents always run every turn
_CORE_AGENTS: frozenset[AgentType] = frozenset({
    AgentType.MEMORY,
    AgentType.TEMPORAL,
    AgentType.PERCEPTUAL,
    AgentType.PREDICTIVE,
})


class TransparentGatingMechanism:
    """Routes a CognitiveSignal to the relevant set of agents."""

    def route(self, signal: CognitiveSignal) -> frozenset[AgentType]:
        active = set(_CORE_AGENTS)
        text = signal.content

        if _REFLECTIVE_PATTERNS.search(text):
            active.add(AgentType.REFLECTIVE)
        if _CREATIVE_PATTERNS.search(text):
            active.add(AgentType.CREATIVE)
        if _SOCIAL_PATTERNS.search(text):
            active.add(AgentType.SOCIAL)

        # Short inputs (greetings, acknowledgements) skip heavy conditional agents
        if len(text.split()) <= 4 and AgentType.REFLECTIVE in active:
            active.discard(AgentType.CREATIVE)

        return frozenset(active)
