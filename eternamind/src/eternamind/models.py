from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AgentType(str, Enum):
    MEMORY = "memory"
    TEMPORAL = "temporal"
    PERCEPTUAL = "perceptual"
    REFLECTIVE = "reflective"
    CREATIVE = "creative"
    SOCIAL = "social"


@dataclass
class CognitiveSignal:
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    agent_type: AgentType
    content: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedExperience:
    primary_content: str
    agent_contributions: list[AgentResponse]
    temporal_context: str = ""
    perceptual_context: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_contribution(self, agent_type: AgentType) -> str:
        for r in self.agent_contributions:
            if r.agent_type == agent_type:
                return r.content
        return ""


@dataclass
class Memory:
    content: str
    context: str = ""
    importance: float = 0.5
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    embedding: list[float] = field(default_factory=list)
