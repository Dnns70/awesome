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
    PREDICTIVE = "predictive"
    EXECUTIVE = "executive"


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
class ExecutiveEvaluation:
    weights: dict[AgentType, float]
    summary: str
    emphasis_order: list[AgentType]
    reasoning: str = ""


@dataclass
class UnifiedExperience:
    primary_content: str
    agent_contributions: list[AgentResponse]
    temporal_context: str = ""
    perceptual_context: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    executive_evaluation: ExecutiveEvaluation | None = None

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


@dataclass
class Goal:
    title: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: int = 5
    status: str = "active"
    progress_notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "user"


@dataclass
class IdentitySnapshot:
    expressed_values: list[str]
    recurring_themes: list[str]
    self_descriptions: list[str]
    drift_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    interaction_count: int = 0
