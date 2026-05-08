from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AgentResponse, AgentType, CognitiveSignal


class BaseAgent(ABC):
    agent_type: AgentType

    @abstractmethod
    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        """Process a cognitive signal and return a response."""

    async def idle_process(self) -> None:
        """Called during idle cycles for background cognition. Override as needed."""
