from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AgentResponse, AgentType, CognitiveSignal


class BaseAgent(ABC):
    agent_type: AgentType
    _tuning_instructions: str = ""

    @abstractmethod
    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        """Process a cognitive signal and return a response."""

    async def idle_process(self) -> None:
        """Called during idle cycles for background cognition. Override as needed."""

    def set_tuning_instructions(self, instructions: str) -> None:
        self._tuning_instructions = instructions

    def _get_system_prompt(self, base_prompt: str) -> str:
        if self._tuning_instructions:
            return base_prompt + f"\n\n[SELF-TUNING NOTE] {self._tuning_instructions}"
        return base_prompt
