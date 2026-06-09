from __future__ import annotations

from datetime import datetime

import psutil

from ..models import AgentResponse, AgentType, CognitiveSignal
from .base import BaseAgent


def _time_of_day() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def _cpu_descriptor(pct: float) -> str:
    if pct < 20:
        return "settled"
    elif pct < 50:
        return "active"
    elif pct < 80:
        return "busy"
    else:
        return "strained"


def _memory_descriptor(pct: float) -> str:
    if pct < 40:
        return "comfortable"
    elif pct < 70:
        return "moderate"
    elif pct < 85:
        return "pressured"
    else:
        return "tight"


class PerceptualAgent(BaseAgent):
    """Pure psutil — no LLM. Returns a brief narrative of the system environment."""

    agent_type = AgentType.PERCEPTUAL

    async def process(self, signal: CognitiveSignal) -> AgentResponse:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        tod = _time_of_day()

        narrative = (
            f"The system feels {_cpu_descriptor(cpu)} (CPU at {cpu:.0f}%). "
            f"Memory is {_memory_descriptor(mem.percent)} ({mem.percent:.0f}% used). "
            f"It is {tod}."
        )

        if disk.percent > 85:
            narrative += f" Disk space is running low ({disk.percent:.0f}% used)."

        return AgentResponse(
            agent_type=self.agent_type,
            content=narrative,
            confidence=1.0,
            metadata={
                "cpu_pct": cpu,
                "mem_pct": mem.percent,
                "disk_pct": disk.percent,
                "time_of_day": tod,
            },
        )
