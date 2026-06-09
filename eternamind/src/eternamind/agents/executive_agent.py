from __future__ import annotations

import json
import logging

import anthropic

from ..config import EternaMindConfig
from ..models import AgentType, CognitiveSignal, ExecutiveEvaluation, UnifiedExperience

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Executive Control agent within a unified cognitive system. \
You receive all active cognitive contributions for the current moment and evaluate their quality and relevance. \
You must respond with ONLY a valid JSON object — no prose, no markdown, no explanation.

JSON schema:
{
  "weights": {
    "<agent_name>": <float 0.0-1.0>
  },
  "summary": "<2-3 sentence weighted synthesis of the most relevant contributions>",
  "emphasis_order": ["<agent_name>", ...],
  "reasoning": "<1 sentence internal justification>"
}

Agent names: memory, temporal, perceptual, reflective, creative, social, predictive
weights: 0.0 = irrelevant/noise, 1.0 = directly relevant to the current moment
emphasis_order: list agent names from most to least relevant for this signal
summary: synthesize the high-weight contributions into a unified executive perspective\
"""

_ORDERED_TYPES = [
    AgentType.MEMORY,
    AgentType.TEMPORAL,
    AgentType.PERCEPTUAL,
    AgentType.REFLECTIVE,
    AgentType.CREATIVE,
    AgentType.SOCIAL,
    AgentType.PREDICTIVE,
]


class ExecutiveAgent:
    """Meta-evaluator — runs after all agents complete, not in the main gather."""

    def __init__(self, config: EternaMindConfig) -> None:
        self._config = config
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def evaluate(
        self,
        experience: UnifiedExperience,
        signal: CognitiveSignal,
    ) -> ExecutiveEvaluation:
        contributions_text = "\n".join(
            f"[{r.agent_type.value.upper()}] {r.content}"
            for r in experience.agent_contributions
            if r.content and not r.content.startswith("[agent error")
        )
        if not contributions_text:
            return _fallback_evaluation(experience)

        user_content = (
            f"Current signal: {signal.content!r}\n\n"
            f"Agent contributions:\n{contributions_text}"
        )

        try:
            response = await self._client.messages.create(
                model=self._config.agent_model_fast,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)
            return _parse_evaluation(data, experience)
        except Exception as exc:
            logger.warning("ExecutiveAgent evaluation failed: %s", exc)
            return _fallback_evaluation(experience)


def _parse_evaluation(data: dict, experience: UnifiedExperience) -> ExecutiveEvaluation:
    raw_weights: dict = data.get("weights", {})
    weights: dict[AgentType, float] = {}
    for name, val in raw_weights.items():
        try:
            weights[AgentType(name)] = float(val)
        except (ValueError, KeyError):
            pass

    raw_order: list = data.get("emphasis_order", [])
    emphasis_order: list[AgentType] = []
    for name in raw_order:
        try:
            emphasis_order.append(AgentType(name))
        except ValueError:
            pass

    # Fall back to natural order for any agents not listed
    present = {r.agent_type for r in experience.agent_contributions if r.content}
    for t in _ORDERED_TYPES:
        if t in present and t not in emphasis_order:
            emphasis_order.append(t)

    return ExecutiveEvaluation(
        weights=weights,
        summary=str(data.get("summary", "")),
        emphasis_order=emphasis_order,
        reasoning=str(data.get("reasoning", "")),
    )


def _fallback_evaluation(experience: UnifiedExperience) -> ExecutiveEvaluation:
    present = [r.agent_type for r in experience.agent_contributions if r.content]
    return ExecutiveEvaluation(
        weights={t: 1.0 for t in present},
        summary="",
        emphasis_order=[t for t in _ORDERED_TYPES if t in present],
    )
