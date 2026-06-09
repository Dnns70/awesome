from __future__ import annotations

import json
import logging

import anthropic

from .config import EternaMindConfig
from .models import AgentType, UnifiedExperience
from .storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are scoring how much each cognitive agent's contribution was reflected in a final response. \
Respond with ONLY a valid JSON object — no prose, no markdown.

JSON schema:
{
  "<agent_name>": {"score": <int 0-10>, "reason": "<one phrase>"},
  ...
}

Score guide:
0 = completely ignored, not referenced at all
3 = acknowledged implicitly but not engaged with
5 = partially reflected, some overlap
8 = clearly drawn upon, paraphrased or addressed
10 = central to the response, directly addressed

Only score agents that have non-empty contributions in the input.\
"""

_TUNING_THRESHOLDS = [
    (0.3, "be more concise and directly relevant — your contributions are being overlooked"),
    (0.5, "focus on the most immediately relevant aspect — stay closely tied to the current signal"),
    (0.7, "good relevance; maintain your current approach"),
    (0.9, "excellent contributions; your depth is well-received — continue"),
]


class AgentScorer:
    def __init__(self, config: EternaMindConfig, sqlite: SQLiteStore) -> None:
        self._config = config
        self._sqlite = sqlite
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def score_contributions(
        self,
        experience: UnifiedExperience,
        ccm_response: str,
        interaction_id: int,
    ) -> dict[AgentType, float]:
        contributions: list[str] = []
        for r in experience.agent_contributions:
            if r.content and not r.content.startswith("[agent error"):
                contributions.append(f"[{r.agent_type.value.upper()}] {r.content}")

        if not contributions:
            return {}

        user_content = (
            "Agent contributions:\n"
            + "\n".join(contributions)
            + f"\n\nFinal response:\n{ccm_response}"
        )

        try:
            response = await self._client.messages.create(
                model=self._config.agent_model_fast,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            data: dict = json.loads(response.content[0].text.strip())
        except Exception as exc:
            logger.warning("AgentScorer scoring failed: %s", exc)
            return {}

        scores: dict[AgentType, float] = {}
        for name, val in data.items():
            try:
                agent_type = AgentType(name.lower())
                raw_score = float(val.get("score", 5)) if isinstance(val, dict) else float(val)
                normalized = max(0.0, min(1.0, raw_score / 10.0))
                reason = val.get("reason", "") if isinstance(val, dict) else ""
                scores[agent_type] = normalized
                self._sqlite.save_agent_score(agent_type.value, interaction_id, normalized, reason)
            except (ValueError, KeyError, TypeError):
                pass

        return scores

    def get_tuning_instructions(self, agent_type: AgentType) -> str:
        avg = self._sqlite.get_average_agent_score(agent_type.value)
        for threshold, instruction in _TUNING_THRESHOLDS:
            if avg <= threshold:
                return instruction
        return _TUNING_THRESHOLDS[-1][1]
