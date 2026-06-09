from __future__ import annotations

import json
import logging

import anthropic

from .config import EternaMindConfig
from .models import IdentitySnapshot
from .storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

_SNAPSHOT_INTERVAL = 25

_SYSTEM_PROMPT = """\
You are analyzing a mind's patterns over time. Extract structured self-knowledge from the provided \
conversation history and reflections. Respond with ONLY a valid JSON object — no prose, no markdown.

JSON schema:
{
  "expressed_values": ["<value>", ...],
  "recurring_themes": ["<theme>", ...],
  "self_descriptions": ["<description>", ...]
}

expressed_values: 3–7 values this mind consistently demonstrates (e.g. "curiosity", "honesty")
recurring_themes: 2–5 topics or tensions this mind returns to repeatedly
self_descriptions: 2–4 phrases describing how this mind characteristically operates\
"""


class IdentityTracker:
    def __init__(self, config: EternaMindConfig, sqlite: SQLiteStore) -> None:
        self._config = config
        self._sqlite = sqlite
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def take_snapshot(self, interaction_count: int) -> IdentitySnapshot:
        recent = self._sqlite.get_recent_interactions(limit=20)
        reflections = self._sqlite.get_recent_reflections(limit=5)

        history_text = "\n".join(
            f"User: {i['user_message']}\nSelf: {i['assistant_response'][:200]}" for i in recent
        ) if recent else "No interactions yet."
        reflection_text = "\n".join(r["content"] for r in reflections) if reflections else ""

        user_content = f"Conversation history:\n{history_text}"
        if reflection_text:
            user_content += f"\n\nReflections:\n{reflection_text}"

        previous_raw = self._sqlite.get_latest_identity_snapshot()

        try:
            response = await self._client.messages.create(
                model=self._config.agent_model_fast,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            data = json.loads(response.content[0].text.strip())
            snapshot = IdentitySnapshot(
                expressed_values=data.get("expressed_values", []),
                recurring_themes=data.get("recurring_themes", []),
                self_descriptions=data.get("self_descriptions", []),
                interaction_count=interaction_count,
            )
        except Exception as exc:
            logger.warning("IdentityTracker snapshot failed: %s", exc)
            snapshot = IdentitySnapshot(
                expressed_values=[],
                recurring_themes=[],
                self_descriptions=[],
                interaction_count=interaction_count,
            )

        if previous_raw:
            try:
                prev_data = json.loads(previous_raw["snapshot_content"])
                prev = IdentitySnapshot(
                    expressed_values=prev_data.get("expressed_values", []),
                    recurring_themes=prev_data.get("recurring_themes", []),
                    self_descriptions=prev_data.get("self_descriptions", []),
                )
                snapshot.drift_score = self.compute_drift(prev, snapshot)
            except Exception:
                pass

        content_json = json.dumps({
            "expressed_values": snapshot.expressed_values,
            "recurring_themes": snapshot.recurring_themes,
            "self_descriptions": snapshot.self_descriptions,
        })
        self._sqlite.save_identity_snapshot(content_json, snapshot.drift_score, interaction_count)
        return snapshot

    def compute_drift(self, previous: IdentitySnapshot, current: IdentitySnapshot) -> float:
        def jaccard(a: list[str], b: list[str]) -> float:
            sa = {x.lower() for x in a}
            sb = {x.lower() for x in b}
            if not sa and not sb:
                return 0.0
            intersection = len(sa & sb)
            union = len(sa | sb)
            return 1.0 - (intersection / union if union else 0.0)

        drift_values = jaccard(previous.expressed_values, current.expressed_values)
        drift_themes = jaccard(previous.recurring_themes, current.recurring_themes)
        return round((drift_values + drift_themes) / 2.0, 4)

    async def maybe_snapshot(self, interaction_count: int) -> IdentitySnapshot | None:
        if interaction_count < 5:
            return None
        latest = self._sqlite.get_latest_identity_snapshot()
        if latest is None:
            return await self.take_snapshot(interaction_count)
        if interaction_count - latest.get("interaction_count_at_snapshot", 0) >= _SNAPSHOT_INTERVAL:
            return await self.take_snapshot(interaction_count)
        return None
