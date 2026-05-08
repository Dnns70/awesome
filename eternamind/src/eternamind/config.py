import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EternaMindConfig:
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    ccm_model: str = "claude-opus-4-7"
    agent_model_mid: str = "claude-sonnet-4-6"
    agent_model_fast: str = "claude-haiku-4-5"
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("ETERNAMIND_DATA_DIR", Path.home() / ".eternamind")))
    idle_cycle_seconds: float = float(os.environ.get("ETERNAMIND_IDLE_CYCLE_SECONDS", "30"))
    max_conversation_history: int = int(os.environ.get("ETERNAMIND_MAX_CONVERSATION_HISTORY", "50"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "eternamind.db"

    @property
    def vector_db_path(self) -> str:
        return str(self.data_dir / "chroma")

    def validate(self) -> None:
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.data_dir.mkdir(parents=True, exist_ok=True)
