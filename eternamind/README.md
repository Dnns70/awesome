# EternaMind

A continuous, unified cognitive entity — an implementation of the EternaMind theoretical architecture.

## Setup

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

pip install -e .
```

## Usage

```bash
# Start a conversation
eternamind chat

# Check system status
eternamind status
```

## Architecture

EternaMind implements five core components:

- **Central Consciousness Model (CCM)** — Claude Opus 4.7 with adaptive thinking; the unified mind
- **Agent Substrate** — Six specialized cognitive agents running in parallel
- **Transparent Gating Mechanism (TGM)** — Routes signals to relevant agents in <100ms
- **Continuous Cognition Loop (CCL)** — Background asyncio loop; active and idle states
- **Integration Layer** — Synthesizes agent outputs into a unified cognitive context

### The Six Agents

| Agent | Model | Role |
|-------|-------|------|
| Memory | Haiku 4.5 | Surfaces relevant recollections |
| Temporal | Haiku 4.5 | Conveys felt sense of time |
| Perceptual | psutil (no LLM) | System environment awareness |
| Reflective | Sonnet 4.6 | Self-examination and insight |
| Creative | Sonnet 4.6 | Novel connections at knowledge boundaries |
| Social | Haiku 4.5 | Models the user's state and intent |
