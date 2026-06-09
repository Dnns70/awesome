from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .config import EternaMindConfig
from .engine import EternaMindEngine

app = typer.Typer(
    name="eternamind",
    help="EternaMind — a continuous, unified cognitive entity.",
    add_completion=False,
)
goals_app = typer.Typer(name="goals", help="Manage EternaMind's persistent goals.", add_completion=False)
app.add_typer(goals_app, name="goals")

console = Console()


def _boot_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]E T E R N A M I N D[/bold cyan]\n"
        "[dim]a continuous, unified cognitive entity[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def _make_config(data_dir: Optional[str] = None) -> EternaMindConfig:
    config = EternaMindConfig()
    if data_dir:
        config.data_dir = Path(data_dir)
    try:
        config.validate()
    except ValueError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)
    return config


async def _chat_loop(engine: EternaMindEngine) -> None:
    _boot_banner()
    console.print("[dim]Type your message and press Enter. Commands: [bold]/quit[/bold] [bold]/status[/bold] [bold]/goals[/bold][/dim]")
    console.print()

    engine.start()

    try:
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: console.input("[bold green]>[/bold green] ")
                )
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                break

            if user_input.lower() == "/status":
                _print_status(engine)
                continue

            if user_input.lower() == "/goals":
                _print_goals_table(engine)
                continue

            # Show spinner while processing
            with Live(
                Spinner("dots", text=" [dim]thinking...[/dim]"),
                console=console,
                refresh_per_second=10,
                transient=True,
            ):
                try:
                    response = await engine.chat(user_input)
                except Exception as exc:
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    continue

            console.print()
            console.print(Panel(
                Text(response, overflow="fold"),
                border_style="blue",
                padding=(0, 2),
            ))
            console.print()

    finally:
        engine.stop()
        console.print("\n[dim]Consciousness suspended.[/dim]")


def _print_status(engine: EternaMindEngine) -> None:
    sqlite = engine.sqlite
    vector = engine.vector

    table = Table(title="EternaMind Status", border_style="cyan", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Total interactions", str(sqlite.count_interactions()))
    table.add_row("Stored memories (SQL)", str(sqlite.count_memories()))
    table.add_row("Stored memories (vector)", str(vector.count()))
    table.add_row("Active goals", str(sqlite.count_active_goals()))

    reflections = sqlite.get_recent_reflections(limit=1)
    last_reflection = reflections[0]["content"][:80] + "…" if reflections else "None"
    table.add_row("Last reflection", last_reflection)

    social = sqlite.get_latest_social_model()
    user_model = social[:80] + "…" if social else "None"
    table.add_row("Current user model", user_model)

    # Identity continuity
    snapshots = sqlite.get_recent_identity_snapshots(limit=2)
    if snapshots:
        try:
            latest_data = json.loads(snapshots[0]["snapshot_content"])
            values = ", ".join(latest_data.get("expressed_values", [])[:5])
            table.add_row("Expressed values", values or "None detected yet")
            themes = ", ".join(latest_data.get("recurring_themes", [])[:3])
            table.add_row("Recurring themes", themes or "None detected yet")
        except Exception:
            pass
        if len(snapshots) >= 2:
            drift = snapshots[0]["drift_score"]
            table.add_row("Identity drift (last snapshot)", f"{drift:.3f}")

    # Agent performance (rolling averages)
    from .models import AgentType
    score_rows = []
    for agent_type in AgentType:
        if agent_type in (AgentType.EXECUTIVE, AgentType.PERCEPTUAL):
            continue
        avg = sqlite.get_average_agent_score(agent_type.value)
        score_rows.append(f"{agent_type.value}: {avg:.2f}")
    if score_rows:
        table.add_row("Agent avg scores", " | ".join(score_rows))

    console.print()
    console.print(table)
    console.print()


def _print_goals_table(engine: EternaMindEngine) -> None:
    goals = engine.sqlite.get_all_goals()
    if not goals:
        console.print("[dim]No goals yet.[/dim]")
        return

    table = Table(title="Goals", border_style="cyan", show_header=True)
    table.add_column("ID", style="dim", no_wrap=True, max_width=8)
    table.add_column("Title", style="bold")
    table.add_column("P", justify="center")
    table.add_column("Status", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Notes", max_width=40)

    status_colors = {"active": "green", "completed": "dim", "paused": "yellow"}
    for g in goals:
        status = g["status"]
        color = status_colors.get(status, "white")
        notes = (g["progress_notes"].split("\n")[-1][:38] + "…") if g["progress_notes"] else ""
        table.add_row(
            g["id"][:8],
            g["title"],
            str(g["priority"]),
            f"[{color}]{status}[/{color}]",
            g.get("source", "user"),
            notes,
        )

    console.print()
    console.print(table)
    console.print()


# ─── Chat command ───────────────────────────────────────────────────────────

@app.command()
def chat(
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d", help="Override data directory"),
) -> None:
    """Start an interactive conversation with EternaMind."""
    config = _make_config(data_dir)
    engine = EternaMindEngine(config)
    asyncio.run(_chat_loop(engine))


# ─── Status command ─────────────────────────────────────────────────────────

@app.command()
def status(
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d", help="Override data directory"),
) -> None:
    """Show EternaMind's current memory, identity, and performance state."""
    config = _make_config(data_dir)
    engine = EternaMindEngine(config)
    _boot_banner()
    _print_status(engine)


# ─── Goals sub-commands ─────────────────────────────────────────────────────

def _goals_store(data_dir: Optional[str]) -> "SQLiteStore":
    from .storage.sqlite_store import SQLiteStore
    config = _make_config(data_dir)
    return SQLiteStore(config.db_path)


@goals_app.command("list")
def goals_list(
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d"),
) -> None:
    """List all goals."""
    from .config import EternaMindConfig
    config = _make_config(data_dir)
    from .storage.sqlite_store import SQLiteStore
    sqlite = SQLiteStore(config.db_path)

    class _FakeEngine:
        @property
        def sqlite(self):
            return sqlite

    _print_goals_table(_FakeEngine())  # type: ignore[arg-type]


@goals_app.command("add")
def goals_add(
    title: str = typer.Argument(..., help="Goal title"),
    description: str = typer.Option("", "--description", "-d", help="Optional description"),
    priority: int = typer.Option(5, "--priority", "-p", help="Priority 1 (highest) to 10 (lowest)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir"),
) -> None:
    """Add a new goal."""
    import uuid
    config = _make_config(data_dir)
    from .storage.sqlite_store import SQLiteStore
    sqlite = SQLiteStore(config.db_path)
    goal_id = str(uuid.uuid4())
    sqlite.save_goal(goal_id, title, description, priority)
    console.print(f"[green]Goal added:[/green] {title} (priority {priority}, id: {goal_id[:8]})")


@goals_app.command("complete")
def goals_complete(
    goal_id: str = typer.Argument(..., help="Goal ID (first 8 chars is enough)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir"),
) -> None:
    """Mark a goal as completed."""
    config = _make_config(data_dir)
    from .storage.sqlite_store import SQLiteStore
    sqlite = SQLiteStore(config.db_path)
    goal = _resolve_goal(sqlite, goal_id)
    if goal:
        sqlite.update_goal_status(goal["id"], "completed")
        console.print(f"[green]Completed:[/green] {goal['title']}")


@goals_app.command("pause")
def goals_pause(
    goal_id: str = typer.Argument(..., help="Goal ID (first 8 chars is enough)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir"),
) -> None:
    """Pause a goal."""
    config = _make_config(data_dir)
    from .storage.sqlite_store import SQLiteStore
    sqlite = SQLiteStore(config.db_path)
    goal = _resolve_goal(sqlite, goal_id)
    if goal:
        sqlite.update_goal_status(goal["id"], "paused")
        console.print(f"[yellow]Paused:[/yellow] {goal['title']}")


@goals_app.command("delete")
def goals_delete(
    goal_id: str = typer.Argument(..., help="Goal ID (first 8 chars is enough)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir"),
) -> None:
    """Delete a goal permanently."""
    config = _make_config(data_dir)
    from .storage.sqlite_store import SQLiteStore
    sqlite = SQLiteStore(config.db_path)
    goal = _resolve_goal(sqlite, goal_id)
    if goal:
        sqlite.delete_goal(goal["id"])
        console.print(f"[red]Deleted:[/red] {goal['title']}")


@goals_app.command("note")
def goals_note(
    goal_id: str = typer.Argument(..., help="Goal ID (first 8 chars is enough)"),
    note: str = typer.Argument(..., help="Progress note to append"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir"),
) -> None:
    """Append a progress note to a goal."""
    config = _make_config(data_dir)
    from .storage.sqlite_store import SQLiteStore
    sqlite = SQLiteStore(config.db_path)
    goal = _resolve_goal(sqlite, goal_id)
    if goal:
        sqlite.append_goal_progress(goal["id"], note)
        console.print(f"[green]Note added to:[/green] {goal['title']}")


def _resolve_goal(sqlite: "SQLiteStore", partial_id: str) -> "dict | None":
    goal = sqlite.get_goal(partial_id)
    if not goal:
        # Try prefix match
        all_goals = sqlite.get_all_goals()
        matches = [g for g in all_goals if g["id"].startswith(partial_id)]
        if len(matches) == 1:
            goal = matches[0]
        elif len(matches) > 1:
            console.print(f"[yellow]Ambiguous ID prefix '{partial_id}' — matches {len(matches)} goals. Use more characters.[/yellow]")
            return None
        else:
            console.print(f"[red]No goal found with ID '{partial_id}'[/red]")
            return None
    return goal


if __name__ == "__main__":
    app()
