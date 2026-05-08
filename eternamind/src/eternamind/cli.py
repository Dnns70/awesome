from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from .config import EternaMindConfig
from .engine import EternaMindEngine

app = typer.Typer(
    name="eternamind",
    help="EternaMind — a continuous, unified cognitive entity.",
    add_completion=False,
)
console = Console()


def _boot_banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]E T E R N A M I N D[/bold cyan]\n"
        "[dim]a continuous, unified cognitive entity[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def _build_engine() -> EternaMindEngine:
    config = EternaMindConfig()
    try:
        config.validate()
    except ValueError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)
    return EternaMindEngine(config)


async def _chat_loop(engine: EternaMindEngine) -> None:
    _boot_banner()
    console.print("[dim]Type your message and press Enter. Type [bold]/quit[/bold] or [bold]/exit[/bold] to leave.[/dim]")
    console.print("[dim]Type [bold]/status[/bold] to see system state.[/dim]")
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

    reflections = sqlite.get_recent_reflections(limit=1)
    last_reflection = reflections[0]["content"][:80] + "…" if reflections else "None"
    table.add_row("Last reflection", last_reflection)

    social = sqlite.get_latest_social_model()
    user_model = social[:80] + "…" if social else "None"
    table.add_row("Current user model", user_model)

    console.print()
    console.print(table)
    console.print()


@app.command()
def chat(
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d", help="Override data directory"),
) -> None:
    """Start an interactive conversation with EternaMind."""
    config = EternaMindConfig()
    if data_dir:
        from pathlib import Path
        config.data_dir = Path(data_dir)

    try:
        config.validate()
    except ValueError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)

    engine = EternaMindEngine(config)
    asyncio.run(_chat_loop(engine))


@app.command()
def status(
    data_dir: Optional[str] = typer.Option(None, "--data-dir", "-d", help="Override data directory"),
) -> None:
    """Show EternaMind's current memory and state."""
    config = EternaMindConfig()
    if data_dir:
        from pathlib import Path
        config.data_dir = Path(data_dir)

    try:
        config.validate()
    except ValueError as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)

    engine = EternaMindEngine(config)
    _boot_banner()
    _print_status(engine)


if __name__ == "__main__":
    app()
