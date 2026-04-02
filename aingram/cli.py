# aingram/cli.py
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help='AIngram — local-first agent memory CLI', no_args_is_help=True)


@app.callback()
def cli_options(
    ctx: typer.Context,
    db: Path = typer.Option(
        Path('agent_memory.db'),
        '--db',
        help='Path to the SQLite memory database',
    ),
    no_telemetry: bool = typer.Option(
        False,
        '--no-telemetry',
        help='Disable anonymous usage telemetry for this invocation',
    ),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj['db'] = str(db.expanduser().resolve())
    ctx.obj['no_telemetry'] = no_telemetry

    def _flush_telemetry() -> None:
        from aingram.config import load_merged_config
        from aingram.telemetry import maybe_send_cli_telemetry

        cfg = load_merged_config()
        enabled = cfg.telemetry_enabled and not no_telemetry
        maybe_send_cli_telemetry(command=ctx.invoked_subcommand, enabled=enabled)

    ctx.call_on_close(_flush_telemetry)


@app.command()
def setup() -> None:
    """Download embedding assets and report cache location."""
    from aingram.models.manager import ModelManager

    mm = ModelManager()
    typer.echo(f'Models directory: {mm.cache_dir}')
    typer.echo('Run a short Python snippet or use `aingram add` to trigger downloads on first use.')


@app.command('status')
def cmd_status(ctx: typer.Context) -> None:
    """Show database stats and capabilities."""
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        typer.echo(json.dumps(mem.stats, indent=2))
    finally:
        mem.close()


@app.command()
def add(ctx: typer.Context, text: str = typer.Argument(..., help='Memory text to store')) -> None:
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        entry_id = mem.remember(text)
        typer.echo(entry_id)
    finally:
        mem.close()


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(...),
    limit: int = typer.Option(10, '--limit', '-n'),
) -> None:
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        for r in mem.recall(query, limit=limit, verify=False):
            typer.echo(f'{r.score:.4f}\t{r.entry.entry_id[:16]}\t{r.entry.content[:120]}')
    finally:
        mem.close()


@app.command()
def entities(
    ctx: typer.Context,
    limit: int = typer.Option(50, '--limit', '-n'),
) -> None:
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        for e in mem._engine.get_entities(limit=limit):
            typer.echo(f'{e.entity_id}\t{e.entity_type}\t{e.name}')
    finally:
        mem.close()


@app.command()
def graph(
    ctx: typer.Context,
    name: str = typer.Argument(..., help='Entity name to expand'),
) -> None:
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        matches = mem._engine.find_entities_by_name(name)
        if not matches:
            typer.echo('No entities matched.')
            raise typer.Exit(code=1)
        for e in matches:
            typer.echo(f'{e.name} ({e.entity_id}) [{e.entity_type}]')
            for r in mem._engine.get_relationships_for_entity(e.entity_id):
                other = r.target_id if r.source_id == e.entity_id else r.source_id
                typer.echo(f'  {r.relation_type}\t{r.fact or ""}\t<-> {other}')
    finally:
        mem.close()


@app.command()
def consolidate(ctx: typer.Context) -> None:
    from dataclasses import asdict

    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        result = mem.consolidate()
        typer.echo(json.dumps(asdict(result), indent=2))
    finally:
        mem.close()


@app.command()
def compact(
    ctx: typer.Context,
    yes: bool = typer.Option(False, '--yes', help='Confirm one-way embedding truncation'),
    target_dim: int = typer.Option(256, '--target-dim', help='Truncated embedding size'),
) -> None:
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        try:
            mem.compact(confirm=yes, target_dim=target_dim)
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=1) from e
        typer.echo(f'Compacted embeddings to {target_dim} dimensions.')
    finally:
        mem.close()


@app.command()
def watch(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, '--json', help='Output as JSONL'),
) -> None:
    """Live tail of new memory entries."""
    from aingram.watch import watch_loop

    watch_loop(ctx.obj['db'], json_output=json_output)


@app.command()
def viz(
    ctx: typer.Context,
    port: int = typer.Option(8420, '--port', help='Server port'),
    no_open: bool = typer.Option(False, '--no-open', help='Do not open browser'),
) -> None:
    """Start local visualization server."""
    from aingram.viz.server import run_viz

    run_viz(ctx.obj['db'], port=port, open_browser=not no_open)


@app.command()
def quantize(
    ctx: typer.Context,
    yes: bool = typer.Option(False, '--yes', help='Confirm one-way uint8 quantization'),
    rebuild: bool = typer.Option(False, '--rebuild', help='Rebuild float32 cache from int8 data'),
) -> None:
    """Quantize embeddings to uint8 for ~4x storage reduction."""
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        if rebuild:
            count = mem._engine.rebuild_vec_from_int8()
            typer.echo(f'Rebuilt {count} float32 embeddings from int8 source.')
        else:
            try:
                mem.quantize(confirm=yes)
            except ValueError as e:
                typer.echo(str(e), err=True)
                raise typer.Exit(code=1) from e
            typer.echo('Embeddings quantized to uint8.')
    finally:
        mem.close()


agent_app = typer.Typer(help='Agent token management', no_args_is_help=True)
app.add_typer(agent_app, name='agent')


@agent_app.command('create')
def agent_create(
    ctx: typer.Context,
    name: str = typer.Argument(..., help='Agent name (unique)'),
    role: str = typer.Option('contributor', '--role', help='reader, contributor, or admin'),
    pubkey: str | None = typer.Option(None, '--pubkey', help='Ed25519 public key hex'),
) -> None:
    """Create an agent and print its bearer token (shown once)."""
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        result = mem._engine.create_agent_token(agent_name=name, role=role, public_key=pubkey)
        typer.echo(f'Agent created: {name} (role={role})')
        typer.echo(f'Token: {result["token"]}')
        typer.echo('Save this token — it will not be shown again.')
    finally:
        mem.close()


@agent_app.command('list')
def agent_list(ctx: typer.Context) -> None:
    """List all agents and their roles."""
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        for a in mem._engine.list_agent_tokens():
            status = 'REVOKED' if a['revoked_at'] else 'active'
            typer.echo(f'{a["agent_name"]}\t{a["role"]}\t{status}')
    finally:
        mem.close()


@agent_app.command('revoke')
def agent_revoke(
    ctx: typer.Context,
    name: str = typer.Argument(..., help='Agent name to revoke'),
) -> None:
    """Revoke an agent token (takes effect on next MCP call)."""
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        mem._engine.revoke_agent_token(name)
        typer.echo(f'Agent {name!r} revoked.')
    finally:
        mem.close()


@app.command()
def export(
    ctx: typer.Context,
    out: Path = typer.Argument(..., help='Output JSON path'),
    agent_id: str | None = typer.Option(None, '--agent', help='Filter memories by agent_id'),
) -> None:
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        mem.export_json(out, agent_id=agent_id)
        typer.echo(f'Wrote {out}')
    finally:
        mem.close()


@app.command('import')
def import_backup(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help='JSON file from aingram export'),
    merge: bool = typer.Option(False, '--merge', help='Allow importing into a non-empty database'),
) -> None:
    from aingram import MemoryStore

    mem = MemoryStore(ctx.obj['db'])
    try:
        mem.import_json(path, merge=merge)
        typer.echo(f'Imported {path}')
    finally:
        mem.close()


def main_entry() -> None:
    app()


if __name__ == '__main__':
    main_entry()
