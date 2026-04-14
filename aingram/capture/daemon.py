from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from aingram.capture.adapters.aider import AiderAdapter
from aingram.capture.adapters.base import ToolAdapter
from aingram.capture.adapters.chatgpt import ChatGPTAdapter
from aingram.capture.adapters.claude import ClaudeCodeAdapter
from aingram.capture.adapters.cline import ClineAdapter
from aingram.capture.adapters.copilot import CopilotAdapter
from aingram.capture.adapters.cursor import CursorAdapter
from aingram.capture.adapters.gemini import GeminiAdapter
from aingram.capture.config import CaptureConfig
from aingram.capture.queue import CaptureQueue

logger = logging.getLogger(__name__)

_ROUTE_MAP = {
    'claude_code': '/capture/claude-code/hook',
    'cursor': '/capture/cursor/hook',
    'gemini': '/capture/gemini/hook',
    'cline': '/capture/cline/hook',
    'chatgpt': '/capture/chatgpt/import',
}

_COPILOT_ROUTE = '/capture/copilot/otlp/v1/traces'
_AIDER_ROUTE = '/capture/aider/notify'

_ADAPTER_CLASSES: dict[str, type[ToolAdapter]] = {
    'claude_code': ClaudeCodeAdapter,
    'cursor': CursorAdapter,
    'gemini': GeminiAdapter,
    'aider': AiderAdapter,
    'cline': ClineAdapter,
    'copilot': CopilotAdapter,
    'chatgpt': ChatGPTAdapter,
}


def _make_capture_route(adapter: ToolAdapter, queue: CaptureQueue):
    async def handler(request: Request) -> JSONResponse:
        toggle = queue.get_toggle(adapter.tool_name)
        if toggle == 'off':
            return JSONResponse({})

        try:
            raw = await request.json()
        except Exception:
            return JSONResponse({'error': 'invalid JSON'}, status_code=400)
        records = adapter.parse_payload(raw)
        for record in records:
            filtered = adapter.apply_filters(record)
            if filtered:
                queue.insert(filtered)
        return JSONResponse({})

    return handler


def create_app(
    *,
    config: CaptureConfig | None = None,
    queue: CaptureQueue | None = None,
) -> Starlette:
    if config is None:
        config = CaptureConfig()

    if queue is None:
        queue = CaptureQueue(os.path.expanduser(config.queue_db_path))

    adapters: dict[str, ToolAdapter] = {}
    for tool_name, adapter_cls in _ADAPTER_CLASSES.items():
        tool_cfg = config.tools.get(tool_name)
        if tool_cfg and tool_cfg.enabled:
            adapters[tool_name] = adapter_cls(config)

    routes = []

    for tool_name, path in _ROUTE_MAP.items():
        if tool_name in adapters:
            routes.append(
                Route(path, _make_capture_route(adapters[tool_name], queue), methods=['POST'])
            )

    if 'copilot' in adapters:
        routes.append(
            Route(_COPILOT_ROUTE, _make_capture_route(adapters['copilot'], queue), methods=['POST'])
        )

    if 'aider' in adapters:
        routes.append(
            Route(_AIDER_ROUTE, _make_capture_route(adapters['aider'], queue), methods=['POST'])
        )

    async def status_handler(request: Request) -> JSONResponse:
        tools_status = {}
        for name, adapter in adapters.items():
            health = adapter.health_check()
            tools_status[name] = {
                'connected': health.connected,
                'toggle': queue.get_toggle(name),
                'tier': health.tier,
                'last_capture': health.last_capture,
                'error': health.error,
            }
        return JSONResponse(
            {
                'tools': tools_status,
                'queue_depth': queue.pending_count(),
            }
        )

    routes.append(Route('/status', status_handler, methods=['GET']))

    async def toggle_handler(request: Request) -> JSONResponse:
        body = await request.json()
        tool = body.get('tool', '')
        state = body.get('state', '')
        if state not in ('on', 'off'):
            return JSONResponse({'error': 'state must be on or off'}, status_code=400)

        if tool == 'all':
            for name in adapters:
                queue.set_toggle(name, state)
        elif tool in adapters:
            queue.set_toggle(tool, state)
        else:
            return JSONResponse({'error': f'unknown tool: {tool}'}, status_code=400)

        return JSONResponse({'ok': True})

    routes.append(Route('/toggle', toggle_handler, methods=['POST']))

    return Starlette(routes=routes)


def run_daemon(
    *,
    config: CaptureConfig,
    queue: CaptureQueue,
    memory_db_path: str,
    embedder=None,
) -> None:
    import uvicorn

    from aingram.capture.drain import CaptureDrain
    from aingram.config import load_merged_config

    merged = load_merged_config()
    models_dir = merged.models_dir
    if not any(models_dir.glob('*.onnx')):
        logger.warning(
            'ONNX embedding model not cached at %s — run "aingram setup" first. '
            'First drain cycle will block on model download.',
            models_dir,
        )

    app = create_app(config=config, queue=queue)

    from aingram.store import MemoryStore

    store = MemoryStore(memory_db_path, agent_name='capture-daemon', embedder=embedder)

    drain = CaptureDrain(
        queue=queue,
        config=config,
        store=store,
    )
    drain.start()

    worker = None
    if merged.extractor_mode != 'none':
        try:
            from aingram.worker import BackgroundWorker

            extractor = None
            if merged.extractor_mode == 'local':
                from aingram.extraction.local import LocalExtractor

                extractor = LocalExtractor(model=merged.extractor_model, base_url=merged.llm_url)
            elif merged.extractor_mode == 'sonnet':
                from aingram.extraction.sonnet import SonnetExtractor

                extractor = SonnetExtractor(api_key=os.environ.get('ANTHROPIC_API_KEY'))

            if extractor:
                worker = BackgroundWorker(engine=store._engine, extractor=extractor)
                worker.start()
                logger.info('BackgroundWorker started for entity extraction')
        except Exception:
            logger.warning('Failed to start BackgroundWorker', exc_info=True)

    aider_stop = threading.Event()
    aider_thread = None
    aider_cfg = config.tools.get('aider')
    if aider_cfg and aider_cfg.enabled:
        from aingram.capture.config import AiderToolConfig

        if isinstance(aider_cfg, AiderToolConfig):

            def _aider_watcher():
                import glob

                adapters_by_path: dict[str, AiderAdapter] = {}

                def _discover():
                    for watch_dir in aider_cfg.watch_directories:
                        expanded = os.path.expanduser(watch_dir)
                        if not os.path.isdir(expanded):
                            continue
                        pattern = os.path.join(expanded, aider_cfg.history_file_pattern)
                        for path in glob.glob(pattern, recursive=True):
                            if path not in adapters_by_path:
                                a = AiderAdapter(config)
                                a.set_history_path(path)
                                adapters_by_path[path] = a
                                logger.info('Discovered aider history: %s', path)

                _discover()
                cycles = 0

                while not aider_stop.is_set():
                    cycles += 1
                    if cycles % 12 == 0:
                        _discover()
                    for adapter in adapters_by_path.values():
                        try:
                            records = adapter.poll_new_entries()
                            for record in records:
                                filtered = adapter.apply_filters(record)
                                if filtered:
                                    queue.insert(filtered)
                        except Exception:
                            logger.warning('Aider watcher error', exc_info=True)
                    aider_stop.wait(5.0)

            aider_thread = threading.Thread(target=_aider_watcher, daemon=True)
            aider_thread.start()
            logger.info('Aider file watcher started')

    sentinel_path = Path.home() / '.aingram' / 'capture.stop'

    def _sentinel_monitor():
        while not drain._stop.is_set():
            if sentinel_path.exists():
                logger.info('Sentinel file detected, initiating shutdown')
                try:
                    sentinel_path.unlink()
                except OSError:
                    pass
                import _thread

                _thread.interrupt_main()
                return
            drain._stop.wait(1.0)

    sentinel_thread = threading.Thread(target=_sentinel_monitor, daemon=True)
    sentinel_thread.start()

    try:
        uvicorn.run(app, host=config.host, port=config.port, log_level='info')
    finally:
        aider_stop.set()
        if aider_thread:
            aider_thread.join(timeout=3.0)
        if worker:
            worker.stop()
        drain.close()
        store.close()
        pid_file = Path.home() / '.aingram' / 'capture.pid'
        if pid_file.exists():
            try:
                pid_file.unlink()
            except OSError:
                pass
