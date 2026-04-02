# aingram/viz/server.py
from __future__ import annotations

import json
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from aingram.storage.engine import StorageEngine

_STATIC_DIR = Path(__file__).parent / 'static'


def _entry_type_str(entry) -> str:
    et = entry.entry_type
    return et.value if hasattr(et, 'value') else str(et)


class VizHandler(SimpleHTTPRequestHandler):
    engine: StorageEngine

    def __init__(self, *args, engine: StorageEngine, **kwargs):
        self.engine = engine
        super().__init__(*args, directory=str(_STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith('/api/'):
            self._handle_api(path, parsed.query)
        else:
            super().do_GET()

    def _handle_api(self, path: str, query_string: str):
        params = parse_qs(query_string)

        if path == '/api/stats':
            self._json_response(VizHandler._get_stats(self.engine))
        elif path == '/api/entities':
            self._json_response(VizHandler._get_entities(self.engine))
        elif path == '/api/chains':
            self._json_response(VizHandler._get_chains(self.engine))
        elif path == '/api/entry':
            entry_id = params.get('id', [None])[0]
            if not entry_id:
                self._json_error(400, 'Missing required parameter: id')
                return
            result = VizHandler._get_entry(self.engine, entry_id)
            if result is None:
                self._json_error(404, f'Entry not found: {entry_id}')
                return
            self._json_response(result)
        else:
            self._json_error(404, f'Unknown API endpoint: {path}')

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status, message):
        self._json_response({'error': message}, status=status)

    @staticmethod
    def _get_stats(engine: StorageEngine) -> dict:
        return {
            'entry_count': engine.get_entry_count(),
            'entity_count': engine.get_entity_count(),
            'chain_count': engine.get_chain_count(),
        }

    @staticmethod
    def _get_entities(engine: StorageEngine) -> dict:
        entities = engine.get_entities(limit=500)
        relationships = engine.list_all_relationships()[:2000]
        nodes = [{'id': e.entity_id, 'name': e.name, 'type': e.entity_type} for e in entities]
        edges = [
            {
                'source': r.source_id,
                'target': r.target_id,
                'type': r.relation_type,
                'weight': r.weight,
            }
            for r in relationships
        ]
        return {'nodes': nodes, 'edges': edges}

    @staticmethod
    def _get_chains(engine: StorageEngine) -> list:
        chains = engine.get_all_chains(limit=100)
        result = []
        for chain in chains:
            entries = engine.get_entries_by_chain(chain.chain_id)
            result.append({
                'chain_id': chain.chain_id,
                'title': chain.title,
                'status': chain.status,
                'created_at': chain.created_at,
                'entries': [
                    {
                        'entry_id': e.entry_id,
                        'type': _entry_type_str(e),
                        'content': e.content,
                        'confidence': e.confidence,
                        'created_at': e.created_at,
                    }
                    for e in entries
                ],
            })
        return result

    @staticmethod
    def _get_entry(engine: StorageEngine, entry_id: str) -> dict | None:
        entry = engine.get_entry(entry_id)
        if entry is None:
            return None

        mentions = engine.get_entities_for_entry(entry_id)

        return {
            'entry_id': entry.entry_id,
            'content_hash': entry.content_hash,
            'entry_type': _entry_type_str(entry),
            'content': entry.content,
            'confidence': entry.confidence,
            'surprise': entry.surprise,
            'importance': entry.importance,
            'created_at': entry.created_at,
            'signature': entry.signature,
            'verified': None,
            'reasoning_chain_id': entry.reasoning_chain_id,
            'entities': [
                {'id': e.entity_id, 'name': e.name, 'type': e.entity_type}
                for e in mentions
            ],
        }

    def log_message(self, format, *args):
        pass


def create_server(
    db_path: str, *, port: int = 8420
) -> tuple[HTTPServer, StorageEngine]:
    engine = StorageEngine(db_path)
    handler = partial(VizHandler, engine=engine)
    server = HTTPServer(('127.0.0.1', port), handler)
    return server, engine


def run_viz(db_path: str, *, port: int = 8420, open_browser: bool = True) -> None:
    server, engine = create_server(db_path, port=port)
    url = f'http://127.0.0.1:{port}'
    print(f'AIngram Viz running at {url}')
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        engine.close()
