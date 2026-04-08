# Examples

Run from the repo root with the package available (editable install recommended):

```bash
pip install -e ".[dev,all]"
python examples/01_basic_remember_recall.py
python examples/02_store_stats.py
python examples/03_recall_with_entry_type.py
python examples/05_multi_agent_shared_memory.py
```

Each script uses a temporary database under the system temp directory.

| Script | What it shows |
|--------|--------------|
| `01_basic_remember_recall.py` | `remember()` and `recall()` basics |
| `02_store_stats.py` | `status()` and entry count introspection |
| `03_recall_with_entry_type.py` | Filtering recall by `entry_type` |
| `05_multi_agent_shared_memory.py` | Three async agents sharing one `MemoryStore`; piggyback recall across agents |
