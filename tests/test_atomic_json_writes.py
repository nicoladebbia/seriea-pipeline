"""Every JSON state file is written atomically — guard + mutation test.

On 2026-09-06 the repo had 163 `json.dump(` writers and 0 atomic ones outside
config.settings.atomic_write_json (47 callers). bet_journal.json — the immutable
source of truth for money — was written with a bare json.dump; a crash or a
concurrent launchd job mid-write truncates it. The guard below fails the suite
the day a bare json.dump to a file path comes back.
"""
import json
import re
from pathlib import Path

import pytest

from config.settings import atomic_write_json

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"tests", "data", "logs", ".git", ".claude", ".plans", "node_modules", "legacy", ".venv", "venv"}

# Sites allowed to call json.dump directly. Each entry names the reason; a
# writer to stdout / a pipe / a caller-owned handle is not a state file.
ALLOWED_JSON_DUMP_SITES = {
    "config/settings.py": "the atomic writer itself",
    "scripts/prediction/predict_unified.py": "print(json.dumps()) to stdout, not a file",
    "scripts/prediction/component_ledger.py": "print(json.dumps()) to stdout, not a file",
    "scripts/data/odds_fetcher.py": "L1024 writes gzip-compressed JSON via gzip.open(path, 'wt'); atomic_write_json has no gzip support",
}


def _py_files():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        yield rel, path


def test_no_bare_json_dump_outside_the_allowlist():
    offenders = []
    pat = re.compile(r"\bjson\.dump\(")
    for rel, path in _py_files():
        if str(rel) in ALLOWED_JSON_DUMP_SITES:
            continue
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if pat.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{rel}:{i}")
    assert not offenders, (
        f"{len(offenders)} bare json.dump( sites — use config.settings.atomic_write_json "
        f"or add a justified ALLOWED_JSON_DUMP_SITES entry:\n  " + "\n  ".join(offenders[:40])
    )


def test_atomic_write_keeps_the_old_file_when_serialisation_fails(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"balance": 1000.0})
    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()}, default=None)  # no default -> TypeError
    assert json.loads(target.read_text()) == {"balance": 1000.0}
    assert list(tmp_path.glob("*.tmp")) == [], "temp file left behind"


def test_atomic_write_passes_dump_kwargs_through(tmp_path):
    target = tmp_path / "x.json"
    atomic_write_json(target, {"b": "è", "a": 1}, ensure_ascii=False, sort_keys=True)
    assert target.read_text() == '{\n  "a": 1,\n  "b": "è"\n}'
    from datetime import datetime
    atomic_write_json(target, {"t": datetime(2026, 9, 6)})          # default=str fallback
    assert json.loads(target.read_text()) == {"t": "2026-09-06 00:00:00"}
    atomic_write_json(target, {"t": datetime(2026, 9, 6)}, default=lambda o: o.isoformat())
    assert json.loads(target.read_text()) == {"t": "2026-09-06T00:00:00"}


def test_atomic_write_creates_parent_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c.json"
    atomic_write_json(target, [1])
    assert json.loads(target.read_text()) == [1]
