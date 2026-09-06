"""A season written into a path literal is a time bomb with an annual fuse.

Three instances paid for (CLAUDE.md): config.settings.SEASONS lagging the
calendar, two fixture filenames in web/app.py frozen on last season, and the
promoted-teams table in betting_unified that "goes stale every August". This
guard fails the suite in the exact week the fuse would otherwise burn silently.
"""
import re
from pathlib import Path

from config.settings import get_current_season

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"tests", "data", "logs", ".git", ".claude", ".plans", "node_modules", "legacy", ".venv", "venv"}

# A season literal inside a string that names a file. Backtest artifacts for a
# FINISHED season are legitimately fixed; each allowed site names its reason.
_PATH_LITERAL = re.compile(r'["\'][^"\']*20\d{2}[_-]20\d{2}[^"\']*\.(?:json|parquet|csv|html)["\']')
ALLOWED_PATH_LITERALS = {
    "web/app.py": "backtest_2023-2024 / backtest_2024-2025 are finished-season artifacts",
    "scripts/diagnostics/epl_loss_pattern.py": "one-shot diagnostic pinned to the season it analysed",
    "scripts/diagnostics/epl_rich_features_test.py": "one-shot diagnostic pinned to the season it analysed",
}


def _py_files():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        yield rel, path


def test_no_season_literal_in_a_path_outside_the_allowlist():
    offenders = []
    for rel, path in _py_files():
        if str(rel) in ALLOWED_PATH_LITERALS:
            continue
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if _PATH_LITERAL.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{rel}:{i}: {line.strip()[:90]}")
    assert not offenders, (
        "season literal in a path — derive it from config.settings.get_current_season() "
        "(or the one helper that already builds that filename):\n  " + "\n  ".join(offenders)
    )


def test_season_keyed_tables_on_the_betting_path_cover_the_current_season():
    """The promoted-teams tables are hand-maintained per season; when the key for
    the season being played is missing the engine silently treats every promoted
    side as established. Fail loudly instead, in the week it happens."""
    import scripts.betting.betting_unified as bu
    src = Path(bu.__file__).read_text()
    season = get_current_season()
    for name in ("_PROMOTED_TEAMS", "_EPL_PROMOTED_TEAMS"):
        block = src[src.index(f"{name} = {{"):]
        block = block[:block.index("\n    }\n")]
        assert f'"{season}"' in block, f"{name} has no entry for the current season {season}"
