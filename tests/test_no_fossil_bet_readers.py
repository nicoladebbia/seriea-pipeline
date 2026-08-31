"""No live code may read the retired February bet artifacts.

data/betting/unified_report.json froze on 2026-02-09 and data/upcoming/
ultimate_bet_slip.json on 2026-02-12, yet a dozen consumers (dashboard
betting page, best-bets API, Telegram, parlays, AI reasoning, health checks)
kept serving them as current until 2026-08-31. The only bet artifact is
data/upcoming/unified_bet_slip.json. This test fails if anyone re-adds a
reader (or the writer fallback) for the fossils.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOSSIL = re.compile(
    r'"betting"\s*/\s*"unified_report\.json"'
    r'|betting/unified_report\.json'
    r'|unified_report_\{league\}'
    r'|ultimate_bet_slip\.json'
    r'|PIPELINE_RUN_REPORT'
)


def test_no_live_reader_of_fossil_bet_files():
    offenders = []
    for d in ("scripts", "web", "features", "ml", "pipeline"):
        for py in (ROOT / d).rglob("*.py"):
            for i, line in enumerate(py.read_text(errors="ignore").splitlines(), 1):
                if line.strip().startswith("#"):
                    continue  # explanatory comments may name the fossils
                if FOSSIL.search(line):
                    offenders.append(f"{py.relative_to(ROOT)}:{i}: {line.strip()[:90]}")
    assert not offenders, "fossil bet-file references:\n" + "\n".join(offenders)
