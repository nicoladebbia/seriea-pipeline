"""Post-round verifier for the T-60 anytime-scorer chain (2026-09-03).

Answers one question after kickoff: did the whole chain run — scheduler
stage -> odds fetch -> edges build -> advisor tilts -> ledger snapshot —
and pushes the per-check verdict to Telegram/macOS so a silent failure is
impossible to miss. Reusable every round; a one-shot launchd job fires the
first check (Genoa-Como, 2026-09-04) and removes itself. All human-facing
times are LOCAL (America/New_York — Miami, per Nicola 2026-09-03), never
CET; the artifacts themselves stay UTC.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "fantacalcio" / "scorer_odds_raw.json"
EDGES = ROOT / "data" / "fantacalcio" / "scorer_edges.json"
LEDGER = ROOT / "data" / "fantacalcio" / "pred_ledger.json"
SCHED_LOG = ROOT / "logs" / "launchd-pre-kickoff-monitor-err.log"
SELF_PLIST = (Path.home() / "Library" / "LaunchAgents"
              / "com.seriea-pipeline.scorer-check-once.plist")


def _local(iso: str | None) -> str:
    """UTC ISO -> local wall time (the machine runs America/New_York)."""
    if not iso:
        return "?"
    try:
        return (datetime.fromisoformat(iso.replace("Z", "+00:00"))
                .astimezone().strftime("%H:%M %Z"))
    except ValueError:
        return iso


def run_check() -> tuple[bool, str]:
    checks: list[tuple[bool, str]] = []
    raw = {}
    try:
        raw = json.loads(RAW.read_text())
    except (OSError, ValueError):
        pass
    events = list((raw.get("events") or {}).values())
    ev = max(events, key=lambda e: e.get("fetched_at") or "", default=None)
    if ev is None:
        checks.append((False, "nessun fetch scorer (scorer_odds_raw.json vuoto)"))
    else:
        try:
            ko = datetime.fromisoformat(ev["commence"].replace("Z", "+00:00"))
            ft = datetime.fromisoformat(ev["fetched_at"])
            lead_min = (ko - ft).total_seconds() / 60
        except (KeyError, ValueError):
            ko, lead_min = None, None
        tag = f"{ev.get('home')}-{ev.get('away')}"
        if lead_min is None:
            checks.append((False, f"{tag}: timestamp fetch illeggibile"))
        elif 0 <= lead_min <= 90:
            checks.append((True, f"{tag}: odds prese a T-{lead_min:.0f}min "
                                 f"({_local(ev['fetched_at'])}, "
                                 f"{len(ev.get('prices') or {})} giocatori)"))
        else:
            checks.append((False, f"{tag}: fetch a T-{lead_min:.0f}min — fuori "
                                  f"finestra ({_local(ev.get('fetched_at'))})"))
    try:
        edges = json.loads(EDGES.read_text())
        n = len(edges.get("by_pid") or {})
        fresh = ev and edges.get("built_at", "") >= ev.get("fetched_at", "")
        checks.append((bool(n >= 15 and fresh),
                       f"edges: {n} giocatori, build {_local(edges.get('built_at'))}"
                       + ("" if fresh else " (PIU VECCHIO del fetch)")))
    except (OSError, ValueError):
        checks.append((False, "scorer_edges.json assente/illeggibile"))
    try:
        led = json.loads(LEDGER.read_text())
        rows = [r for snap in (led.get("rounds") or {}).values()
                for r in (snap.get("players") or [])] or \
               [r for r in led.get("players", [])]
        tilted = [r for r in rows if r.get("scorer_edge") is not None]
        checks.append((bool(tilted),
                       f"ledger: {len(tilted)} righe con scorer_edge"))
    except (OSError, ValueError):
        checks.append((False, "pred_ledger.json assente — snapshot non scattato?"))
    try:
        lines = [ln for ln in SCHED_LOG.read_text().splitlines()[-400:]
                 if "Fanta scorer props" in ln]
        checks.append((bool(lines), "scheduler: " +
                       (lines[-1].split("Fanta scorer props:")[-1].strip()
                        if lines else "nessuna riga 'Fanta scorer props' nel log")))
    except OSError:
        checks.append((False, "log pre-kickoff-monitor illeggibile"))

    ok = all(c[0] for c in checks)
    body = "\n".join(("OK " if c else "FAIL ") + m for c, m in checks)
    return ok, body


def main() -> None:
    once = "--once" in sys.argv
    ok, body = run_check()
    verdict = ("Catena scorer T-60: TUTTO OK" if ok
               else "Catena scorer T-60: QUALCOSA NON HA GIRATO")
    stamp = datetime.now(UTC).astimezone().strftime("%a %H:%M %Z")
    try:
        from scripts.pipeline.notify import notify
        notify(f"{verdict} ({stamp})\n{body}",
               title="Fanta scorer check",
               level="success" if ok else "warning", category="system",
               tg_html=(f"<b>{'✅' if ok else '⚠️'} {verdict}</b> ({stamp})\n"
                        + "\n".join(("✅ " if c else "❌ ") + m
                                    for c, m in
                                    [(ln.startswith("OK "), ln.split(" ", 1)[1])
                                     for ln in body.splitlines()])))
    except Exception as e:  # noqa: BLE001 — the check must still print
        print(f"notify failed: {e}")
    print(verdict)
    print(body)
    if once and SELF_PLIST.exists():
        # Order matters: `launchctl unload` SIGTERMs THIS process, so on
        # 2026-09-04 the unlink below it never ran and buffered stdout was
        # lost (empty logs, plist left behind). Delete + flush FIRST; dying
        # inside the final unload is then harmless.
        SELF_PLIST.unlink(missing_ok=True)
        print("one-shot plist removed")
        sys.stdout.flush()
        subprocess.run(  # noqa: S603 — fixed argv, own label
            ["/bin/launchctl", "remove", "com.seriea-pipeline.scorer-check-once"],
            capture_output=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
