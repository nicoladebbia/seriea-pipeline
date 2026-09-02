"""Daily off-disk snapshot of the small mutable state that cannot be re-derived.

The bet journal (the immutable source of truth real money settles against),
the fantacalcio league state, the prediction/discipline ledgers and the
monitoring history are all gitignored and live on one SSD. This tars them
(~6 MB) into iCloud Drive so a disk failure costs a day, not the season.
Rotation keeps the newest KEEP archives. Scheduled by
com.seriea-pipeline.state-backup.plist (daily 04:45); freshness is watched by
monitor.check_state_backup via the heartbeat this writes.
"""
from __future__ import annotations

import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    "data/betting",
    "data/fantacalcio",
    "data/monitoring",
    "data/pipeline_state.json",
)
DEFAULT_DEST = (Path.home() / "Library" / "Mobile Documents"
                / "com~apple~CloudDocs" / "seriea-backups")
KEEP = 14
HEARTBEAT = ROOT / "data" / "monitoring" / "state_backup.json"


def run(dest: Path | None = None, keep: int = KEEP,
        targets: tuple[str, ...] = TARGETS, root: Path = ROOT,
        heartbeat: Path = HEARTBEAT) -> Path:
    # explicit dest wins; the env var is the plist/test-level override
    dest = Path(dest or os.environ.get("SERIEA_BACKUP_DEST") or DEFAULT_DEST)
    if not dest.parent.exists():
        # Fail LOUDLY: a backup that silently stops is worse than none,
        # because the monitor heartbeat is the only thing that would notice.
        raise SystemExit(
            f"backup destination parent missing: {dest.parent} "
            "— is iCloud Drive signed in?")
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    out = dest / f"state_{stamp}.tar.gz"
    tmp = out.with_name(out.name + ".part")
    with tarfile.open(tmp, "w:gz") as tf:
        for t in targets:
            p = root / t
            if p.exists():
                tf.add(p, arcname=t)
    tmp.rename(out)
    archives = sorted(dest.glob("state_*.tar.gz"))
    for old in archives[:-keep]:
        old.unlink()
    # written AFTER the tar closes, so each archive carries the PREVIOUS
    # run's heartbeat — cosmetic (nothing reads it from inside the archive)
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.write_text(json.dumps(
        {"ran_at": datetime.now(UTC).isoformat(), "dest": str(out),
         "bytes": out.stat().st_size,
         "kept": min(len(archives), keep)}, indent=1))
    return out


if __name__ == "__main__":
    print(run())
