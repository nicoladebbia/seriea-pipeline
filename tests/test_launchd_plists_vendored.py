"""The 21 launchd jobs are the pipeline's entry points; until 2026-09-06 only 3 of
them existed in the repo (two under config/launchd, one under deploy/) and the
installed copies had been found stripped to bare arrays before (CLAUDE.md,
"All launchd plists look stripped"). Now config/launchd/ is canonical and the
health monitor reports drift between it and ~/Library/LaunchAgents.
"""
import plistlib
from pathlib import Path

from scripts.pipeline.health_check import check_launchd_plists

ROOT = Path(__file__).resolve().parent.parent
REPO_PLISTS = ROOT / "config" / "launchd"


def test_every_vendored_plist_is_valid_xml_with_label_and_program():
    files = sorted(REPO_PLISTS.glob("com.seriea-pipeline.*.plist"))
    assert len(files) >= 21, [f.name for f in files]
    for f in files:
        assert f.read_bytes().lstrip().startswith(b"<"), f"{f.name} is stripped, not XML"
        d = plistlib.loads(f.read_bytes())
        assert d["Label"] == f.stem, f.name
        assert d.get("ProgramArguments"), f.name
        env = d.get("EnvironmentVariables", {})
        for k, v in env.items():
            assert not any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD")), (f.name, k)
            assert not any(t in str(v).upper() for t in ("SK-", "GSK_", "BOT")), (f.name, k)


def _write(dirpath, name, body: bytes):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / name).write_bytes(body)


def test_drift_check_names_missing_stripped_and_changed(tmp_path):
    repo = tmp_path / "repo"
    inst = tmp_path / "installed"
    good = plistlib.dumps({"Label": "com.seriea-pipeline.a", "ProgramArguments": ["python3", "x.py"]})
    _write(repo, "com.seriea-pipeline.a.plist", good)
    _write(repo, "com.seriea-pipeline.b.plist", good.replace(b"pipeline.a", b"pipeline.b"))
    _write(repo, "com.seriea-pipeline.c.plist", good.replace(b"pipeline.a", b"pipeline.c"))
    _write(repo, "com.seriea-pipeline.d.plist", good.replace(b"pipeline.a", b"pipeline.d"))
    _write(inst, "com.seriea-pipeline.a.plist", good)                                  # identical
    _write(inst, "com.seriea-pipeline.b.plist", b'["python3", "x.py"]')                  # stripped
    _write(inst, "com.seriea-pipeline.c.plist", good.replace(b"x.py", b"y.py"))          # changed
    # d missing
    launchctl = "1\t0\tcom.seriea-pipeline.a\n-\t1\tcom.seriea-pipeline.c\n-\t-9\tcom.seriea-pipeline.b\n"
    out = check_launchd_plists(repo_dir=repo, installed_dir=inst, launchctl_output=launchctl)
    assert out["status"] == "WARNING"
    d = out["detail"]
    assert "b: stripped" in d and "c: differs from repo" in d and "d: not installed" in d
    assert "c: last exit 1" in d and "a" not in out.get("problems", [""])[0] if out.get("problems") else True


def test_drift_check_is_ok_when_everything_matches(tmp_path):
    repo = tmp_path / "repo"
    inst = tmp_path / "installed"
    body = plistlib.dumps({"Label": "com.seriea-pipeline.a", "ProgramArguments": ["python3"]})
    _write(repo, "com.seriea-pipeline.a.plist", body)
    _write(inst, "com.seriea-pipeline.a.plist", body)
    out = check_launchd_plists(repo_dir=repo, installed_dir=inst, launchctl_output="123\t0\tcom.seriea-pipeline.a\n")
    assert out["status"] == "OK", out


def test_drift_check_fails_closed_when_launchctl_is_unavailable(tmp_path):
    repo = tmp_path / "repo"
    inst = tmp_path / "installed"
    body = plistlib.dumps({"Label": "com.seriea-pipeline.a", "ProgramArguments": ["python3"]})
    _write(repo, "com.seriea-pipeline.a.plist", body)
    _write(inst, "com.seriea-pipeline.a.plist", body)
    out = check_launchd_plists(repo_dir=repo, installed_dir=inst, launchctl_output=None)
    assert out["status"] == "WARNING" and "launchctl" in out["detail"]
