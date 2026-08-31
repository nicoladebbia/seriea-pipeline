"""_leagues_with_active_matches must read the CURRENT season's fixture files.

It hardcoded fixtures_2025_2026*.json; after the 2026-08-01 rollover those
files froze and the predicate returned "none" on every matchday — zero live
score API calls all season, /live permanently empty. The fix delegates to
match_timing._sofascore_fixture_files() (derived season). This test pins the
delegation: it feeds fixtures through the deriver, so the old hardcoded code
fails it (true positive verified at introduction).
"""

import json
import time


def test_active_leagues_come_from_the_derived_fixture_files(tmp_path, monkeypatch):
    import scripts.utils.match_timing as mt
    from scripts.data import live_monitor as lm

    sa = tmp_path / "fixtures_x.json"
    epl = tmp_path / "fixtures_x_premier_league.json"
    now = time.time()
    sa.write_text(json.dumps([{"startTimestamp": now - 600}]))      # 10 min in — live
    epl.write_text(json.dumps([{"startTimestamp": now + 86400}]))   # tomorrow — not live

    monkeypatch.setattr(
        mt, "_sofascore_fixture_files",
        lambda: [(sa, "serie_a"), (epl, "premier_league")],
    )

    assert lm._leagues_with_active_matches() == {"serie_a"}
