"""scraper.referee must never persist an empty per-league cache: on 2026-08-31
worldfootball had not published 2026-27, both caches were written with 0 rows,
and the exists() short-circuit then served 0 rows on every later call."""
import pandas as pd

import scraper.referee as ref


def test_empty_scrape_writes_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ref, "REFEREE_DIR", tmp_path)
    monkeypatch.setattr(ref, "_resolve_wf_league", lambda league: ("ita-serie-a", {"2026-2027": None}))
    monkeypatch.setattr(ref, "get_season_referees", lambda season, league="serie_a": [])
    monkeypatch.setattr(ref.time, "sleep", lambda s: None)
    out = ref.scrape_all_referee_assignments(seasons=["2026-2027"], league="serie_a")
    assert out.empty and not (tmp_path / "referee_assignments_serie_a.parquet").exists()
    # a real frame is still cached and served on the next call
    monkeypatch.setattr(ref, "get_season_referees", lambda season, league="serie_a": [("Davide Massa", "1", "davide-massa")])
    monkeypatch.setattr(ref, "scrape_referee_matches", lambda *a, **k: [
        {"match_date": "2026-08-23", "home_team": "Inter", "away_team": "Torino", "referee": "Davide Massa",
         "matchweek": 1, "ref_yellows": 3, "ref_second_yellows": 0, "ref_reds": 0, "season": "2026-2027"}])
    out = ref.scrape_all_referee_assignments(seasons=["2026-2027"], league="serie_a")
    assert len(out) == 1 and (tmp_path / "referee_assignments_serie_a.parquet").exists()
    assert len(ref.scrape_all_referee_assignments(seasons=["2026-2027"], league="serie_a")) == 1
