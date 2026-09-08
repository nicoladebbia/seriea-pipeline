"""Clock-controlled and mutation-shaped tests on the money path (reliability
phase 11, 2026-09-06). Before this, 12 of 1,634 tests controlled time; the
T-30 stage window, the odds cache clock, the budget pacing and the monthly
reset had never been exercised at a frozen clock, and the journal dedup and
the atomic writer had only ever been tested in their steady state.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# 1. The T-30 stage window at a frozen clock (scheduler._stages_due)
# ---------------------------------------------------------------------------
NOW = datetime(2026, 9, 13, 16, 15, tzinfo=UTC)


def _match(minutes_to_kickoff: int, key: str = "Inter vs Milan", date: str = "2026-09-13"):
    return {"match": key, "kickoff_utc": NOW + timedelta(minutes=minutes_to_kickoff), "date": date,
            "home_team": "Inter", "away_team": "Milan", "league": "serie_a"}


def _due(matches, processed=None):
    from scripts.pipeline.scheduler import _stages_due
    actions, processed = _stages_due(matches, processed or {}, NOW, "2026-09-13")
    return {k: [m["match"] for m in v] for k, v in actions.items() if v}, processed


def test_prediction_update_fires_at_t30_and_only_once():
    fired, processed = _due([_match(30)])
    assert "Inter vs Milan" in fired["prediction_update"]
    assert "Inter vs Milan" in fired["lineup_fetch"]          # 5..58 window, retried each cycle
    assert "odds_T5m" not in fired and "player_props_T60" not in fired
    marker = processed["Inter vs Milan"]["stages"]["prediction_update"]
    assert marker["minutes_until_kickoff"] == 30 and marker["triggered_at"] == NOW.isoformat()
    # the mutation the markers exist to survive: the same tick evaluated again
    again, _ = _due([_match(30)], processed)
    assert "prediction_update" not in again, "fired twice for one match"


def test_the_window_edges_are_inclusive_and_a_missed_window_never_fires_late():
    fired, _ = _due([_match(45)])
    assert "Inter vs Milan" in fired["prediction_update"]     # upper edge inclusive
    fired, _ = _due([_match(10)])
    assert "Inter vs Milan" in fired["prediction_update"]     # lower edge inclusive
    fired, _ = _due([_match(46)])
    assert "prediction_update" not in fired                    # too early
    fired, _ = _due([_match(9)])
    assert "prediction_update" not in fired                    # window passed: no late commit


def test_a_stage_marked_needs_retry_refires_but_a_done_stage_does_not():
    _, processed = _due([_match(30)])
    processed["Inter vs Milan"]["stages"]["lineup_fetch"]["needs_retry"] = True
    fired, _ = _due([_match(30)], processed)
    assert "Inter vs Milan" in fired["lineup_fetch"]
    assert "prediction_update" not in fired


def test_the_same_fixture_on_another_date_is_a_new_match():
    """Serie A fixtures repeat every season; the marker is scoped by the match's
    own date (the journal dedup was date-blind until 2026-09-05 — same trap)."""
    _, processed = _due([_match(30, date="2026-03-01")])
    processed["Inter vs Milan"]["date"] = "2026-03-01"
    fired, processed = _due([_match(30, date="2026-09-13")], processed)
    assert "Inter vs Milan" in fired["prediction_update"]
    assert processed["Inter vs Milan"]["date"] == "2026-09-13"


def test_settlement_check_fires_after_kickoff_not_before():
    fired, _ = _due([_match(-120)])
    assert "Inter vs Milan" in fired["settlement_check"]
    assert "prediction_update" not in fired
    fired, _ = _due([_match(-100)])
    assert "settlement_check" not in fired


# ---------------------------------------------------------------------------
# 2. The odds cache clock (odds_fetcher._is_cache_valid)
# ---------------------------------------------------------------------------
@pytest.fixture
def frozen_odds_clock(monkeypatch):
    from scripts.data import odds_fetcher as of
    fixed = datetime(2026, 9, 13, 18, 0, tzinfo=UTC)
    monkeypatch.setattr(of, "now_utc", lambda: fixed)
    monkeypatch.setattr(of, "now_local", lambda: fixed.astimezone())
    return of, fixed


def _cache(tmp_path, cached_at):
    p = tmp_path / "cache.json"
    p.write_text(json.dumps({"cached_at": cached_at, "data": []}))
    return p


def test_cache_expires_exactly_on_the_configured_age(frozen_odds_clock, tmp_path):
    of, fixed = frozen_odds_clock
    fresh = (fixed - timedelta(minutes=of.CACHE_DURATION_MINUTES - 1)).isoformat()
    stale = (fixed - timedelta(minutes=of.CACHE_DURATION_MINUTES + 1)).isoformat()
    assert of._is_cache_valid(_cache(tmp_path, fresh)) is True
    assert of._is_cache_valid(_cache(tmp_path, stale)) is False


def test_a_naive_cache_stamp_is_read_as_utc_and_garbage_is_stale(frozen_odds_clock, tmp_path):
    of, fixed = frozen_odds_clock
    naive_fresh = (fixed - timedelta(minutes=5)).replace(tzinfo=None).isoformat()
    assert of._is_cache_valid(_cache(tmp_path, naive_fresh)) is True
    assert of._is_cache_valid(_cache(tmp_path, "not a date")) is False
    assert of._is_cache_valid(tmp_path / "missing.json") is False


# ---------------------------------------------------------------------------
# 3. Budget pacing and the monthly reset at a frozen calendar
# ---------------------------------------------------------------------------
def _pace(monkeypatch, day: int, used_frac: float, *, year=2026, month=9):
    """Freeze the calendar at `day` with `used_frac` of MONTHLY_LIMIT spent this month."""
    from scripts.data import odds_fetcher as of
    month_used = int(of.MONTHLY_LIMIT * used_frac)
    fixed = datetime(year, month, day, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(of, "now_local", lambda: fixed)
    monkeypatch.setattr(of, "now_utc", lambda: fixed)
    monkeypatch.setattr(of, "check_quota_budget", lambda critical=False: (True, "ok"))
    monkeypatch.setattr(of, "_load_usage", lambda: {"monthly_calls": {f"{year}-{month:02d}": month_used},
                                                    "daily_calls": {}, "last_call": None})
    return of


def test_pacing_drops_the_lowest_priority_first_when_ahead_of_schedule(monkeypatch):
    """PRIORITY_TOLERANCE is a floor on slack: extras survive 2% over schedule,
    props must be exactly on pace, backfill needs 20% UNDER. The bottom drops first."""
    of = _pace(monkeypatch, day=15, used_frac=0.51)
    # day 15 of 30 = 50% of the month, 51% spent: slack -1%
    assert of.check_budget_pacing(priority=of.PRIORITY_EXTRAS)[0] is True       # tol +2%
    assert of.check_budget_pacing(priority=of.PRIORITY_PROPS)[0] is False       # tol 0
    assert of.check_budget_pacing(priority=of.PRIORITY_BACKFILL)[0] is False    # needs slack >= +20%
    assert of.check_budget_pacing(priority=of.PRIORITY_BACKFILL, critical=True)[0] is True
    of = _pace(monkeypatch, day=15, used_frac=0.60)
    ok, msg = of.check_budget_pacing(priority=of.PRIORITY_EXTRAS)
    assert ok is False and "slack=-10.0%" in msg


def test_pacing_allows_everything_when_under_schedule(monkeypatch):
    of = _pace(monkeypatch, day=20, used_frac=0.30)
    for prio in of.PRIORITY_TOLERANCE:
        assert of.check_budget_pacing(priority=prio)[0] is True


def test_monthly_reset_rolls_the_year_on_december_31(monkeypatch):
    of = _pace(monkeypatch, day=31, used_frac=0.0, month=12)
    summary = of.get_usage_summary()
    assert summary["days_to_reset"] == 0
    of = _pace(monkeypatch, day=1, used_frac=0.0, month=12)
    assert of.get_usage_summary()["days_to_reset"] == 30   # Dec 1 12:00 -> Jan 1 00:00 = 30 whole days
    of = _pace(monkeypatch, day=1, used_frac=0.0, month=2)
    assert of.get_usage_summary()["days_to_reset"] == 27   # Feb 2026: 28 days


# ---------------------------------------------------------------------------
# 4. Journal dedup — insert in the middle, never only append
# ---------------------------------------------------------------------------
def _bet(**over):
    base = {"match": "Inter vs Milan", "date": "2026-09-13", "market": "O/U 1.5", "selection": "Over 1.5",
            "model_prob": 0.8, "sharp_implied_prob": 0.7, "edge_pct": 5.0, "odds": 1.4, "bookmaker": "Pinnacle",
            "stake": 10.0, "confidence": "MEDIUM", "placed_at": "2026-09-13T15:45:00+00:00"}
    base.update(over)
    return base


def test_market_scoped_dedup_keeps_three_markets_and_blocks_the_true_duplicate(tmp_path, monkeypatch):
    from scripts.betting import bet_journal as bj
    monkeypatch.setattr(bj, "JOURNAL_PATH", tmp_path / "bet_journal.json")
    a = bj.add_bet(_bet(market="1X2", selection="1"), dedup_by_market=True)
    b = bj.add_bet(_bet(market="1H 1X2", selection="1"), dedup_by_market=True)      # same selection text, other market
    c = bj.add_bet(_bet(market="O/U 1.5", selection="Over 1.5"), dedup_by_market=True)
    assert len({a, b, c}) == 3
    assert bj.add_bet(_bet(market="1H 1X2", selection="1"), dedup_by_market=True) == b   # the real duplicate
    # the same fixture on another date is a new bet, even without market scoping
    d = bj.add_bet(_bet(date="2027-02-01", placed_at="2027-02-01T15:45:00+00:00"))
    assert d not in {a, b, c}
    # without market scoping the market-name-variant guard still holds for one fixture+date
    assert bj.add_bet(_bet(market="OU_1.5", selection="Over 1.5")) == c


# ---------------------------------------------------------------------------
# 5. The atomic writer when the rename itself fails
# ---------------------------------------------------------------------------
def test_atomic_write_survives_a_failed_rename_with_the_old_file_intact(tmp_path, monkeypatch):
    from pathlib import Path

    from config.settings import atomic_write_json
    target = tmp_path / "state.json"
    target.write_text('{"balance": 1000}')
    real_rename = Path.rename

    def boom(self, dst):
        if str(dst) == str(target):
            raise OSError("disk full")
        return real_rename(self, dst)

    monkeypatch.setattr(Path, "rename", boom)
    with pytest.raises(OSError):
        atomic_write_json(target, {"balance": 0})
    assert json.loads(target.read_text()) == {"balance": 1000}
    assert list(tmp_path.glob("*.tmp")) == [], "temp file left behind"


# ---------------------------------------------------------------------------
# 6. The incumbent stake ladder at the bar's own boundary
# ---------------------------------------------------------------------------
def _real(n_won, n_lost, odds=1.41, placed="2026-09-13T17:00:00+00:00", clv=(1.2, 3.4, 0.8, 2.9),
          move=(0.9, -0.2, 0.6, 0.8)):
    out = []
    for i in range(n_won + n_lost):
        won = i < n_won
        out.append({"market": "O/U 1.5", "selection": "Over 1.5", "status": "won" if won else "lost",
                    "stake": 10.0, "odds": odds, "profit": round(10.0 * (odds - 1), 2) if won else -10.0,
                    "placed_at": placed, "extra": None, "pipeline_status": "current",
                    # varying, because a constant series has no t-statistic
                    "clv_pct": None if clv is None else clv[i % len(clv)],
                    "clv_move_pct": None if move is None else move[i % len(move)]})
    return out


def test_full_stake_unlocks_at_thirty_held_up_bets_not_twenty_nine():
    from scripts.betting import market_promotion as MP
    n = MP.INCUMBENT_FULL_STAKE_MIN_N
    assert n == MP.DEMOTION_BAR["min_real_bets"] == 30, "the ladder keeps its own count"
    short = MP.incumbent_records(_real(22, n - 1 - 22))["ou_over_1_5"]
    full = MP.incumbent_records(_real(22, n - 22))["ou_over_1_5"]
    assert short["real_since_live"]["n"] == n - 1
    assert short["stake_scale"] == MP.PROMOTED_KELLY_SCALE and f"{n - 1}/{n}" in short["stake_reason"]
    assert full["real_since_live"]["n"] == n
    assert full["stake_scale"] == 1.0 and "bar cleared" in full["stake_reason"]
    # a full count that trips the demotion bar stays on the half stake, however
    # far the line moved our way — demotion is checked before the quality leg
    bad = MP.incumbent_records(_real(15, 15))["ou_over_1_5"]
    assert bad["real_since_live"]["clv_move_z"] > MP.INCUMBENT_FULL_STAKE_BAR["min_clv_move_z"]
    assert bad["stake_scale"] == MP.PROMOTED_KELLY_SCALE and "demotion bar" in bad["stake_reason"]


def test_the_quality_leg_is_the_line_MOVING_not_beating_the_close():
    """The ladder has read the wrong leg four times. This pins the current one
    from both sides: a record whose RETURN says nothing (z +0.30, and z >= 2.5
    on the return would need a +28.1% run over 30 bets) but whose sharp line
    moved our way clears; the same record with the money going the other way
    does not, because ROI > 0 is kept as a floor.

    Beat-the-close is NOT the leg, and the third case is why: decomposed on the
    live journal 2026-09-08, +2.07 of O/U 1.5's +2.61% beat-the-close was the
    same-moment spread between our best-of-N price and the one sharp book the
    close is read from — 0 of 48 negative, the engine's entry edge restated.
    A gate on that opens for any market where we shop."""
    from scripts.betting import market_promotion as MP
    good = MP.incumbent_records(_real(22, 8))["ou_over_1_5"]["real_since_live"]
    assert good["z"] < MP.PROMOTION_BAR["min_z"], "precondition: the return-z rule blocks this"
    assert good["clv_move_z"] >= MP.INCUMBENT_FULL_STAKE_BAR["min_clv_move_z"]
    assert MP.full_stake_misses(good) == []

    losing = MP.incumbent_records(_real(21, 9))["ou_over_1_5"]["real_since_live"]
    assert losing["clv_move_z"] >= MP.INCUMBENT_FULL_STAKE_BAR["min_clv_move_z"] and losing["roi_pct"] < 0
    assert MP.full_stake_misses(losing) == ["ROI -1.3%"]

    # the spread-shaped record: beat-the-close is huge and near-deterministic,
    # the line never moved. The old rule cleared it; this one must not.
    spread_only = MP.incumbent_records(
        _real(22, 8, clv=(2.0, 2.1, 2.05, 2.15), move=(0.05, -0.04, 0.03, -0.05))
    )["ou_over_1_5"]["real_since_live"]
    assert spread_only["clv_z"] > 20, "precondition: the beat-the-close rule would clear this"
    assert MP.full_stake_misses(spread_only), "but the line never moved"

    # and a book with no same-book closing prices cannot clear a gate reading them
    blind = MP.incumbent_records(_real(22, 8, clv=None, move=None))["ou_over_1_5"]["real_since_live"]
    assert MP.full_stake_misses(blind) == [
        f"0/{MP.INCUMBENT_FULL_STAKE_BAR['min_clv_move_n']} sharp closing lines"]
