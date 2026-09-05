"""The scores API reaches 3 days back. A superseded bet (replaced by a later
slip, no stake) older than that can never resolve through the settler: 12 of
them from Feb–Apr re-warned "No result found" every 15-min cycle for five
months. Pending bets keep the warning — an orphaned REAL bet must stay loud."""
import logging

import scripts.data.results_fetcher as rf


def test_unreachable_superseded_bets_are_skipped_but_pending_ones_still_warn(monkeypatch, caplog):
    import scripts.betting.bet_journal as bj
    old_sup = {"bet_id": "2026-03-20_X", "match": "Cagliari vs Napoli", "date": "2026-03-20", "market": "1X2",
               "selection": "1", "odds": 2.0, "stake": 0.0, "status": "superseded"}
    old_pending = {"bet_id": "2026-04-19_Y", "match": "Verona vs Milan", "date": "2026-04-19", "market": "O/U 1.5",
                   "selection": "Over 1.5", "odds": 1.3, "stake": 5.0, "status": "pending"}
    monkeypatch.setattr(bj, "get_pending_bets", lambda *a, **k: [old_sup, old_pending])
    monkeypatch.setattr(rf, "_load_history", lambda: [])
    monkeypatch.setattr(rf, "_save_history", lambda h: None)
    monkeypatch.setattr(rf, "_rebuild_bankroll_from_history", lambda: None)
    monkeypatch.setattr(rf, "_load_bankroll", lambda: {"current_balance": 1000.0, "peak_balance": 1000.0, "lowest_balance": 1000.0})
    with caplog.at_level(logging.DEBUG, logger="scripts.data.results_fetcher"):
        out = rf._settle_bets_locked({})
    assert out["settled"] == 0
    warned = [r.getMessage() for r in caplog.records if "No result found" in r.getMessage()]
    assert any("Verona vs Milan" in w for w in warned)
    assert not any("Cagliari vs Napoli" in w for w in warned)
    assert any("Skipping 1 superseded" in r.getMessage() for r in caplog.records)
