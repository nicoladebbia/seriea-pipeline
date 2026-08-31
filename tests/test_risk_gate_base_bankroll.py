"""Risk gates must measure P&L against the journal's INITIAL bankroll.

Every live caller of check_risk_gates passes the CURRENT bankroll
(scheduler: get_effective_bankroll(); ledger: metrics.bankroll.current). Until
2026-08-31 that value was used as RiskConfig.starting_bankroll, so
check_bankroll_floor computed current + total_pnl — P&L counted twice
(risk_state.json: €1048.34 on a €1024.17 book, +24.17 settled).
"""

from scripts.betting import risk_controls as rc


def _bets(*profits):
    return [{"status": "won" if p > 0 else "lost", "profit": p,
             "market": "O/U_Over 2.5", "stake": 10.0} for p in profits]


def test_current_bankroll_is_initial_plus_pnl_not_current_plus_pnl(monkeypatch):
    monkeypatch.setattr(rc, "_load_settled_bets", lambda: _bets(50.0, -25.83))
    monkeypatch.setattr("scripts.betting.ledger.get_initial_bankroll", lambda: 1000.0)

    # caller passes the CURRENT bankroll, exactly like scheduler.py does
    gates = rc.check_risk_gates(bankroll=1024.17)

    floor = gates["checks"]["bankroll_floor"]
    assert floor["current_bankroll"] == 1024.17          # bug version: 1048.34
    assert floor["floor"] == 500.0                        # 50% of INITIAL, not of current
    # 2 settled bets is below the drawdown check's minimum sample; the floor
    # check is the one that carried the double count, so it is the assertion.


def test_explicit_cfg_is_respected(monkeypatch):
    monkeypatch.setattr(rc, "_load_settled_bets", lambda: _bets(10.0))
    cfg = rc.RiskConfig(starting_bankroll=2000.0)
    gates = rc.check_risk_gates(bankroll=1.0, cfg=cfg)
    assert gates["checks"]["bankroll_floor"]["current_bankroll"] == 2010.0


def test_unreadable_journal_falls_back_to_config_default(monkeypatch):
    monkeypatch.setattr(rc, "_load_settled_bets", lambda: _bets(10.0))

    def boom():
        raise OSError("no journal")

    monkeypatch.setattr("scripts.betting.ledger.get_initial_bankroll", boom)
    gates = rc.check_risk_gates(bankroll=5000.0)
    default = rc.RiskConfig().starting_bankroll
    assert gates["checks"]["bankroll_floor"]["current_bankroll"] == round(default + 10.0, 2)
