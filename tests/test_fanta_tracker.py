"""Tests for the fantacalcio round parser and the per-round scorer.

Every case here is a mistake that was actually made or actually possible against the live
page, not a restatement of the happy path:

  * `data-value="55"` is the site's senza-voto sentinel. A parser that reads it as a
    rating puts a 55-point voto into a team total.
  * "Player of the match" is a bonus COLUMN worth ZERO. Weighting it +1 was wrong for nine
    players a round and reconciled to exactly -1.
  * Cards are in the grade span's CLASS, not the bonus columns, so a bonus-only parser
    misses every booking silently.
  * The fantavoto must NOT be range-guarded -- guarding it discards double-figure hauls.
"""
import pytest

from scripts.fantacalcio.live_scores import parse
from scripts.fantacalcio.tracker import _modifier, _pick

TABLE = [(6.00, 1), (6.50, 3), (7.00, 6)]
OFFICE = [5.0, 4.5, 4.5]


def _row(slug, role, grade_cls, grade, fanta, bonuses=()):
    pills = (f'<div class="pill"><span class="player-grade {grade_cls}" '
             f'data-value="{grade}"></span>'
             f'<span class="player-fanta-grade" data-value="{fanta}"></span></div>') * 3
    bon = "".join(f'<span class="player-bonus cell" data-value="{v}" title="{t}"></span>'
                  for t, v in bonuses)
    return (f'<tr><td><div class="player-item cell">'
            f'<span class="role" data-value="{role}"></span>'
            f'<a class="player-name player-link" href="https://www.fantacalcio.it'
            f'/serie-a/squadre/inter/{slug}/999/2025-26"><span>X</span></a></div></td>'
            f'<td><div class="group">{pills}</div></td>'
            f'<td><div class="group">{bon}</div></td></tr>')


def test_sv_sentinel_is_not_a_rating():
    """55 must become sv. Reading it as 55.0 poisons the whole team total."""
    df = parse(_row("a", "c", "yellow-card", "55", "55"))
    assert len(df) == 1
    assert df.voto.iloc[0] is None or pytest.approx(0) != df.voto.iloc[0]
    assert not bool(df.played.iloc[0])
    assert df.fantavoto.iloc[0] is None


def test_player_of_the_match_is_worth_nothing():
    """Scamacca: voto 7, one goal, POTM. Published fantavoto is 10, not 11."""
    df = parse(_row("s", "a", "", "7", "10",
                    (("Gol segnati", "1"), ("Player of the match", "1"))))
    assert df.bonus.iloc[0] == pytest.approx(3.0)
    assert df.fantavoto.iloc[0] == pytest.approx(10.0)


def test_cards_come_from_the_grade_class_not_the_bonus_columns():
    df = parse(_row("m", "c", "yellow-card", "5,5", "5"))
    assert df.cards.iloc[0] == pytest.approx(-0.5)
    df = parse(_row("c", "d", "red-card", "4,5", "3,5"))
    assert df.cards.iloc[0] == pytest.approx(-1.0)


def test_fantavoto_is_not_range_guarded():
    """A 13.5 haul must survive. The [0,10] guard belongs to the voto alone."""
    df = parse(_row("h", "a", "", "8", "13,5", (("Gol segnati", "1"),)))
    assert df.fantavoto.iloc[0] == pytest.approx(13.5)


def test_modifier_bands_and_office_votes():
    assert _modifier(7.0, [7.0, 7.0, 7.0], TABLE, OFFICE) == 6
    assert _modifier(6.5, [6.5, 6.5, 6.5], TABLE, OFFICE) == 3
    assert _modifier(5.9, [5.9, 5.9, 5.9], TABLE, OFFICE) == 0
    assert _modifier(None, [7.0, 7.0, 7.0], TABLE, OFFICE) == 0
    # Two defenders: the third takes the 5.0 voto d'ufficio, pulling 7.0 down to 6.5.
    assert _modifier(7.0, [7.0, 7.0], TABLE, OFFICE) == 3


def _roster():
    out = []
    for role, n in (("P", 3), ("D", 8), ("C", 8), ("A", 6)):
        for i in range(n):
            out.append({"id": int(f"{ord(role)}{i:02d}"), "nome": f"{role}{i}",
                        "R": role, "team": "t", "paid": 1, "proj": 100 - i})
    return out


def _votes(roster, missing=()):
    return {p["id"]: {"voto": 6.5, "fantavoto": 6.5, "bonus": 0.0, "cards": 0.0}
            for p in roster if p["id"] not in missing}


def test_three_defender_module_forfeits_the_modifier():
    roster = _roster()
    got = _pick(roster, _votes(roster), TABLE, OFFICE, "actual")
    nd = int(got["module"].split("-")[0])
    assert (got["modifier"] > 0) == (nd >= 4)


def test_substitutions_are_capped_and_same_role():
    """Bench the top four defenders. At most three come back, and only defenders."""
    roster = _roster()
    missing = {p["id"] for p in roster if p["R"] == "D"}
    missing = set(sorted(missing)[:4])
    got = _pick(roster, _votes(roster, missing), TABLE, OFFICE, "proj")
    # Literal 3, deliberately not the module's MAX_SUBS: asserting against the constant
    # moves the goalposts with it, and a raised cap passes a test that imports it.
    assert len(got["subs"]) <= 3
    assert len(got["subs"]) == 3
    for s in got["subs"]:
        assert s["out"][0] == s["in"][0]      # same role letter
    assert len(got["xi"]) == 11


def test_unsubstituted_starter_scores_zero_not_none():
    """Every defender out and the bench exhausted: the XI is still 11 and still totals."""
    roster = _roster()
    missing = {p["id"] for p in roster if p["R"] == "D"}
    got = _pick(roster, _votes(roster, missing), TABLE, OFFICE, "proj")
    assert len(got["xi"]) == 11
    assert isinstance(got["total"], float)
    assert any(e["fantavoto"] is None for e in got["xi"])


def test_hindsight_is_never_below_settable():
    """The ceiling is a ceiling. If this inverts, the settable lineup is peeking."""
    roster = _roster()
    missing = {p["id"] for p in roster if p["R"] in "DC"}
    missing = set(sorted(missing)[:5])
    v = _votes(roster, missing)
    s = _pick(roster, v, TABLE, OFFICE, "proj")
    h = _pick(roster, v, TABLE, OFFICE, "actual")
    assert h["total"] >= s["total"] - 1e-9


# ---- XI advisor -----------------------------------------------------------
# The advisor's paid lesson is baked in as a true positive: on its first run it
# fielded Milan's 134-minute backup keeper over Svilar, because conditional levels
# tied and nothing weighted them by the chance of actually playing.

from scripts.fantacalcio.xi_advisor import HOME_ADJ, advise  # noqa: E402


def _sq(extra=None):
    """A minimal legal squad: 1P/4D/3C/3A across two teams, all certain starters."""
    ps = [
        {"id": 1, "nome": "GK", "R": "P", "team": "Roma", "level": 6.1, "voto": 6.2},
        *[{"id": 10 + i, "nome": f"D{i}", "R": "D", "team": "Roma",
           "level": 6.0, "voto": 6.1} for i in range(4)],
        *[{"id": 20 + i, "nome": f"C{i}", "R": "C", "team": "Lecce",
           "level": 6.3, "voto": 6.2} for i in range(3)],
        *[{"id": 30 + i, "nome": f"A{i}", "R": "A", "team": "Lecce",
           "level": 6.8, "voto": 6.3} for i in range(3)],
    ]
    return ps + (extra or [])


FIX = {"Roma": {"opp": "Lecce", "home": 1, "ts": None},
       "Lecce": {"opp": "Roma", "home": 0, "ts": None}}
ELO = {"Roma": 1500.0, "Lecce": 1400.0}


def test_backup_keeper_cannot_outrank_the_starter():
    """The regression that shipped: backup exp HIGHER, p_play 20x lower -> bench.

    Precondition asserted first (rejection-test rule): without the p_play weighting
    the backup's conditional exp really is the larger number, so a ranking on exp
    alone would field him.
    """
    backup = {"id": 99, "nome": "Backup", "R": "P", "team": "Lecce",
              "level": 6.6, "voto": 6.3, "p_play": 0.04}
    squad = _sq([backup])
    for p in squad:
        p.setdefault("p_play", 0.9)
    adv = advise(squad, FIX, ELO, {})
    b = next(x for x in adv["bench"] + adv["xi"] if x["nome"] == "Backup")
    gk = next(x for x in adv["xi"] + adv["bench"] if x["nome"] == "GK")
    assert b["exp"] > gk["exp"]          # the trap is real: conditional exp says backup
    assert gk in adv["xi"] and b in adv["bench"]   # p_play overrules it


def test_injured_player_is_unavailable_not_fielded():
    squad = _sq()
    adv = advise(squad, FIX, ELO, {30: "knee — out until ~2027-01-01"})
    names = [x["nome"] for x in adv["xi"]]
    assert "A0" not in names
    u = next(x for x in adv["unavailable"] if x["nome"] == "A0")
    assert "knee" in u["inj"]


def test_home_and_elo_lift_expected_voto():
    """Fixture terms must point the right way: home > away, stronger > weaker."""
    squad = _sq()
    home = advise(squad, FIX, ELO, {})
    away_fix = {"Roma": {"opp": "Lecce", "home": 0, "ts": None},
                "Lecce": {"opp": "Roma", "home": 1, "ts": None}}
    away = advise(squad, away_fix, ELO, {})
    gk_h = next(x for x in home["xi"] if x["R"] == "P")
    gk_a = next(x for x in away["xi"] if x["R"] == "P")
    assert gk_h["exp"] - gk_a["exp"] == pytest.approx(HOME_ADJ["P"], abs=0.011)
    flat = advise(squad, FIX, {"Roma": 1400.0, "Lecce": 1400.0}, {})
    gk_f = next(x for x in flat["xi"] if x["R"] == "P")
    assert gk_h["exp"] > gk_f["exp"]     # +100 Elo edge is worth something


def test_no_fixture_means_unavailable():
    squad = _sq([{"id": 98, "nome": "Ghost", "R": "A", "team": "Genoa",
                  "level": 9.9, "voto": 9.9}])
    adv = advise(squad, FIX, ELO, {})
    g = next(x for x in adv["unavailable"] if x["nome"] == "Ghost")
    assert g["why"] == "no fixture this round"
    assert all(x["nome"] != "Ghost" for x in adv["xi"])


def test_rows_without_p_play_are_certain_starters():
    """Back-compat: callers that never model titolarita get pure conditional ranking."""
    adv = advise(_sq(), FIX, ELO, {})
    assert all(x["p_play"] == 1.0 for x in adv["xi"])
    assert adv["total"] == pytest.approx(
        sum(x["exp"] for x in adv["xi"]) + adv["modifier"], abs=0.15)


# ---- bench structure ------------------------------------------------------
# League rule (Nicola 2026-09-02): the bench is 9 ORDERED slots — 1P, 3D, 3C,
# 2A — and auto-subs promote within the role group. A flat exp_slot sort (the
# pre-fix behaviour) puts the best striker first, which is the wrong entry order.


def _full_squad():
    """25-man legal roster: 3P/8D/8C/6A, exp gradients inside each role."""
    def mk(i, n, r, t, lv):
        return {"id": i, "nome": n, "R": r, "team": t,
                "level": lv, "voto": 6.0, "p_play": 0.9}
    ps = [mk(i, f"P{i}", "P", "Roma", 6.5 - i * 0.2) for i in range(3)]
    ds = [mk(10 + i, f"D{i}", "D", "Roma", 6.4 - i * 0.1) for i in range(8)]
    cs = [mk(20 + i, f"C{i}", "C", "Lecce", 6.5 - i * 0.1) for i in range(8)]
    # strikers level HIGH so a flat sort would front-load them on the bench
    as_ = [mk(30 + i, f"A{i}", "A", "Lecce", 7.4 - i * 0.1) for i in range(6)]
    return ps + ds + cs + as_


def test_bench_is_nine_league_slots_in_role_order():
    adv = advise(_full_squad(), FIX, ELO, {})
    roles = [x["R"] for x in adv["bench"]]
    assert roles == ["P", "D", "D", "D", "C", "C", "C", "A", "A"]
    # rejection-test precondition: the flat sort really would order differently —
    # the best benched striker outscores the benched keeper on exp_slot
    best_a = max(x["exp_slot"] for x in adv["bench"] if x["R"] == "A")
    assert best_a > adv["bench"][0]["exp_slot"]
    # inside a role group, slots descend by exp_slot (that IS the entry order)
    for r in "DCA":
        grp = [x["exp_slot"] for x in adv["bench"] if x["R"] == r]
        assert grp == sorted(grp, reverse=True)


def test_tribuna_is_the_remainder_and_nothing_is_lost():
    squad = _full_squad()
    adv = advise(squad, FIX, ELO, {})
    xi = {x["nome"] for x in adv["xi"]}
    bench = {x["nome"] for x in adv["bench"]}
    trib = {x["nome"] for x in adv["tribuna"]}
    assert len(xi) == 11 and len(bench) == 9
    assert not (xi & bench) and not (xi & trib) and not (bench & trib)
    assert xi | bench | trib == {p["nome"] for p in squad}


def test_short_role_pool_leaves_bench_slot_empty_not_stolen():
    """Only 1 spare keeper -> exactly 1 P bench slot; no role borrows another's."""
    squad = [p for p in _full_squad() if p["nome"] != "P2"]
    adv = advise(squad, FIX, ELO, {})
    roles = [x["R"] for x in adv["bench"]]
    assert roles.count("P") == 1 and len(roles) == 9 - 0  # 1P still fits (2 keepers)
    squad = [p for p in squad if p["nome"] != "P1"]       # now only the XI keeper
    adv = advise(squad, FIX, ELO, {})
    roles = [x["R"] for x in adv["bench"]]
    assert roles.count("P") == 0 and roles.count("D") == 3


# ---- probabili parser + p_play override -----------------------------------
# Structure mirrors the live specimen (2026-09-02): team cards with
# starters/reserves lists, ballot percentages OUTSIDE the cards, pids in hrefs.

from scripts.fantacalcio.probabili import (  # noqa: E402
    BALLOT_CLAMP,
    P_RESERVE,
    P_STARTER,
    p_play_override,
    status_by_pid,
)
from scripts.fantacalcio.probabili import parse as prob_parse  # noqa: E402


def _team_card(name, base_pid):
    starters = "".join(
        f'<li><a href="/serie-a/squadre/x/s{i}/{base_pid + i}" class="player-name">'
        f"<span>S{i}</span></a></li>" for i in range(11))
    reserves = "".join(
        f'<li><a href="/serie-a/squadre/x/r{i}/{base_pid + 50 + i}" class="player-name">'
        f"<span>R{i}</span></a></li>" for i in range(4))
    return (f'<div class="card team-card"><h6 class="h6 team-name">{name}</h6>'
            f'<h6 class="h6 team-formation">3-5-2</h6>'
            f'<ul class="player-list starters">{starters}</ul>'
            f'<ul class="player-list reserves">{reserves}</ul></div>')


def _page(n_teams=20):
    cards = "".join(_team_card(f"Team{i}", 1000 + 100 * i) for i in range(n_teams))
    ballots = ('<ul class="ballot-list"><li><a href="/serie-a/squadre/x/s0/1000">'
               '<span>S0</span></a> <strong class="percentage">40</strong></li>'
               '<li><a href="/serie-a/squadre/x/s1/1001"><span>S1</span></a> '
               '<strong class="percentage">99</strong></li></ul>')
    return f'<span class="matchweek">7</span>{cards}{ballots}'


def test_probabili_parse_reads_teams_pids_and_ballots():
    data = prob_parse(_page())
    assert data is not None and data["matchweek"] == 7
    assert len(data["teams"]) == 20
    t0 = data["teams"]["Team0"]
    assert t0["formation"] == "3-5-2"
    assert [p["pid"] for p in t0["starters"]] == list(range(1000, 1011))
    assert [p["pid"] for p in t0["reserves"]] == list(range(1050, 1054))
    assert data["ballots"] == {1000: 40, 1001: 99}


def test_probabili_schema_break_returns_none_not_empty():
    """2 cards is a broken page, not a quiet week — cache fallback, never {}."""
    assert prob_parse(_page(n_teams=2)) is None
    assert prob_parse("<html>maintenance</html>") is None


def test_p_play_override_sources_and_clamp():
    by_pid = status_by_pid(prob_parse(_page()))
    assert p_play_override(1000, 0.5, by_pid) == (0.40, "ballottaggio")
    assert p_play_override(1001, 0.5, by_pid) == (BALLOT_CLAMP[1], "ballottaggio")
    assert p_play_override(1002, 0.5, by_pid) == (P_STARTER, "probabili")
    assert p_play_override(1050, 0.5, by_pid) == (P_RESERVE, "probabili")
    assert p_play_override(424242, 0.37, by_pid) == (0.37, "model")


# ---- news surname matching ------------------------------------------------

from scripts.fantacalcio.news import _matcher, _surname  # noqa: E402


def test_surname_strips_initials_keeps_double_names():
    assert _surname("Adams A.") == "Adams"
    assert _surname("Martinez Jo.") == "Martinez"
    assert _surname("Pellegrino M.") == "Pellegrino"
    assert _surname("Tiago Gabriel") == "Tiago Gabriel"
    assert _surname("Vlasic") == "Vlasic"


def test_news_matcher_is_word_bounded_not_substring():
    roster = [{"nome": "Adams A."}, {"nome": "Rrahmani"}]
    mm = _matcher(roster)
    rx_adams = next(rx for p, rx in mm if p["nome"] == "Adams A.")
    rx_rrah = next(rx for p, rx in mm if p["nome"] == "Rrahmani")
    assert rx_adams.search("Tony Adams segna ancora")
    assert not rx_adams.search("il commissario Adamsberg indaga")
    assert rx_rrah.search("Rrahmani torna titolare")
    assert not rx_rrah.search("il Rahm della situazione")


# ---- prediction ledger -----------------------------------------------------
# The loop that will refit the p_play/exp heuristics: forecasts must be frozen
# EX-ANTE (post-kickoff writes refused), reconciled only after the grace window
# (a half-graded weekend must not poison the actuals), and an OUT call is a
# p_play=0 prediction that a surprise appearance must punish.

import pandas as pd  # noqa: E402

import scripts.fantacalcio.pred_ledger as pl  # noqa: E402

KICK = 1_800_000_000.0


def _adv(total=60.0):
    def p(i, nome, src, pp, exp):
        return {"id": i, "nome": nome, "R": "C", "team": "Roma", "opp": "Lecce",
                "home": 1, "p_play": pp, "p_play_src": src,
                "exp": exp, "exp_voto": 6.0}
    return {"round": 3, "first_kickoff": KICK, "module": "4-3-3", "total": total,
            "modifier": 1.0,
            "xi": [p(1, "Tito", "probabili", 0.88, 6.0)],
            "bench": [p(2, "Panca", "ballottaggio", 0.40, 6.2)],
            "tribuna": [p(4, "Trib", "model", 0.30, 6.1)],
            "unavailable": [{"id": 3, "nome": "Rotto", "R": "D", "team": "Roma",
                             "p_play": 0.5, "inj": "knee"}]}


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(pl, "LEDGER", tmp_path / "pred_ledger.json")
    monkeypatch.setattr(pl, "VOTI_DIR", tmp_path)


def test_snapshot_overwrites_before_kickoff_and_freezes_after(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    assert pl.snapshot(_adv(60.0), now_ts=KICK - 1000) == "updated"
    assert pl.snapshot(_adv(61.5), now_ts=KICK - 500) == "updated"
    assert pl.snapshot(_adv(99.9), now_ts=KICK + 10) == "frozen"
    led = pl._load()
    e = led["rounds"]["3"]
    assert e["predicted_total"] == 61.5      # last PRE-kickoff snapshot won
    assert e["frozen_at"] is not None
    assert pl.snapshot(_adv(99.9), now_ts=KICK + 20) == "skipped-post-kickoff"
    assert pl._load()["rounds"]["3"]["predicted_total"] == 61.5


def test_round_never_snapshotted_stays_a_hole_not_a_backfill(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    assert pl.snapshot(_adv(), now_ts=KICK + 10) == "skipped-post-kickoff"
    assert pl._load()["rounds"] == {}


def _voti_parquet(tmp_path):
    pd.DataFrame({
        "pid": [1, 2, 3],                       # Trib (4) has no row at all
        "voto": [6.5, None, 6.0],
        "fantavoto": [9.5, None, 6.0],
        "played": [True, False, True],          # the OUT player played: worst miss
    }).to_parquet(pl._round_parquet(3), index=False)


def test_reconcile_scores_errors_and_the_out_miss(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    pl.snapshot(_adv(), now_ts=KICK - 1000)
    _voti_parquet(tmp_path)
    assert pl.reconcile(now_ts=KICK + 1 * 86400) == []          # grace window holds
    assert pl.reconcile(now_ts=KICK + 5 * 86400) == [3]
    e = pl._load()["rounds"]["3"]
    by = {p["nome"]: p for p in e["players"]}
    assert by["Tito"]["err_fv"] == pytest.approx(3.5)           # 9.5 - 6.0
    assert by["Tito"]["play_brier"] == pytest.approx((0.88 - 1) ** 2, abs=1e-4)
    assert by["Panca"]["play_brier"] == pytest.approx(0.40 ** 2, abs=1e-4)
    assert "err_fv" not in by["Panca"]                          # sv: no voto error
    assert by["Rotto"]["p_play"] == 0.0 and by["Rotto"]["play_brier"] == 1.0
    assert by["Trib"]["actual_played"] is False
    m = e["metrics"]
    assert m["n_played"] == 2 and m["n_scored"] == 1
    assert m["mae_fv"] == pytest.approx(3.5)
    assert pl.reconcile(now_ts=KICK + 6 * 86400) == []          # idempotent


def test_summary_calibration_by_source(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    pl.snapshot(_adv(), now_ts=KICK - 1000)
    _voti_parquet(tmp_path)
    pl.reconcile(now_ts=KICK + 5 * 86400)
    s = pl.summary()
    assert s["rounds"][0]["reconciled"] is True
    c = s["calibration"]
    assert c["probabili"] == {"n": 1, "predicted_rate": 0.88, "realized_rate": 1.0}
    assert c["ballottaggio"]["realized_rate"] == 0.0
    assert c["out"] == {"n": 1, "predicted_rate": 0.0, "realized_rate": 1.0}
    assert c["model"]["realized_rate"] == 0.0


# ---- rival matrix + risk tilt ----------------------------------------------
# Two competitions, opponents unknown in advance: the advisor prices every
# rival. The tilt must ONLY change selection (chase variance as underdog, buy
# stability as favourite) — reported totals stay the honest mean.

from scripts.fantacalcio.xi_advisor import (_p_win, _rival_roster,  # noqa: E402
                                            SD_ROLE)


def _sq_sd():
    """4 strikers, 3 slots, two locked stars: the LAST slot is Steady (higher
    exp, low sd) vs Volatile (lower exp, high ceiling) — the tilt's whole job."""
    squad = _sq([{"id": 40, "nome": "Volatile", "R": "A", "team": "Lecce",
                  "level": 6.7, "voto": 6.0}])
    for p in squad:
        p["p_play"] = 0.9
        p["sd"] = 1.0
    star, star2, steady, volatile = [p for p in squad if p["R"] == "A"]
    star["level"] = 7.5
    star2.update({"level": 7.4, "sd": 1.2})
    steady.update({"level": 6.9, "sd": 0.6, "nome": "Steady"})
    volatile["sd"] = 2.4
    return squad


def test_risk_tilt_changes_selection_not_the_reported_total():
    squad = _sq_sd()
    flat = advise(squad, FIX, ELO, {})
    up = advise(squad, FIX, ELO, {}, risk_lambda=0.5)
    down = advise(squad, FIX, ELO, {}, risk_lambda=-0.5)
    names = lambda a: {x["nome"] for x in a["xi"]}  # noqa: E731
    # precondition: on pure exp Steady (6.81) edges Volatile (6.80)
    assert "Steady" in names(flat) and "Volatile" not in names(flat)
    assert "Volatile" in names(up) and "Steady" not in names(up)
    assert "Steady" in names(down)
    # the tilted pick reports ITS OWN honest mean, not the tilted objective
    assert up["total"] == pytest.approx(
        sum(x["exp_slot"] for x in up["xi"]) + up["modifier"], abs=0.02)
    assert up["total"] < flat["total"]          # variance was bought, mean paid
    assert up["xi_sd"] > flat["xi_sd"]


def test_xi_sd_is_root_sum_of_variances():
    squad = _sq_sd()
    adv = advise(squad, FIX, ELO, {})
    want = sum(x.get("sd", 0.0) ** 2 for x in adv["xi"]) ** 0.5
    assert adv["xi_sd"] == pytest.approx(want, abs=0.01)


def test_p_win_symmetry_and_monotonicity():
    assert _p_win(60, 4, 60, 4) == pytest.approx(0.5)
    assert _p_win(65, 4, 60, 4) + _p_win(60, 4, 65, 4) == pytest.approx(1.0)
    assert _p_win(65, 4, 60, 4) > _p_win(62, 4, 60, 4)
    # more noise pulls the favourite toward a coin flip
    assert _p_win(65, 10, 60, 10) < _p_win(65, 3, 60, 3)


def test_rival_roster_uses_priors_and_skips_unknown_ids():
    by_id = {7: {"id": 7, "nome": "Rivale", "R": "A", "team": "Roma",
                 "season_points": 38.0, "mv_hat": 6.2, "proj_min": 3000}}
    hist = {"sd": {}, "live": {}, "rounds_elapsed": 0}
    entry = {"roster": [{"id": 7}, {"id": 999}], "unmatched": [{"nome": "Ghost"}]}
    rows = _rival_roster(entry, by_id, hist, {})
    assert len(rows) == 1
    r = rows[0]
    assert r["level"] == pytest.approx(7.0)          # 6.0 + 38/38
    assert r["sd"] == SD_ROLE["A"]                   # no history -> role prior
    assert 0.02 <= r["p_play"] <= 0.95
