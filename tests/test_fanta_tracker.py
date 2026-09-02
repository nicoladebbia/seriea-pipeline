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

from scripts.fantacalcio.xi_advisor import SD_ROLE, _p_win, _rival_roster  # noqa: E402


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


# ---- calendar import, congestion, press pulse ------------------------------

import scripts.fantacalcio.team_pulse as tp  # noqa: E402
from scripts.fantacalcio.import_rosters import parse_calendar_xlsx  # noqa: E402
from scripts.fantacalcio.xi_advisor import _congestion_from_events  # noqa: E402

CAL_TEAMS = {"Alpha FC", "Beta United", "Gamma Club", "Delta Town"}


def _cal_xlsx(tmp_path):
    """Synthetic Leghe layout: 2 rounds per row-block, gironi + Riposa + one
    PLAYED score — the parse must key on team names, never on cell position."""
    rows = [["Calendario Test Cup"] + [None] * 9,
            ["https://leghe.example"] + [None] * 9]
    for blk in range(4):
        l1, l2 = 2 * blk + 1, 2 * blk + 2
        rows.append([f"{l1}ª Giornata lega", f"{l1 + 2}ª Giornata serie a",
                     None, None, None,
                     f"{l2}ª Giornata lega", f"{l2 + 2}ª Giornata serie a",
                     None, None, None])
        score = "71.5" if blk == 0 else "0.0"      # a settled round has scores
        rows.append(["A", "Alpha FC", score, 1, "Beta United",
                     "A", "Gamma Club", "0.0", 0, "Delta Town"])
        rows.append(["A", "Gamma Club", "0.0", 0, "Delta Town",
                     "A", "Alpha FC", "0.0", 0, "Beta United"])
        rows.append(["A", "Riposa Alpha FC", None, None, None,
                     "A", "Riposa Beta United", None, None, None])
    import pandas as pd
    path = tmp_path / "cal.xlsx"
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    return path


def test_calendar_parser_reads_both_halves_and_survives_scores(tmp_path):
    comp, rounds = parse_calendar_xlsx(_cal_xlsx(tmp_path), CAL_TEAMS)
    assert comp == "Test Cup"
    assert [r["league_round"] for r in rounds] == list(range(1, 9))
    assert rounds[0]["sa_round"] == 3 and rounds[7]["sa_round"] == 10
    r1 = rounds[0]
    assert {(f["home"], f["away"]) for f in r1["fixtures"]} == \
        {("Alpha FC", "Beta United"), ("Gamma Club", "Delta Town")}
    assert r1["rests"] == [{"girone": "A", "team": "Alpha FC"}]
    assert all(f["girone"] == "A" for f in r1["fixtures"])


def test_congestion_picks_latest_finished_before_kickoff():
    now = 1_800_000_000
    evs = [
        {"startTimestamp": now - 9 * 86400, "status": {"type": "finished"},
         "tournament": {"name": "Serie A"}},
        {"startTimestamp": now - 3 * 86400, "status": {"type": "finished"},
         "tournament": {"name": "Coppa Italia"}},
        {"startTimestamp": now + 86400, "status": {"type": "notstarted"},
         "tournament": {"name": "Serie A"}},
    ]
    c = _congestion_from_events(evs, next_ts=now)
    assert c["last_comp"] == "Coppa Italia"
    assert c["rest_d"] == pytest.approx(3.0) and c["congested"] is True
    c2 = _congestion_from_events(evs[:1], next_ts=now)
    assert c2["congested"] is False
    assert _congestion_from_events([evs[2]], next_ts=now) is None


def test_lexicon_signs():
    assert tp.lex_score("Vittoria e doppietta, che show") > 0
    assert tp.lex_score("Crisi nera e infortunio muscolare") < 0
    assert tp.lex_score("conferenza stampa ordinaria") == 0


def test_pulse_updates_learns_and_decays(monkeypatch, tmp_path):
    monkeypatch.setattr(tp, "STATE", tmp_path / "pulse.json")
    item = {"link": "l1", "source": "T", "title": "Toro, vittoria e show",
            "desc": ""}
    st = tp.update([item], next_round=3)
    assert st["teams"]["Torino"]["pulse"] > 0          # "Toro" alias matched
    assert st["pending"] and st["pending"][0]["sa_round"] == 3
    p0 = st["teams"]["Torino"]["pulse"]

    # same link again: dedup, nothing moves
    assert tp.update([item], next_round=3)["teams"]["Torino"]["n_items"] == 1

    # the round settles as a WIN -> the words train the NB layer
    monkeypatch.setattr(tp, "_results_by_round", lambda: {("Torino", 3): 1})
    st = tp.update([], next_round=4)
    assert st["nb"]["n_pos"] == 1 and not st["pending"]
    assert st["nb"]["pos"].get("vittoria") == 1

    # a week of silence halves the pulse (half-life decay)
    from datetime import UTC, datetime, timedelta
    week_ago = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    raw = __import__("json").loads((tmp_path / "pulse.json").read_text())
    raw["teams"]["Torino"]["updated_at"] = week_ago
    (tmp_path / "pulse.json").write_text(__import__("json").dumps(raw))
    st = tp.update([], next_round=4)
    assert st["teams"]["Torino"]["pulse"] == pytest.approx(p0 / 2, rel=0.05)


# ---- discipline, push phases, final-check diff -----------------------------

from scripts.fantacalcio.tracker import _advice_diff, _discipline_from_frames, _push_phase  # noqa: E402


def _cards_df(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["pid", "cards", "played"])


def test_discipline_counts_diffida_and_bans():
    frames = [
        (1, _cards_df([(7, -0.5, True), (8, 0.0, True)])),
        (2, _cards_df([(7, -0.5, True), (9, -1.0, True)])),
        (3, _cards_df([(7, -0.5, True), (10, -0.5, True)])),
    ]
    d = _discipline_from_frames(frames)
    assert d[7]["yellows"] == 3 and not d[7]["diffidato"] and not d[7]["banned_next"]
    assert d[10] == {"yellows": 1, "diffidato": False, "banned_next": False,
                     "why": None}
    # red in a NON-latest round is already served: NO forward state at all
    assert 9 not in d


def test_fifth_yellow_in_latest_round_bans_next():
    frames = [(r, _cards_df([(7, -0.5, True)])) for r in range(1, 6)]
    d = _discipline_from_frames(frames)
    assert d[7]["yellows"] == 5 and d[7]["banned_next"]
    assert "5°" in d[7]["why"]
    # once round 6 is played clean, the flag self-clears
    frames.append((6, _cards_df([(7, 0.0, True)])))
    assert not _discipline_from_frames(frames)[7]["banned_next"]
    # 4 yellows = diffidato, no ban
    d4 = _discipline_from_frames(frames[:4])
    assert d4[7]["diffidato"] and not d4[7]["banned_next"]


def test_red_in_latest_round_bans_even_first_appearance():
    d = _discipline_from_frames([(1, _cards_df([(9, -1.0, True)]))])
    assert d[9]["banned_next"] and "espulsione" in d[9]["why"]
    # sv rows (played=False) never count
    d2 = _discipline_from_frames([(1, _cards_df([(9, -1.0, False)]))])
    assert 9 not in d2


def test_push_phase_state_machine():
    kick = 1_800_000_000.0
    h = 3600
    # no state yet: first push only inside 48h
    assert _push_phase({}, 3, kick, kick - 72 * h) is None
    assert _push_phase({}, 3, kick, kick - 40 * h) == "first"
    st = {"round": 3, "final_checked": False}
    # same round, still far out -> silent
    assert _push_phase(st, 3, kick, kick - 20 * h) is None
    # inside the final window -> final, once
    assert _push_phase(st, 3, kick, kick - 3 * h) == "final"
    assert _push_phase({"round": 3, "final_checked": True}, 3, kick,
                       kick - 2 * h) is None
    # after kickoff or missing round: nothing
    assert _push_phase(st, 3, kick, kick + 60) is None
    assert _push_phase(st, None, kick, kick - 3 * h) is None
    # a NEW round resets the machine
    assert _push_phase(st, 4, kick + 7 * 86400, kick + 6 * 86400) == "first"


def test_advice_diff_reports_only_actionable_changes():
    prev = {"module": "3-4-3", "xi": ["A", "B"], "bench": ["C", "D"]}
    assert _advice_diff(prev, {"module": "3-4-3", "xi": ["A", "B"],
                               "bench": ["C", "D"]}) is None
    d = _advice_diff(prev, {"module": "4-3-3", "xi": ["A", "E"],
                            "bench": ["C", "D"]})
    assert "3-4-3 → 4-3-3" in d and "Dentro: E" in d and "Fuori: B" in d
    d2 = _advice_diff(prev, {"module": "3-4-3", "xi": ["A", "B"],
                             "bench": ["D", "C"]})
    assert "Panchina riordinata" in d2


# ---------------------------------------------------------------------------
# mercato sync: live-listone parse, board reconciliation, departed guards
# ---------------------------------------------------------------------------
from scripts.fantacalcio.import_rosters import _apply_live, parse_quotazioni  # noqa: E402


def _qrow(nome, role, team_slug, pid):
    return (f'<tr class="player-row" data-index="0" '
            f'data-filter-keywords="{nome}" data-filter-team-id="1" '
            f'data-filter-playeds="0" data-filter-role-classic="{role}">'
            f'<a href="https://www.fantacalcio.it/serie-a/squadre/'
            f'{team_slug}/x/{pid}">x</a></tr>')


def test_parse_quotazioni_extracts_and_unescapes_or_refuses():
    clubs = ["roma", "milan", "inter", "juventus", "napoli", "lazio",
             "atalanta", "bologna", "fiorentina", "torino", "genoa", "como",
             "cagliari", "lecce", "udinese", "sassuolo", "parma", "monza",
             "frosinone", "venezia"]
    rows = [_qrow("Svilar", "p", "roma", 5841),
            _qrow("Kessi&#xE8;", "c", "atalanta", 1850)]
    pid = 10000
    for i in range(420):
        rows.append(_qrow(f"Player{i}", "d", clubs[i % 20], pid + i))
    live = parse_quotazioni("".join(rows))
    assert live is not None and len(live) == 422
    assert live[1850] == {"nome": "Kessiè", "R": "C", "team": "Atalanta"}
    assert live[5841]["team"] == "Roma"
    # a page with too few rows is a schema break, not a small week
    assert parse_quotazioni("".join(rows[:50])) is None


def test_apply_live_never_touches_departures_and_adopts_placeholders():
    board = {"players": [
        # genuinely departed, still listed on the quotazioni page (measured):
        {"id": 5876, "nome": "Di Gregorio", "R": "P", "team": "Juventus",
         "status": "DEPARTED", "note": "→ Bournemouth 2026-08-25"},
        # intra-SA move the snapshot missed:
        {"id": 111, "nome": "Mover", "R": "C", "team": "Torino",
         "status": "TEAM_MISMATCH", "note": ""},
        # auction-time placeholder pid — exists in no other artifact:
        {"id": 99004, "nome": "Theate", "R": "D", "team": "Bologna",
         "status": "OK", "note": "", "fvm": 30.0, "proj_min": 1772.0},
    ]}
    live = {5876: {"nome": "Di Gregorio", "R": "P", "team": "Juventus"},
            111: {"nome": "Mover", "R": "C", "team": "Lecce"},
            5675: {"nome": "Theate", "R": "D", "team": "Bologna"},
            222: {"nome": "Nuovo", "R": "A", "team": "Monza"}}
    changes = _apply_live(board, live)
    by_id = {p["id"]: p for p in board["players"]}
    # the departure survives with its note intact — presence on the page
    # proves nothing (the page keeps rows of players who left)
    assert by_id[5876]["status"] == "DEPARTED"
    assert "Bournemouth" in by_id[5876]["note"]
    # the move lands, the listone club is preserved, status verified
    assert by_id[111]["team"] == "Lecce"
    assert by_id[111]["team_listone"] == "Torino"
    assert by_id[111]["status"] == "OK"
    # the placeholder row was corrected IN PLACE (auction priors kept),
    # not duplicated by a bare arrival row
    assert 99004 not in by_id and by_id[5675]["proj_min"] == 1772.0
    assert sum(p["nome"] == "Theate" for p in board["players"]) == 1
    # the true newcomer is appended
    assert by_id[222]["nome"] == "Nuovo"
    kinds = {c.split()[0] for c in changes}
    assert kinds == {"VERIFIED", "MOVED", "PID-FIX", "ARRIVED"}


def test_departed_rostered_player_cannot_be_fielded():
    by_id = {9: {"id": 9, "nome": "Gone", "R": "A", "team": "Milan",
                 "status": "DEPARTED", "season_points": 38.0}}
    hist = {"sd": {}, "live": {}, "rounds_elapsed": 2}
    rows = _rival_roster({"roster": [{"id": 9}]}, by_id, hist, {})
    assert rows[0]["departed"] is True
    assert rows[0]["p_play"] == 0.02 and rows[0]["p_play_src"] == "departed"


# ---------------------------------------------------------------------------
# standings, risk alerts, round digest, trade scanner
# ---------------------------------------------------------------------------
from scripts.fantacalcio.news import classify_risk  # noqa: E402
from scripts.fantacalcio.tracker import _new_risk_alerts, _round_digest, _standings_from_schedule  # noqa: E402
from scripts.fantacalcio.trades import evaluate_offer, scan_windows, starter_lines, team_strength  # noqa: E402


def test_standings_count_only_real_scores_and_split_gironi():
    sch = {"competitions": {"CDN": {"format": "gironi", "rounds": [
        {"league_round": 1, "sa_round": 3, "rests": [], "fixtures": [
            {"girone": "A", "home": "X", "away": "Y", "score": "2-1",
             "fp_home": 74.5, "fp_away": 68.0},
            {"girone": "B", "home": "Z", "away": "W", "score": "0-0",
             "fp_home": 60.0, "fp_away": 61.5},
            # unplayed shapes must be invisible, whatever the cells hold
            {"girone": "A", "home": "P", "away": "Q", "score": None,
             "fp_home": 0.0, "fp_away": 0.0},
            {"girone": "B", "home": "R", "away": "S", "score": "-",
             "fp_home": 0.0, "fp_away": 0.0},
        ]}]}}}
    st = _standings_from_schedule(sch)["CDN"]
    assert st["rounds_played"] == 1
    a = {r["team"]: r for r in st["tables"]["A"]}
    assert a["X"]["pts"] == 3 and a["X"]["gf"] == 2 and a["X"]["fp"] == 74.5
    assert a["Y"]["pts"] == 0 and "P" not in a
    b = {r["team"]: r for r in st["tables"]["B"]}
    assert b["Z"]["pts"] == 1 and b["W"]["pts"] == 1
    # no giocata at all -> empty, never wrong
    for f in sch["competitions"]["CDN"]["rounds"][0]["fixtures"]:
        f["score"] = "-"
    st2 = _standings_from_schedule(sch)["CDN"]
    assert st2["rounds_played"] == 0 and st2["tables"] == {}


def test_risk_classifier_and_weekly_latch():
    from scripts.fantacalcio.news import risk_hits
    inj = {"title": "Vlasic si ferma: lesione al flessore", "desc": "",
           "players": ["Vlasic"]}
    calm = {"title": "Vlasic decisivo, che assist nel derby", "desc": "",
            "players": ["Vlasic"]}
    # risky story about SOMEONE ELSE that name-drops my player in the body
    # (the David-to-Atletico / coach-Simeone false positive): no alert
    other = {"title": "David all'Atletico: è ufficiale la cessione",
             "desc": "presentato da Simeone in conferenza",
             "players": ["Simeone"]}
    assert classify_risk(inj) == "infortunio"
    assert classify_risk(calm) is None
    assert risk_hits(inj) == [("Vlasic", "infortunio")]
    assert risk_hits(other) == []
    state: dict = {}
    l1 = _new_risk_alerts([inj, calm, other], state,
                          "2026-09-02T10:00:00+00:00", risk_hits)
    assert len(l1) == 1 and "Vlasic" in l1[0]
    # same signature two days later: silent
    l2 = _new_risk_alerts([inj], state, "2026-09-04T10:00:00+00:00",
                          risk_hits)
    assert l2 == []
    # after the re-alert window: fires again
    l3 = _new_risk_alerts([inj], state, "2026-09-10T10:00:01+00:00",
                          risk_hits)
    assert len(l3) == 1


def test_round_digest_fires_once_per_settled_round(monkeypatch, tmp_path):
    import json

    import scripts.fantacalcio.tracker as trk
    tf = tmp_path / "tracker.json"
    tf.write_text(json.dumps({"rounds_played": 2, "rounds": [
        {"round": 1, "settable": {"total": 60.0, "module": "4-3-3", "xi": []},
         "hindsight": {"total": 62.0, "module": "4-3-3", "xi": []}},
        {"round": 2,
         "settable": {"total": 67.0, "module": "3-4-3",
                      "xi": [{"nome": "A", "fantavoto": 7.0}]},
         "hindsight": {"total": 74.5, "module": "3-4-3",
                       "xi": [{"nome": "A", "fantavoto": 7.0},
                              {"nome": "Douvikas", "fantavoto": 10.0}]}},
    ]}))
    monkeypatch.setattr(trk, "OUT", tf)
    state: dict = {}
    d = _round_digest(state)
    assert d and "giornata 2" in d["title"]
    assert "Panchina costata: +7.5" in d["text"]
    assert "Douvikas" in d["text"]
    assert state["digest_round"] == 2
    assert _round_digest(state) is None       # latched


def _trow(nome, R, level, p_play=0.9, pid=None):
    return {"id": pid or hash(nome) % 10000, "nome": nome, "R": R,
            "level": level, "p_play": p_play}


def test_team_strength_picks_best_legal_module():
    rows = ([_trow("Gk", "P", 6.0)]
            + [_trow(f"D{i}", "D", 6.0) for i in range(3)]
            + [_trow(f"C{i}", "C", 6.0) for i in range(4)]
            + [_trow(f"A{i}", "A", 7.0) for i in range(3)])
    tot, mod = team_strength(rows)
    assert mod == "3-4-3"                    # the only feasible module
    lines = starter_lines(rows, (3, 4, 3))
    assert lines["A"] == 0.9 * 7.0 and lines["P"] == 0.9 * 6.0


def test_scan_windows_requires_mutual_gain():
    me = ([_trow("MyGk", "P", 6.2)]
          + [_trow(f"MyD{i}", "D", 6.4) for i in range(4)]
          + [_trow("SurplusD", "D", 6.3)]          # would start for the rival
          + [_trow(f"MyC{i}", "C", 5.8) for i in range(4)]
          + [_trow(f"MyA{i}", "A", 6.5) for i in range(3)])
    rival = ([_trow("RGk", "P", 6.0)]
             + [_trow(f"RD{i}", "D", 5.6) for i in range(4)]   # D poverty
             + [_trow(f"RC{i}", "C", 6.6) for i in range(4)]
             + [_trow("SurplusC", "C", 6.5)]       # upgrades my C line
             + [_trow(f"RA{i}", "A", 6.2) for i in range(3)])
    ws = scan_windows({"Me": me, "Riv": rival}, "Me")
    dc = [w for w in ws if w["give_R"] == "D" and w["get_R"] == "C"]
    assert dc, f"expected a D->C window vs Riv, got {ws}"
    assert all(w["my_gain"] >= 0.05 and w["their_gain"] >= 0.02 for w in ws)
    # one best window per (rival, role pair), never a spam of equal swaps
    assert len(dc) == 1
    # a rival with no poverty offers no window
    strong = ([_trow("SGk", "P", 7.0)]
              + [_trow(f"SD{i}", "D", 7.0) for i in range(5)]
              + [_trow(f"SC{i}", "C", 7.0) for i in range(5)]
              + [_trow(f"SA{i}", "A", 7.5) for i in range(3)])
    assert scan_windows({"Me": me, "Str": strong}, "Me") == []


def test_evaluate_offer_verdicts_and_shape_guard():
    me = ([_trow(f"P{i}", "P", 6.0) for i in range(3)]
          + [_trow(f"D{i}", "D", 6.0) for i in range(8)]
          + [_trow(f"C{i}", "C", 6.0) for i in range(8)]
          + [_trow(f"A{i}", "A", 6.0) for i in range(6)])
    up = evaluate_offer([me[3]], [_trow("Star", "D", 7.5)], me)
    assert up["verdict"] == "ACCETTA" and up["delta"] > 0
    down = evaluate_offer([me[-1]], [_trow("Scrub", "A", 4.0)], me)
    assert down["verdict"] in ("RIFIUTA", "TRATTA")
    cross = evaluate_offer([me[3]], [_trow("Wrong", "A", 7.0)], me)
    assert cross["verdict"] == "ROSA ILLEGALE" and not cross["shape_ok"]


def test_board_priors_prefer_model_then_real_history_then_floor():
    from scripts.fantacalcio.xi_advisor import _board_priors
    model = {"R": "A", "season_points": 19.0, "mv_hat": 6.4,
             "proj_min": 38.0 * 60 * 0.8}
    lvl, voto, pp = _board_priors(model, {"n": 30, "fv": 7.5, "voto": 7.0})
    assert lvl == 6.5 and voto == 6.4          # the model wins when it scored him
    # no model projection, REAL 30-game 6.33 season on disk (the David case):
    lvl2, voto2, pp2 = _board_priors({"R": "A"}, {"n": 30, "fv": 6.33,
                                                  "voto": 6.1})
    assert 6.2 < lvl2 < 6.33 and 6.0 < voto2 < 6.1
    assert pp2 == 30 / 38
    # nothing known at all: the honest floor, not a fake confidence
    lvl3, voto3, pp3 = _board_priors({"R": "A"}, None)
    assert (lvl3, voto3, pp3) == (6.0, 6.0, 0.02)


def test_wiki_departure_pass_marks_gone_spares_moves_and_collisions():
    from scripts.fantacalcio.import_rosters import _mark_departures
    board = {"players": [
        # deadline-day departure of a post-listone arrival (Jonathan David):
        {"id": 5544, "nome": "David", "R": "A", "team": "Juventus",
         "status": "OK", "note": "post-listone arrival"},
        # intra-SA move: never a departure
        {"id": 1, "nome": "Frattesi", "R": "C", "team": "Inter",
         "status": "OK", "note": ""},
        # surname near-miss must not fire (Ankeye is not David)
        {"id": 2, "nome": "Ankeye", "R": "A", "team": "Genoa",
         "status": "OK", "note": ""},
        # already departed: untouched
        {"id": 3, "nome": "Leao", "R": "A", "team": "Milan",
         "status": "DEPARTED", "note": "→ Galatasaray 2026-08-30"},
    ]}
    live = {10: {"nome": "X", "R": "D", "team": "Inter"},
            11: {"nome": "Y", "R": "D", "team": "Lazio"}}
    wiki = [
        {"player_name": "Jonathan David", "from_club": "Juventus",
         "to_club": "Atlético Madrid", "transfer_date": "2026-09-01"},
        {"player_name": "Davide Frattesi", "from_club": "Inter",
         "to_club": "Lazio", "transfer_date": "2026-08-14"},
        {"player_name": "David Ankeye", "from_club": "Genoa",
         "to_club": "Krasava ENY", "transfer_date": "2026-01-26"},
    ]
    ch = _mark_departures(board, live, wiki, listed_pids={2})
    by = {p["nome"]: p for p in board["players"]}
    assert by["David"]["status"] == "DEPARTED"
    assert "Atlético Madrid" in by["David"]["note"]
    assert by["Frattesi"]["status"] == "OK"
    # Ankeye left per wiki, BUT his pid is on the probabili page — the
    # pid-exact fresh source beats the name-matched wiki row
    assert by["Ankeye"]["status"] == "OK"
    assert "Galatasaray" in by["Leao"]["note"]        # untouched
    assert len(ch) == 1
    # a row wrongly wiki-marked earlier is restored once he shows up listed
    board["players"][0].update(status="DEPARTED",
                               note="→ Atlético Madrid [wiki 2026-09-02]")
    ch2 = _mark_departures(board, live, wiki, listed_pids={5544})
    assert by["David"]["status"] == "OK" and ch2 and "RESTORED" in ch2[0]


def test_advise_module_restriction_and_fielded_ledger(tmp_path):
    from scripts.fantacalcio.xi_advisor import _observed_modules, advise, record_fielded
    roster = ([{"id": 1, "nome": "Gk", "R": "P", "team": "X", "level": 6.0,
                "voto": 6.0}]
              + [{"id": 10 + i, "nome": f"D{i}", "R": "D", "team": "X",
                  "level": 6.0, "voto": 6.0} for i in range(5)]
              + [{"id": 20 + i, "nome": f"C{i}", "R": "C", "team": "X",
                  "level": 6.0, "voto": 6.0} for i in range(5)]
              + [{"id": 30 + i, "nome": f"A{i}", "R": "A", "team": "X",
                  "level": 7.0, "voto": 6.5} for i in range(3)])
    fx = {"X": {"opp": "Y", "home": 1}}
    free = advise(roster, fx, {}, {})
    # 3 strikers up front, and nd>=4 keeps the defense modifier
    assert free["module"] == "4-3-3"
    forced = advise(roster, fx, {}, {}, modules=[(5, 4, 1)])
    assert forced["module"] == "5-4-1"      # observed repertoire wins
    led = tmp_path / "rm.json"
    record_fielded("Munnezz FC", "4-4-2", 3, path=led)
    record_fielded("Munnezz FC", "3-5-2", 4, path=led)
    record_fielded("Munnezz FC", "4-4-2", 5, path=led)
    assert _observed_modules("Munnezz FC", path=led) == [(4, 4, 2), (3, 5, 2)]
    assert _observed_modules("Sconosciuti", path=led) == []


def test_vs_block_collapses_comps_and_diff_guard():
    from scripts.fantacalcio.tracker import _advice_diff, _vs_block
    riv = {
        "me": {"team": "Whisky Palermo", "module": "3-4-3"},
        "next_opponents": [
            {"competition": "Coppa Del Nonno", "opponent": "Munnezz FC"},
            {"competition": "Hunger Games", "opponent": "Munnezz FC"},
        ],
        "rivals": [{"team": "Munnezz FC", "module": "4-4-2",
                    "module_src": "osservato", "total": 63.0, "p_win": 0.46,
                    "alt": {"module": "3-5-2", "p_win": 0.49,
                            "in": ["Schmid"], "out": ["Simeone"]}}],
    }
    txt, tg, sig = _vs_block(riv)
    # same opponent, same recommendation in both comps -> ONE collapsed line
    assert txt.count("Munnezz FC") == 1 and "Coppa Del Nonno + Hunger" in txt
    assert "visto" in txt and "gioca 3-5-2" in txt and "Schmid" in txt
    assert sig["Coppa Del Nonno"] == sig["Hunger Games"]
    assert sig["Coppa Del Nonno"]["module"] == "3-5-2"
    # no alt -> base is the recommendation
    riv["rivals"][0]["alt"] = None
    txt2, _, sig2 = _vs_block(riv)
    assert "già la migliore" in txt2
    assert sig2["Hunger Games"]["module"] == "3-4-3"
    # riposo / unknown rival rows are skipped, empty payload is silent
    assert _vs_block(None) == (None, None, {})
    riv["next_opponents"] = [{"competition": "CDN", "opponent": None}]
    assert _vs_block(riv) == (None, None, {})
    # latch guard: state written BEFORE the vs field existed must not fire
    base_adv = {"module": "3-4-3", "xi": ["A"], "bench": ["B"]}
    assert _advice_diff(base_adv, {**base_adv, "vs": sig}) is None
    # but a genuine recommendation flip fires with the opponent named
    d = _advice_diff({**base_adv, "vs": sig2}, {**base_adv, "vs": sig})
    assert d and "Contro Munnezz FC" in d and "3-5-2" in d and "Schmid" in d


_INDISP_HTML = """
<span class="team-name">Lecce</span>
<div class="col"><header><a aria-label="Infortunati">Infortunati</a></header>
<ul><li><strong class="item-name">Geubbels</strong>
<div class="item-description"><p>KO, lo terr&agrave; fuori luned&igrave;.
Tempi di recupero da valutare.</p></div></li></ul></div>
<div class="col"><header><strong class="label">Squalificati</strong></header>
<ul><li><strong class="item-name">Rossi</strong>
<div class="item-description"><p>un turno dal giudice sportivo</p></div></li>
</ul></div>
<div class="col"><header><strong class="label">Diffidati</strong></header>
<ul><li><strong class="item-name">Bianchi</strong>
<div class="item-description"><p>a rischio</p></div></li></ul></div>
<span class="team-name">Roma</span>
<div class="col"><header><a aria-label="Infortunati">Infortunati</a></header>
<ul><li><strong class="item-name">Kon&#xE8; I.</strong>
<div class="item-description"><p>da valutare quotidianamente</p></div></li>
</ul></div>
""" + "".join(f'<span class="team-name">T{i}</span>' for i in range(14)) \
    + '<span class="team-name">Inter</span>'


def test_parse_indisponibili_categories_doubt_and_sentinel():
    from scripts.fantacalcio.probabili import parse_indisponibili
    d = parse_indisponibili(_INDISP_HTML)
    lecce = {i["nome"]: i for i in d["teams"]["Lecce"]}
    # "terra' fuori" is a hard out even with "da valutare" in the tail
    assert lecce["Geubbels"]["status"] == "infortunato"
    assert lecce["Rossi"]["status"] == "squalificato"
    assert "Bianchi" not in lecce          # diffidati column skipped
    roma = {i["nome"]: i for i in d["teams"]["Roma"]}
    assert roma["Konè I."]["status"] == "infortunato_dubbio"  # unescaped
    # sentinel: a page without enough club blocks is a schema break
    assert parse_indisponibili('<span class="team-name">Inter</span>') is None


def test_apply_availability_hierarchy_and_fold():
    from scripts.fantacalcio.xi_advisor import _apply_availability
    avail = {"teams": {
        "Lecce": [{"nome": "Geubbels", "status": "infortunato", "note": "ko"},
                  {"nome": "Rossi", "status": "squalificato", "note": "1t"}],
        "Roma": [{"nome": "Konè I.", "status": "infortunato_dubbio",
                  "note": "50-50"},
                 {"nome": "Esposito", "status": "infortunato", "note": "x"}],
    }}
    rows = [
        # probabili wins outright, injury rides along as the conflict flag
        {"nome": "Geubbels", "team": "Lecce", "p_play": 0.88,
         "p_play_src": "probabili"},
        {"nome": "Rossi", "team": "Lecce", "p_play": 0.7,
         "p_play_src": "model"},
        # accent fold: listone Koné vs page Konè
        {"nome": "Koné I.", "team": "Roma", "p_play": 0.8,
         "p_play_src": "model"},
        # two same-surname teammates -> ambiguous -> fail open
        {"nome": "Esposito Pio", "team": "Roma", "p_play": 0.9,
         "p_play_src": "model"},
        {"nome": "Esposito Seb.", "team": "Roma", "p_play": 0.9,
         "p_play_src": "model"},
        # nothing structured -> news tier caps, never zeroes
        {"nome": "Verdi", "team": "Milan", "p_play": 0.9,
         "p_play_src": "model"},
        {"nome": "Gialli", "team": "Milan", "p_play": 0.3,
         "p_play_src": "model"},
        {"nome": "Neri", "team": "Milan", "p_play": 0.9,
         "p_play_src": "model", "departed": True},
    ]
    _apply_availability(rows, avail,
                        news_caps={"Verdi": "infortunio", "Gialli": "infortunio",
                                   "Neri": "infortunio"})
    by = {r["nome"]: r for r in rows}
    assert by["Geubbels"]["p_play"] == 0.88 \
        and by["Geubbels"]["p_play_src"] == "probabili" \
        and "infortunato" in by["Geubbels"]["avail_note"]
    assert by["Rossi"]["p_play"] == 0.02 \
        and by["Rossi"]["p_play_src"] == "squalificato_sito"
    assert by["Koné I."]["p_play"] == 0.35 \
        and by["Koné I."]["p_play_src"] == "infortunio_dubbio"
    assert by["Esposito Pio"]["p_play_src"] == "model"      # fail open
    assert by["Verdi"]["p_play"] == 0.6 \
        and by["Verdi"]["p_play_src"] == "news_risk"
    assert by["Gialli"]["p_play"] == 0.3                     # cap, no raise
    assert by["Neri"]["p_play"] == 0.9                       # departed skip
