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
    gk_pp = next(x["p_play"] for x in adv["xi"] if x["R"] == "P")
    assert adv["total"] == pytest.approx(
        sum(x["exp"] for x in adv["xi"]) + adv["modifier"]
        + adv["p_cs"] * gk_pp, abs=0.15)


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
    # inside a role group, slots descend by exp (that IS the entry order:
    # auto-subs skip no-voto players, so raw expected voto ranks the subs)
    for r in "DCA":
        grp = [x["exp"] for x in adv["bench"] if x["R"] == r]
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
    # s0 and s2 carry the page's titolarita bar (s0's ballot must still win);
    # r0 is a 60% super-sub — the bar beats the flat P_RESERVE for him.
    def bar(i):
        pct = {0: 35, 2: 65}.get(i)
        return (f'<div class="progress"><div class="progress-bar" '
                f'aria-valuenow="{pct}"></div></div>') if pct is not None else ""
    starters = "".join(
        f'<li><a href="/serie-a/squadre/x/s{i}/{base_pid + i}" class="player-name">'
        f"<span>S{i}</span></a>{bar(i)}</li>" for i in range(11))
    reserves = "".join(
        f'<li><a href="/serie-a/squadre/x/r{i}/{base_pid + 50 + i}" class="player-name">'
        f"<span>R{i}</span></a>"
        + ('<div class="progress-bar" aria-valuenow="60"></div>' if i == 0 else "")
        + "</li>" for i in range(4))
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
    # ballot beats the bar (s0 has both: ballot 40, bar 35)
    assert p_play_override(1000, 0.5, by_pid) == (0.40, "ballottaggio")
    assert p_play_override(1001, 0.5, by_pid) == (BALLOT_CLAMP[1], "ballottaggio")
    # the page's own titolarita bar beats the flat constants, both ways
    assert p_play_override(1002, 0.5, by_pid) == (0.65, "titolarita")
    assert p_play_override(1050, 0.5, by_pid) == (0.60, "titolarita")
    # no bar on the row -> the flat fallback survives
    assert p_play_override(1003, 0.5, by_pid) == (P_STARTER, "probabili")
    assert p_play_override(1051, 0.5, by_pid) == (P_RESERVE, "probabili")
    assert p_play_override(424242, 0.37, by_pid) == (0.37, "model")


# ---- SosFanta second source (markup verbatim from the 2026-09-03 live
# specimen: h2+formation header pairs, two-ul grids per section, pct badge
# before the truncate name span, ballots as "A - B" with paired badges) ----

from scripts.fantacalcio.probabili import parse_sosfanta  # noqa: E402


def _sf_pl(pct, nome):
    return (f'<li><div><span class="badge"> {pct}% </span>'
            f'<span class="text-sm truncate">{nome}</span></div></li>')


def _sf_match(i):
    home, away = f"Casa{i}", f"Ospite{i}"
    xi_h = "".join(_sf_pl(95, f"H{i}n{j}") for j in range(11))
    xi_a = "".join(_sf_pl(95, f"A{i}n{j}") for j in range(11))
    ul = '<ul class="flex flex-col min-w-0">'
    body = (
        f'<h2 class="t truncate">{home}</h2>'
        f'<span class="s text-primary">3-5-2</span>'
        f'<time datetime="2026-09-0{4 + i % 3}T18:45:00.000Z"></time>'
        f'<h2 class="t truncate">{away}</h2>'
        f'<span class="s text-primary">4-2-3-1</span>'
        f"<h3>Titolari</h3>{ul}{xi_h}</ul>{ul}{xi_a}</ul>")
    if i == 0:
        body += (
            f"<h3>Ballottaggi</h3>{ul}"
            '<li><div><span> 60% </span><span aria-hidden="true">-</span>'
            '<span> 40% </span></div>'
            '<span class="x truncate"> H0n0 - Panchinaro </span></li>'
            f"</ul>{ul}</ul>"
            f"<h3>Panchina</h3>{ul}{_sf_pl(40, 'Panchinaro')}"
            f"{_sf_pl(5, 'Terzo')}</ul>{ul}</ul>"
            f"<h3>In dubbio</h3>{ul}"
            '<li><div><span class="x truncate">Forse</span></div>'
            '<span class="text-[#7b809a]">problema muscolare</span></li>'
            f"</ul>{ul}</ul>"
            f"<h3>Indisponibili</h3>{ul}"
            '<li><div><span class="x truncate">Rotto</span>'
            '<span title="Infortunato"><img></span></div>'
            '<span class="text-[#7b809a]">stagione finita</span></li>'
            '<li><div><span class="x truncate">Boh</span></div></li>'
            f"</ul>{ul}</ul>")
    return body


def _sf_page(n_matches=8):
    return "<html>" + "".join(_sf_match(i) for i in range(n_matches)) + "</html>"


def test_sosfanta_parse_teams_ballots_and_status():
    d = parse_sosfanta(_sf_page())
    assert d is not None and len(d["teams"]) == 16 and len(d["matches"]) == 8
    assert d["matches"][0] == {"home": "Casa0", "away": "Ospite0",
                               "kickoff": "2026-09-04T18:45:00.000Z"}
    t = d["teams"]["Casa0"]
    assert t["formation"] == "3-5-2"
    assert d["teams"]["Ospite0"]["formation"] == "4-2-3-1"
    # titolari pct wins over the same player's ballot entry (setdefault)
    assert t["players"]["H0n0"] == 95
    # ballot loser folded in from the pair; bench keeps its own badge
    assert t["players"]["Panchinaro"] == 40 and t["players"]["Terzo"] == 5
    assert t["doubt"] == [{"nome": "Forse", "status": "indisponibile",
                           "note": "problema muscolare"}]
    assert t["out"] == [
        {"nome": "Rotto", "status": "infortunato", "note": "stagione finita"},
        {"nome": "Boh", "status": "indisponibile", "note": ""}]
    # away side of match 0 has none of Casa0's sections bleeding in
    assert "Panchinaro" not in d["teams"]["Ospite0"]["players"]


def _rig_card(name, base_pid):
    takers = "".join(
        f'<li><a href="/serie-a/squadre/x/t{k}/{base_pid + k}" '
        f'class="player-name"><span>T{k}</span></a></li>' for k in range(3))
    return (f'<div class="card team-card"><span class="team-name">{name}'
            f'</span><header class="primary">Rigori</header>'
            f'<ol class="pill-list ranked">{takers}</ol></div>')


def test_rigoristi_parse_ranks_and_sentinel():
    from scripts.fantacalcio.probabili import parse_rigoristi, rigoristi_by_pid
    page = "".join(_rig_card(f"Team{i}", 2000 + 100 * i) for i in range(20))
    d = parse_rigoristi(page)
    assert d is not None and len(d["teams"]) == 20
    assert d["teams"]["Team0"] == [
        {"pid": 2000, "nome": "T0", "rank": 1},
        {"pid": 2001, "nome": "T1", "rank": 2},
        {"pid": 2002, "nome": "T2", "rank": 3}]
    assert rigoristi_by_pid(d)[2101] == 2
    # 2 cards is a broken page — cache fallback, never {}
    assert parse_rigoristi(_rig_card("A", 1) + _rig_card("B", 5)) is None
    assert parse_rigoristi("<html>maintenance</html>") is None


def test_rigorista_bump_flows_into_exp():
    """Rank-1 taker outranks an otherwise IDENTICAL teammate by exactly the
    declared premium; rank 3 gets the flag but no bump (refit handle only)."""
    from scripts.fantacalcio.xi_advisor import RIGORISTA_BONUS, _apply_rigoristi, advise
    def mk(pid, nome):
        return {"id": pid, "nome": nome, "R": "A", "team": "Genoa",
                "level": 6.5, "voto": 6.0, "sd": 1.0, "p_play": 1.0}
    rows = ([mk(1, "Taker"), mk(2, "Vice"), mk(3, "Terzo"), mk(4, "Nessuno")]
            + [mk(10 + i, f"D{i}") for i in range(8)])
    for i, r in enumerate(rows[4:]):
        r.update(R="P" if i == 0 else ("D" if i < 5 else "C"), level=6.0)
    _apply_rigoristi(rows, {1: 1, 2: 2, 3: 3})
    assert rows[0]["rig_bonus"] == RIGORISTA_BONUS[1]
    assert rows[1]["rig_bonus"] == RIGORISTA_BONUS[2]
    assert rows[2] == {**mk(3, "Terzo"), "rigorista": 3}   # flag, no bump
    assert "rig_bonus" not in rows[3]
    adv = advise(rows, {"Genoa": {"opp": "Como", "home": 1}},
                 {"Genoa": 1500.0, "Como": 1500.0}, {})
    by = {x["nome"]: x for x in adv["xi"] + adv["bench"]}
    assert round(by["Taker"]["exp"] - by["Nessuno"]["exp"], 2) == 0.20
    assert round(by["Vice"]["exp"] - by["Nessuno"]["exp"], 2) == 0.06


def test_sosfanta_schema_break_returns_none_not_empty():
    """A 2-match page is a broken page (or the 2015 fossil article) — cache
    fallback, never {}."""
    assert parse_sosfanta(_sf_page(n_matches=2)) is None
    assert parse_sosfanta("<html>maintenance</html>") is None


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
    gk_pp = next(x["p_play"] for x in up["xi"] if x["R"] == "P")
    assert up["total"] == pytest.approx(
        sum(x["exp_slot"] for x in up["xi"]) + up["modifier"]
        + up["p_cs"] * gk_pp, abs=0.02)
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
        "Napoli": [{"nome": "Anguissa", "status": "infortunato",
                    "note": "ko"}],
        "Inter": [{"nome": "Esposito Pio", "status": "infortunato",
                   "note": "ko"}],
        "Milan": [{"nome": "Bilbao", "status": "infortunato", "note": "ko"}],
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
        # two same-surname teammates (initials stripped by the surname
        # key) vs a bare page 'Esposito' -> ambiguous -> fail open
        {"nome": "Esposito F.", "team": "Roma", "p_play": 0.9,
         "p_play_src": "model"},
        {"nome": "Esposito M.", "team": "Roma", "p_play": 0.9,
         "p_play_src": "model"},
        # nothing structured -> news tier caps, never zeroes
        {"nome": "Verdi", "team": "Milan", "p_play": 0.9,
         "p_play_src": "model"},
        {"nome": "Gialli", "team": "Milan", "p_play": 0.3,
         "p_play_src": "model"},
        {"nome": "Neri", "team": "Milan", "p_play": 0.9,
         "p_play_src": "model", "departed": True},
        # page 'Anguissa' must reach listone 'Zambo Anguissa' (last token)
        {"nome": "Zambo Anguissa", "team": "Napoli", "p_play": 0.9,
         "p_play_src": "model"},
        # exact page name picks the right one of two same-surname teammates
        {"nome": "Esposito Pio", "team": "Inter", "p_play": 0.9,
         "p_play_src": "model"},
        {"nome": "Esposito Seb.", "team": "Inter", "p_play": 0.9,
         "p_play_src": "model"},
        # a departed same-surname ghost must not block his teammate's match
        {"nome": "Bilbao B.", "team": "Milan", "p_play": 0.9,
         "p_play_src": "model"},
        {"nome": "Bilbao Z.", "team": "Milan", "p_play": 0.9,
         "p_play_src": "model", "departed": True},
        # listed player with breaking news keeps p but gets the flag
        {"nome": "Blu", "team": "Milan", "p_play": 0.88,
         "p_play_src": "probabili"},
    ]
    _apply_availability(rows, avail,
                        news_caps={"Verdi": "infortunio", "Gialli": "infortunio",
                                   "Neri": "infortunio", "Blu": "infortunio"})
    by = {r["nome"]: r for r in rows}
    assert by["Geubbels"]["p_play"] == 0.88 \
        and by["Geubbels"]["p_play_src"] == "probabili" \
        and "infortunato" in by["Geubbels"]["avail_note"]
    assert by["Rossi"]["p_play"] == 0.02 \
        and by["Rossi"]["p_play_src"] == "squalificato_sito"
    assert by["Koné I."]["p_play"] == 0.35 \
        and by["Koné I."]["p_play_src"] == "infortunio_dubbio"
    assert by["Esposito F."]["p_play_src"] == "model"       # fail open
    assert by["Esposito M."]["p_play_src"] == "model"
    assert by["Verdi"]["p_play"] == 0.6 \
        and by["Verdi"]["p_play_src"] == "news_risk"
    assert by["Gialli"]["p_play"] == 0.3                     # cap, no raise
    assert by["Neri"]["p_play"] == 0.9                       # departed skip
    assert by["Zambo Anguissa"]["p_play_src"] == "infortunio_sito"
    assert by["Esposito Pio"]["p_play_src"] == "infortunio_sito"
    assert by["Esposito Seb."]["p_play_src"] == "model"      # untouched
    assert by["Bilbao B."]["p_play_src"] == "infortunio_sito"
    assert by["Blu"]["p_play"] == 0.88 \
        and by["Blu"]["p_play_src"] == "probabili" \
        and by["Blu"]["avail_note"] == "news: infortunio"


def test_apply_sosfanta_combines_sources_per_label():
    from scripts.fantacalcio.xi_advisor import _apply_sosfanta
    sf = {"teams": {"Genoa": {"players": {
        "Tito": 90, "Ballo": 70, "Modello": 55, "Flat": 25, "Certo": 99}}}}
    rows = [
        {"nome": "Tito", "team": "Genoa", "p_play": 0.8,
         "p_play_src": "titolarita"},
        {"nome": "Ballo", "team": "Genoa", "p_play": 0.5,
         "p_play_src": "ballottaggio"},
        {"nome": "Modello", "team": "Genoa", "p_play": 0.4,
         "p_play_src": "model"},
        {"nome": "Flat", "team": "Genoa", "p_play": 0.88,
         "p_play_src": "probabili"},
        {"nome": "Certo", "team": "Genoa", "p_play": 0.4,
         "p_play_src": "model"},
        {"nome": "Assente", "team": "Genoa", "p_play": 0.4,
         "p_play_src": "model"},
        {"nome": "Altro", "team": "Como", "p_play": 0.4,
         "p_play_src": "model"},
    ]
    _apply_sosfanta(rows, sf)
    by = {r["nome"]: r for r in rows}
    assert by["Tito"]["p_play"] == 0.85 \
        and by["Tito"]["p_play_src"] == "titolarita2"
    assert by["Ballo"]["p_play"] == 0.6 \
        and by["Ballo"]["p_play_src"] == "ballottaggio2"
    assert by["Modello"]["p_play"] == 0.55 \
        and by["Modello"]["p_play_src"] == "sosfanta"
    # one measurement beats the flat P_STARTER/P_RESERVE constant
    assert by["Flat"]["p_play"] == 0.25 \
        and by["Flat"]["p_play_src"] == "sosfanta"
    assert by["Certo"]["p_play"] == 0.95            # clamp
    assert by["Assente"]["p_play_src"] == "model"   # unmatched untouched
    assert by["Altro"]["p_play_src"] == "model"     # club not on the page


def test_titolarita_listing_survives_injury_page():
    """Restores the documented hierarchy: a probabili LISTING (any pct
    label) beats the injury page, which only rides along as avail_note.
    Pre-2026-09-03 the skip tuple lacked 'titolarita', so every pct-listed
    player (393/479 listings) was silently zeroed by the injury page."""
    from scripts.fantacalcio.xi_advisor import P_OUT, _apply_availability
    avail = {"teams": {"Genoa": [
        {"nome": "Vitinha", "status": "infortunato", "note": "ko"}]}}
    # TRUE POSITIVE first: an unlisted row IS dropped for this exact
    # name/club shape — proves the injury match actually fires here.
    unlisted = [{"nome": "Vitinha", "team": "Genoa", "p_play": 0.9,
                 "p_play_src": "model"}]
    _apply_availability(unlisted, avail)
    assert unlisted[0]["p_play"] == P_OUT \
        and unlisted[0]["p_play_src"] == "infortunio_sito"
    # the fix: same player listed with a pct keeps his listing
    listed = [{"nome": "Vitinha", "team": "Genoa", "p_play": 0.62,
               "p_play_src": "titolarita"}]
    _apply_availability(listed, avail)
    assert listed[0]["p_play"] == 0.62 \
        and listed[0]["p_play_src"] == "titolarita"
    assert "infortunato" in listed[0]["avail_note"]
    # end-to-end with the second source: overlay first (mean, relabel),
    # then the injury page still only annotates
    combo = [{"nome": "Vitinha", "team": "Genoa", "p_play": 0.62,
              "p_play_src": "titolarita"}]
    sf = {"teams": {"Genoa": {"players": {"Vitinha": 80}}}}
    _apply_availability(combo, avail, sf=sf)
    assert combo[0]["p_play"] == 0.71 \
        and combo[0]["p_play_src"] == "titolarita2"
    assert "infortunato" in combo[0]["avail_note"]


# ---- anytime-scorer market tilt (T-60) -------------------------------------

def test_pois_cdf_and_team_lambda_solver():
    """The solver must invert its own forward model: build fair odds from a
    known (lam_h, lam_a), recover them."""
    import math

    from scripts.fantacalcio.xi_advisor import _market_team_lambdas, _pois_cdf
    assert abs(_pois_cdf(2, 2.4) - (math.exp(-2.4) * (1 + 2.4 + 2.88))) < 1e-9
    lam_h, lam_a = 1.6, 0.8
    lam_t = lam_h + lam_a
    p_under = _pois_cdf(2, lam_t)
    ph = [math.exp(-lam_h) * lam_h ** i / math.factorial(i) for i in range(13)]
    pa = [math.exp(-lam_a) * lam_a ** i / math.factorial(i) for i in range(13)]
    p_hw = sum(ph[i] * pa[j] for i in range(13) for j in range(i))
    p_aw = sum(ph[i] * pa[j] for i in range(13) for j in range(i + 1, 13))
    p_d = 1 - p_hw - p_aw
    vig = 1.06   # any flat vig must cancel in both de-vig ratios
    odds = {"Casa vs Fuori": {
        "h2h": {"home": vig / p_hw, "draw": vig / p_d, "away": vig / p_aw},
        "totals": [{"line": 2.5, "over": vig / (1 - p_under),
                    "under": vig / p_under},
                   {"line": 2.0, "over": 1.5, "under": 2.5}]}}
    r = _market_team_lambdas("Casa", "Fuori", odds=odds)
    assert r is not None
    assert abs(r[0] - lam_h) < 0.05 and abs(r[1] - lam_a) < 0.05
    # missing markets fail open
    assert _market_team_lambdas("X", "Y", odds={}) is None
    assert _market_team_lambdas("Casa", "Fuori", odds={
        "Casa vs Fuori": {"h2h": {"home": 2.0, "draw": 3.0, "away": 4.0},
                          "totals": [{"line": 2.0, "over": 1.5,
                                      "under": 2.5}]}}) is None


def test_scorer_edges_zero_sum_and_name_ladder(tmp_path, monkeypatch):
    """Full chain on a synthetic raw file: market full names -> board
    surnames + understat nicknames via the folded ladder; shares zero-sum
    per club; the hot-market newcomer gets the positive edge."""
    import json
    import math

    import pandas as pd

    import scripts.fantacalcio.xi_advisor as xa
    (tmp_path / "data" / "upcoming").mkdir(parents=True)
    (tmp_path / "data" / "parsed").mkdir(parents=True)
    (tmp_path / "data" / "fantacalcio").mkdir(parents=True)
    lam_h, lam_a = 1.2, 1.2
    p_under = xa._pois_cdf(2, lam_h + lam_a)
    ph = [math.exp(-lam_h) * lam_h ** i / math.factorial(i) for i in range(13)]
    p_hw = sum(ph[i] * ph[j] for i in range(13) for j in range(i))
    p_d = 1 - 2 * p_hw
    (tmp_path / "data" / "upcoming" / "odds_full.json").write_text(json.dumps(
        {"matches": {"Genoa vs Como": {
            "h2h": {"home": 1 / p_hw, "draw": 1 / p_d, "away": 1 / p_hw},
            "totals": [{"line": 2.5, "over": 1 / (1 - p_under),
                        "under": 1 / p_under}]}}}))
    raw = {"events": {"ev1": {
        "home": "Genoa", "away": "Como",
        "commence": "2026-09-04T18:45:00Z",
        "fetched_at": "2026-09-03T21:00:00+00:00",
        # Vecchio: market 25%, history says he IS the man (0.5 g/app).
        # Nuovo: market 25% but 2-match history — shrinks to market, ~0 edge.
        # Gregario: market 5%, history 0.05 — small negative absorbs the rest.
        "prices": {"Anastasios Vecchio": [4.0, 4.0],
                   "Nico Nuovo": [4.0],
                   "Luca Gregario": [20.0],
                   "Carlos Ospite": [3.0]}}}}
    board = {"players": [
        {"id": 1, "nome": "Vecchio", "R": "A", "team": "Genoa", "status": "OK"},
        {"id": 2, "nome": "Nuovo", "R": "A", "team": "Genoa", "status": "OK"},
        {"id": 3, "nome": "Gregario", "R": "C", "team": "Genoa", "status": "OK"},
        {"id": 4, "nome": "Ospite", "R": "A", "team": "Como", "status": "OK"},
        {"id": 5, "nome": "Fuori Rosa", "R": "A", "team": "Genoa",
         "status": "DEPARTED"}]}
    (tmp_path / "data" / "fantacalcio" / "auction_board.json").write_text(
        json.dumps(board))
    pd.DataFrame([
        {"league": "ITA-Serie A", "season": "2025-2026", "team": "Genoa",
         "player": "Tasos Vecchio", "matches": 38, "goals": 19},
        {"league": "ITA-Serie A", "season": "2026-2027", "team": "Genoa",
         "player": "Tasos Vecchio", "matches": 2, "goals": 1},
        {"league": "ITA-Serie A", "season": "2026-2027", "team": "Genoa",
         "player": "Nico Nuovo", "matches": 2, "goals": 0},
        {"league": "ITA-Serie A", "season": "2025-2026", "team": "Genoa",
         "player": "Luca Gregario", "matches": 38, "goals": 2},
        {"league": "ITA-Serie A", "season": "2025-2026", "team": "Como",
         "player": "Carlos Ospite", "matches": 30, "goals": 9},
    ]).to_parquet(tmp_path / "data" / "parsed" / "understat_players.parquet")
    monkeypatch.setattr(xa, "ROOT", tmp_path)
    monkeypatch.setattr(xa, "SCORER_RAW",
                        tmp_path / "data" / "fantacalcio" / "raw.json")
    monkeypatch.setattr(xa, "SCORER_EDGES",
                        tmp_path / "data" / "fantacalcio" / "edges.json")
    monkeypatch.setattr(xa, "BOARD",
                        tmp_path / "data" / "fantacalcio" / "auction_board.json")
    xa.SCORER_RAW.write_text(json.dumps(raw))
    out = xa.build_scorer_edges()
    assert out is not None and out["matches"][0]["matched"] == 4
    by = {int(k): v for k, v in out["by_pid"].items()}
    genoa = [by[i]["edge"] for i in (1, 2, 3)]
    assert abs(sum(genoa)) < 0.01                     # zero-sum per club
    # the proven scorer's history out-argues the market -> negative edge;
    # the 2-match newcomer shrinks to the market -> near zero
    assert by[1]["edge"] < -0.05
    assert abs(by[2]["edge"]) < abs(by[1]["edge"])
    assert by[1]["n_app"] == 40                       # nickname matched, pooled
    # Como's single priced player IS the whole share -> edge exactly 0
    assert by[4]["edge"] == 0.0
    assert 5 not in by                                # departed never priced


def test_apply_scorer_tilt_replaces_rig_bonus():
    from scripts.fantacalcio.xi_advisor import _apply_scorer, advise
    rows = [
        {"id": 1, "nome": "Taker", "R": "A", "team": "Genoa", "level": 6.5,
         "voto": 6.0, "sd": 1.0, "p_play": 1.0, "rigorista": 1,
         "rig_bonus": 0.20},
        {"id": 2, "nome": "Altro", "R": "A", "team": "Genoa", "level": 6.5,
         "voto": 6.0, "sd": 1.0, "p_play": 1.0, "rigorista": 2,
         "rig_bonus": 0.06},
    ]
    _apply_scorer(rows, {1: {"edge": -0.10, "lam_mkt": 0.2, "lam_own": 0.27}})
    assert rows[0]["scorer_edge"] == -0.10 and "rig_bonus" not in rows[0]
    assert rows[0]["rigorista"] == 1                  # rank kept for ledger
    assert rows[1]["rig_bonus"] == 0.06               # unpriced untouched
    fillers = [{"id": 10 + i, "nome": f"X{i}", "R": r, "team": "Genoa",
                "level": 6.0, "voto": 6.0, "sd": 1.0, "p_play": 1.0}
               for i, r in enumerate("P" + "D" * 4 + "C" * 3 + "A")]
    adv = advise(rows + fillers, {"Genoa": {"opp": "Como", "home": 1}},
                 {"Genoa": 1500.0, "Como": 1500.0}, {})
    by = {x["nome"]: x for x in adv["xi"] + adv["bench"]}
    # market tilt (negative) REPLACED the +0.20 rig bonus for Taker;
    # Altro still enjoys his +0.06
    assert round(by["Altro"]["exp"] - by["Taker"]["exp"], 2) == 0.16


def test_scorer_events_due_window_and_dedup():
    from datetime import UTC, datetime

    from scripts.data.odds_fetcher import _scorer_events_due
    now = datetime(2026, 9, 4, 17, 45, tzinfo=UTC)
    evs = [
        {"id": "a", "commence_time": "2026-09-04T18:45:00Z"},   # T-60: due
        {"id": "b", "commence_time": "2026-09-04T21:00:00Z"},   # outside
        {"id": "c", "commence_time": "2026-09-04T17:35:00Z"},   # just kicked
        {"id": "d", "commence_time": "2026-09-04T18:30:00Z"},   # fresh fetch
        {"id": "e", "commence_time": "2026-09-04T18:30:00Z"},   # stale fetch
        {"id": "bad"},
    ]
    store = {"events": {
        "d": {"fetched_at": "2026-09-04T17:30:00+00:00"},
        "e": {"fetched_at": "2026-09-04T16:30:00+00:00"}}}
    due = [e["id"] for e in _scorer_events_due(evs, store, now, 1.25)]
    assert due == ["a", "c", "e"]


# ---------- blind-spots batch: silent-death visibility (Fix 1) ----------

def test_feed_age_h():
    from datetime import UTC, datetime, timedelta

    from scripts.fantacalcio.probabili import feed_age_h
    now = datetime.now(UTC)
    assert feed_age_h(None) is None
    assert feed_age_h({}) is None
    assert feed_age_h({"fetched_at": "garbage"}) is None
    fresh = feed_age_h({"fetched_at": now.isoformat()})
    assert fresh is not None and 0 <= fresh < 0.1
    old = feed_age_h({"fetched_at": (now - timedelta(hours=30)).isoformat()})
    assert old is not None and 29.9 < old < 30.1


def test_feed_age_line():
    from scripts.fantacalcio.tracker import _feed_age_line
    assert _feed_age_line(None) is None
    assert _feed_age_line({}) is None
    ok = _feed_age_line({"probabili_h": 2.4, "indisponibili_h": 3.0})
    assert ok is not None and "2h" in ok and "3h" in ok and "⚠" not in ok
    warn = _feed_age_line({"probabili_h": 30.0, "indisponibili_h": 1.0})
    assert warn is not None and "⚠" in warn
    # a feed that never fetched must read as a warning, not crash
    never = _feed_age_line({"probabili_h": None, "indisponibili_h": 2.0})
    assert never is not None and "⚠" in never and "mai" in never


def test_write_heartbeat(tmp_path):
    import json as _json

    from scripts.fantacalcio.tracker import _write_heartbeat
    p = tmp_path / "hb.json"
    _write_heartbeat(True, None, path=p)
    hb = _json.loads(p.read_text())
    assert hb["ok"] is True and hb["error"] is None and hb["ran_at"]
    _write_heartbeat(False, "boom", path=p)
    hb = _json.loads(p.read_text())
    assert hb["ok"] is False and hb["error"] == "boom"


def test_fanta_health_gating():
    from scripts.pipeline.monitor import _fanta_health
    # dead job trumps everything, season or not
    assert _fanta_health({"ok": True}, 31.0, 1.0, 1.0, None)["status"] == "CRITICAL"
    # ran-and-crashed is critical
    r = _fanta_health({"ok": False, "error": "boom"}, 1.0, 1.0, 1.0, 3.0)
    assert r["status"] == "CRITICAL" and "boom" in r["detail"]
    # off-season: frozen feeds are legitimate — no alarm
    assert _fanta_health({"ok": True}, 2.0, 500.0, 500.0, None)["status"] == "OK"
    # in-season stale feed warns
    r = _fanta_health({"ok": True}, 2.0, 31.0, 1.0, 2.0)
    assert r["status"] == "WARNING" and "probabili" in r["detail"]
    # in-season never-fetched feed warns too (missing != fine)
    assert _fanta_health({"ok": True}, 2.0, -1.0, 1.0, 2.0)["status"] == "WARNING"
    # healthy in-season
    assert _fanta_health({"ok": True}, 2.0, 3.0, 3.0, 2.0)["status"] == "OK"
    # tracker alive but its push layer erroring -> visible as warning
    r = _fanta_health({"ok": True, "error": "push: 500"}, 2.0, 3.0, 3.0, 2.0)
    assert r["status"] == "WARNING"
    # no heartbeat yet
    assert _fanta_health(None, -1.0, 1.0, 1.0, 2.0)["status"] == "WARNING"


# ---------- blind-spots batch: bench order (Fix 3) ----------

def test_bench_order_ranks_by_exp_given_plays_not_exp_slot():
    """Auto-subs SKIP a bench player without a voto, so within a role the
    entry order must put the best E[voto|plays] first even at low p_play —
    absent costs nothing, present is the best sub. Selection is untouched."""
    from scripts.fantacalcio.xi_advisor import _bench_split
    def mk(n, r, exp, pp):
        return {"nome": n, "R": r, "exp": exp,
                "exp_slot": round(pp * exp, 2), "p_play": pp}
    dubbio_star = mk("Star", "D", 7.5, 0.35)     # exp_slot 2.62
    solid = mk("Solid", "D", 6.0, 0.90)          # exp_slot 5.40
    ok = mk("Ok", "D", 5.8, 0.88)                # exp_slot 5.10
    cand = [dubbio_star, solid, ok,
            mk("GK2", "P", 5.5, 0.9),
            mk("M1", "C", 6.0, 0.9), mk("M2", "C", 5.9, 0.9),
            mk("M3", "C", 5.8, 0.9), mk("A1", "A", 6.5, 0.9),
            mk("A2", "A", 6.4, 0.9)]
    bench, trib = _bench_split(cand, xi=[])
    d_order = [x["nome"] for x in bench if x["R"] == "D"]
    # rejection-test precondition: the old exp_slot order really differs
    old = [x["nome"] for x in sorted([dubbio_star, solid, ok],
                                     key=lambda x: -x["exp_slot"])]
    assert old == ["Solid", "Ok", "Star"]
    assert d_order == ["Star", "Solid", "Ok"]
    assert trib == []


# ---------- blind-spots batch: official lineups (Fix 2) ----------

def _mini_confirmed():
    return {"fetched_at": "2026-09-05T13:00:00+00:00", "matches": {
        "Lecce vs Roma": {
            "home_lineup": ["Wladimiro Falcone", "Tiago Gabriel", "K. One",
                            "A B", "C D", "E F", "G H", "I J", "K L",
                            "M N", "O P"],
            "home_bench": ["Nikola Stulic"],
            "away_lineup": ["Mile Svilar", "Evan Ndicka",
                            "Andre-Frank Zambo Anguissa", "Q R", "S T",
                            "U V", "W X", "Y Z", "A2 B2", "C2 D2", "E2 F2"],
            "away_bench": ["Stephan El Shaarawy"],
        }}}


def test_official_overrides_matching_and_tiers():
    from scripts.fantacalcio.lineup_check import (
        P_OFFICIAL_BENCH,
        P_OFFICIAL_XI,
        _official_overrides,
    )
    players = [
        {"id": 1, "nome": "Falcone", "team": "Lecce"},
        {"id": 2, "nome": "Stulić", "team": "Lecce"},          # accent folds
        {"id": 3, "nome": "Svilar", "team": "Roma"},
        {"id": 4, "nome": "Zambo Anguissa", "team": "Roma"},   # 2-token suffix
        {"id": 5, "nome": "N'Dicka", "team": "Roma"},
    ]
    out = _official_overrides(_mini_confirmed(), players)
    assert out[1]["src"] == "official_xi" and out[1]["p_play"] == P_OFFICIAL_XI
    assert out[2]["src"] == "official_bench" and out[2]["p_play"] == P_OFFICIAL_BENCH
    assert out[3]["src"] == "official_xi"
    assert out[4]["src"] == "official_xi"


def test_official_out_needs_coverage_and_ambiguity_fails_open():
    from scripts.fantacalcio.lineup_check import _official_overrides
    # coverage case: 3 of 4 Lecce board rows match -> the 4th is a real exclusion
    players = [
        {"id": 1, "nome": "Falcone", "team": "Lecce"},
        {"id": 2, "nome": "Stulić", "team": "Lecce"},
        {"id": 6, "nome": "Gabriel T.", "team": "Lecce"},
        {"id": 7, "nome": "Ghostinho", "team": "Lecce"},   # in neither list
    ]
    out = _official_overrides(_mini_confirmed(), players)
    assert out[7]["src"] == "official_out"
    # low-coverage club: unmatched names must NOT be branded excluded
    players_low = [
        {"id": 10, "nome": "Nessuno A", "team": "Roma"},
        {"id": 11, "nome": "Nessuno B", "team": "Roma"},
        {"id": 12, "nome": "Svilar", "team": "Roma"},
    ]
    out2 = _official_overrides(_mini_confirmed(), players_low)
    assert out2[12]["src"] == "official_xi"
    assert 10 not in out2 and 11 not in out2
    # ambiguity: two board rows folding to the same sofa name -> both skipped
    players_amb = [
        {"id": 20, "nome": "Svilar", "team": "Roma"},
        {"id": 21, "nome": "Svilar M.", "team": "Roma"},
    ]
    out3 = _official_overrides(_mini_confirmed(), players_amb)
    assert 20 not in out3 and 21 not in out3


def test_apply_official_beats_every_probabilistic_tier():
    from scripts.fantacalcio.xi_advisor import _apply_official
    rows = [{"id": 1, "nome": "A", "p_play": 0.35, "p_play_src": "infortunio_dubbio"},
            {"id": 2, "nome": "B", "p_play": 0.88, "p_play_src": "probabili"},
            {"id": 3, "nome": "C", "p_play": 0.15, "p_play_src": "probabili"}]
    _apply_official(rows, {1: {"p_play": 0.97, "src": "official_xi"},
                           2: {"p_play": 0.03, "src": "official_out"}})
    assert rows[0]["p_play"] == 0.97 and rows[0]["p_play_src"] == "official_xi"
    assert rows[1]["p_play"] == 0.03 and rows[1]["p_play_src"] == "official_out"
    assert rows[2]["p_play"] == 0.15   # untouched


# ---------- blind-spots batch: roster export age (Fix 6) ----------

def test_rosters_age_line(tmp_path):
    import os
    import time

    from scripts.fantacalcio.tracker import _rosters_age_line
    p = tmp_path / "league_rosters.json"
    p.write_text("{}")
    assert _rosters_age_line(path=p) is None                    # fresh
    old = time.time() - 20 * 86400
    os.utime(p, (old, old))
    line = _rosters_age_line(path=p)
    assert line is not None and "20 giorni" in line
    assert _rosters_age_line(path=tmp_path / "missing.json") is None


def test_first_push_state_preserves_sibling_latches():
    """Regression for the 2026-09-02 double-send: the first-push write used a
    fresh literal dict, wiping digest_round/risk_alerts so the next run
    re-sent both. The latch update must preserve every sibling key."""
    from scripts.fantacalcio.tracker import _first_push_state
    state = {"digest_round": 2,
             "risk_alerts": {"Simeone|mercato-out": "2026-09-02T19:47:08Z"},
             "official_sig": "old"}
    cur = {"module": "3-5-2", "xi": ["A"], "bench": ["B"], "vs": None}
    out = _first_push_state(state, 3, cur)
    assert out is state
    assert out["digest_round"] == 2
    assert out["risk_alerts"] == {"Simeone|mercato-out": "2026-09-02T19:47:08Z"}
    assert out["round"] == 3 and out["final_checked"] is False
    assert out["advice"] == cur and out["sent_at"]


# ---------- bring-to-9: substitution-aware objective ----------

def test_role_recovery_formula():
    from scripts.fantacalcio.xi_advisor import _role_recovery
    chain = [{"p_play": 0.35, "exp": 7.5},
             {"p_play": 0.9, "exp": 6.0},
             {"p_play": 0.88, "exp": 5.8}]
    expect = 0.35 * 7.5 + 0.65 * (0.9 * 6.0 + 0.1 * 0.88 * 5.8)
    assert abs(_role_recovery(chain) - expect) < 1e-9
    assert _role_recovery([]) == 0.0


def test_bench_set_selection_prefers_recoverable_star():
    """A dubbio star (7.5 at 35%) belongs on the bench over a safe mediocrity:
    absent he is skipped for free, present he is the best sub. The old
    exp_slot top-n selection excluded him — pinned as the rejection case."""
    from scripts.fantacalcio.xi_advisor import _best_bench_for_role
    def mk(n, exp, pp):
        return {"nome": n, "R": "D", "exp": exp, "p_play": pp,
                "exp_slot": round(pp * exp, 2)}
    star, solid = mk("Star", 7.5, 0.35), mk("Solid", 6.0, 0.90)
    ok, meh = mk("Ok", 5.8, 0.88), mk("Meh", 5.5, 0.85)
    pool = [star, solid, ok, meh]
    old = sorted(sorted(pool, key=lambda x: -x["exp_slot"])[:3],
                 key=lambda x: -x["exp"])
    assert [x["nome"] for x in old] == ["Solid", "Ok", "Meh"]   # precondition
    new = _best_bench_for_role(pool, 3)
    assert [x["nome"] for x in new] == ["Star", "Solid", "Ok"]
    # all-certain pool: recovery only sees the first sub — tie-break must
    # still keep the deepest chain, i.e. plain top-n by exp
    sure = [mk(n, e, 1.0) for n, e in
            (("A", 6.5), ("B", 6.2), ("C", 6.0), ("D", 5.0))]
    assert [x["nome"] for x in _best_bench_for_role(sure, 3)] == ["A", "B", "C"]


def test_bench_recovery_ev_and_exp_total():
    from scripts.fantacalcio.xi_advisor import _bench_recovery_ev, _role_recovery
    xi = [{"R": "D", "p_play": 0.9, "exp": 6.0},
          {"R": "D", "p_play": 0.8, "exp": 6.2},
          {"R": "A", "p_play": 1.0, "exp": 7.0}]
    bench = [{"R": "D", "p_play": 0.9, "exp": 5.9},
             {"R": "D", "p_play": 0.9, "exp": 5.5}]
    # expected D absences 0.3, no A chain -> ev = 0.3 * R(D chain)
    expect = 0.3 * _role_recovery(bench)
    assert abs(_bench_recovery_ev(xi, bench) - expect) < 1e-9
    # certain XI recovers nothing
    assert _bench_recovery_ev([{"R": "D", "p_play": 1.0, "exp": 6.0}],
                              bench) == 0.0


def test_advise_exposes_exp_total():
    adv = advise(_full_squad(), FIX, ELO, {})
    assert adv["exp_total"] == round(adv["total"] + adv["bench_ev"], 2)
    assert adv["bench_ev"] >= 0.0


# ---------- bring-to-9: H2H calibration + ban-cost line ----------

def _mini_riv():
    return {"next_opponents": [
                {"competition": "Coppa Del Nonno", "opponent": "Munnezz FC"},
                {"competition": "Hunger Games", "opponent": "Munnezz FC"},
                {"competition": "X", "opponent": None}],   # riposo -> skipped
            "me": {"team": "Whisky Palermo", "total": 62.4},
            "rivals": [{"team": "Munnezz FC", "p_win": 0.61, "total": 58.1}]}


def test_h2h_forecasts_lifts_per_competition():
    from scripts.fantacalcio.pred_ledger import _h2h_forecasts
    out = _h2h_forecasts(_mini_riv())
    assert [h["competition"] for h in out] == ["Coppa Del Nonno",
                                              "Hunger Games"]
    assert all(h["opponent"] == "Munnezz FC" and h["p_win"] == 0.61
               and h["opp_exp"] == 58.1 for h in out)
    assert _h2h_forecasts(None) == []


def test_h2h_result_sides_and_unplayed():
    from scripts.fantacalcio.pred_ledger import _h2h_result
    fx = {"home": "Whisky Palermo", "away": "Munnezz FC", "score": "1-0",
          "fp_home": 71.5, "fp_away": 64.0}
    r = _h2h_result(fx, "Whisky Palermo")
    assert r["result"] == "W" and r["fp_mine"] == 71.5 and r["goals"] == "1-0"
    r2 = _h2h_result(fx, "Munnezz FC")
    assert r2["result"] == "L" and r2["fp_mine"] == 64.0
    assert _h2h_result({"home": "A", "away": "B", "score": None}, "A") is None
    assert _h2h_result(fx, "Terzo Incomodo") is None


def test_reconcile_h2h_grades_when_scores_appear(tmp_path, monkeypatch):
    import json as _json

    import scripts.fantacalcio.pred_ledger as pl
    monkeypatch.setattr(pl, "LEDGER", tmp_path / "led.json")
    monkeypatch.setattr(pl, "SCHEDULE", tmp_path / "sched.json")
    monkeypatch.setattr(pl, "ROSTERS", tmp_path / "rosters.json")
    (tmp_path / "led.json").write_text(_json.dumps({"rounds": {"3": {
        "first_kickoff": 0,
        "h2h": [{"competition": "Coppa Del Nonno", "opponent": "Munnezz FC",
                 "p_win": 0.61}]}}}))
    (tmp_path / "rosters.json").write_text('{"my_team": "Whisky Palermo"}')
    sched = {"competitions": {"Coppa Del Nonno": {"rounds": [
        {"sa_round": 3, "fixtures": [
            {"home": "Whisky Palermo", "away": "Munnezz FC",
             "score": "-", "fp_home": 0.0, "fp_away": 0.0}], "rests": []}]}}}
    (tmp_path / "sched.json").write_text(_json.dumps(sched))
    assert pl.reconcile_h2h() == []          # unplayed cell grades nothing
    sched["competitions"]["Coppa Del Nonno"]["rounds"][0]["fixtures"][0].update(
        score="2-1", fp_home=74.0, fp_away=69.5)
    (tmp_path / "sched.json").write_text(_json.dumps(sched))
    assert pl.reconcile_h2h() == ["3:Coppa Del Nonno"]
    led = _json.loads((tmp_path / "led.json").read_text())
    h = led["rounds"]["3"]["h2h"][0]
    assert h["result"] == "W" and h["fp_mine"] == 74.0 and h["graded_at"]
    assert pl.reconcile_h2h() == []          # idempotent


def test_ban_cost_line_names_next_round_h2h():
    from scripts.fantacalcio.tracker import _ban_cost_line
    sched = {"competitions": {
        "Coppa Del Nonno": {"rounds": [
            {"sa_round": 4, "fixtures": [], "rests":
                [{"team": "Whisky Palermo"}]}]},
        "Hunger Games": {"rounds": [
            {"sa_round": 4, "fixtures": [
                {"home": "DELICATISSIMI", "away": "Whisky Palermo"}],
             "rests": []}]}}}
    line = _ban_cost_line(3, sched, "Whisky Palermo")
    assert line is not None and "G4" in line
    assert "CDN: riposo" in line and "HG vs DELICATISSIMI" in line
    assert _ban_cost_line(3, sched, None) is None
    assert _ban_cost_line(38, {"competitions": {}}, "Whisky Palermo") is None


# ---------- automation sweep: voti finalization window ----------

def _fr_setup(tmp_path, monkeypatch, rows=350, voto=6.0):
    import pandas as pd

    import scripts.fantacalcio.live_scores as ls
    monkeypatch.setattr(ls, "CACHE", tmp_path)
    df = pd.DataFrame({"pid": range(rows), "voto": [voto] * rows})
    monkeypatch.setattr(ls, "parse", lambda html: df.copy())

    class _R:
        status_code = 200
        text = "<html/>"
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _R()
    import curl_cffi.requests as rq
    monkeypatch.setattr(rq, "get", fake_get)
    return ls, df, calls


def test_fetch_round_finalization_window(tmp_path, monkeypatch):
    """A young cache re-fetches (provisional voti may still be corrected) but
    rewrites ONLY on change, so the mtime ages the file out of the window;
    an old cache never touches the network."""
    import os
    import time

    import pandas as pd
    ls, df, calls = _fr_setup(tmp_path, monkeypatch)
    p = tmp_path / "round_2026_27_03.parquet"

    out = ls.fetch_round("2026-27", 3)           # first fetch: writes
    assert len(out) == 350 and len(calls) == 1 and p.exists()
    mt1 = p.stat().st_mtime

    out = ls.fetch_round("2026-27", 3)           # young + unchanged: fetches,
    assert len(calls) == 2                        # but does NOT rewrite
    assert p.stat().st_mtime == mt1

    df["voto"] = 6.5                              # correction lands: rewrites
    time.sleep(0.01)
    out = ls.fetch_round("2026-27", 3)
    assert len(calls) == 3 and out["voto"].iloc[0] == 6.5
    assert p.stat().st_mtime > mt1

    old = time.time() - (ls.FINALIZE_H + 1) * 3600
    os.utime(p, (old, old))                       # settled: no network at all
    out = ls.fetch_round("2026-27", 3)
    assert len(calls) == 3 and out["voto"].iloc[0] == 6.5


def test_fetch_round_finalization_failures_keep_cache(tmp_path, monkeypatch):
    """During the window a failed or floor-rejected re-fetch must serve the
    cached round, never lose it."""
    ls, df, calls = _fr_setup(tmp_path, monkeypatch)
    ls.fetch_round("2026-27", 3)

    class _Bad:
        status_code = 503
        text = ""
    import curl_cffi.requests as rq
    monkeypatch.setattr(rq, "get", lambda *a, **k: _Bad())
    out = ls.fetch_round("2026-27", 3)
    assert out is not None and len(out) == 350    # 503 -> cached

    monkeypatch.setattr(ls, "parse",
                        lambda html: df.head(10))  # under MIN_ROWS floor
    class _Ok:
        status_code = 200
        text = "<html/>"
    monkeypatch.setattr(rq, "get", lambda *a, **k: _Ok())
    out = ls.fetch_round("2026-27", 3)
    assert out is not None and len(out) == 350    # floor -> cached


def test_refit_lines_threshold_and_table():
    from scripts.fantacalcio.tracker import _refit_lines
    summ = {"rounds": [{"reconciled": True}] * 4,
            "calibration": {"probabili": {"n": 40, "predicted_rate": 0.88,
                                          "realized_rate": 0.79}}}
    assert _refit_lines(summ) is None                    # below the floor
    summ["rounds"].append({"reconciled": True})
    lines = _refit_lines(summ)
    assert lines and "5" in lines[0]
    assert any("88%" in ln and "79%" in ln and "n=40" in ln for ln in lines)
    assert _refit_lines({"rounds": [{"reconciled": True}] * 9,
                         "calibration": {}}) is None     # no data, no push


# ── render_xi (shared by weekly push and bot /xi) ─────────────────────────


def _mini_adv():
    def row(nome, r, team="Inter", opp="Como", home=True, **kw):
        d = {"nome": nome, "R": r, "team": team, "opp": opp, "home": home,
             "p_play": kw.pop("p_play", 0.9)}
        d.update(kw)
        return d

    return {
        "round": 3, "module": "3-5-2", "total": 62.5, "modifier": 0.0,
        "generated_at": "2026-09-03T12:00:00+00:00",
        "feed_ages": {"probabili": 2.0, "indisponibili": 2.0},
        "xi": [row("Skorupski", "P"), row("Dimarco", "D"),
               row("Vlasic", "C"), row("Simeone", "A", diffidato=True)],
        "bench": [row("Theate", "D"), row("Lontani", "P")],
        "tribuna": [],
        "unavailable": [{"nome": "Zappacosta", "inj": "infortunato",
                         "why": "infortunato"}],
    }


def test_render_xi_both_formats_carry_the_board():
    from scripts.fantacalcio.tracker import render_xi

    msg, tg = render_xi(_mini_adv(), riv=None)
    for out in (msg, tg):
        assert "3-5-2" in out and "Giornata 3" in out.replace("giornata", "Giornata")
        assert "Skorupski" in out and "Theate" in out
        assert "Zappacosta" in out          # Out line
        assert "Simeone" in out             # diffidato listed
    # XI is role-ordered P→D→C→A in the body
    assert msg.index("Skorupski") < msg.index("Dimarco") < msg.index("Vlasic")
    # HTML only in the tg variant
    assert "<b>" in tg and "<b>" not in msg


def test_bot_xi_handler_serves_disk_advice(tmp_path, monkeypatch):
    import scripts.pipeline.telegram_bot as tb

    monkeypatch.setattr(tb, "PROJECT_ROOT", tmp_path)
    # No file on disk → graceful message, no crash
    assert "tracker" in tb._handle_xi()
    assert "rivali" in tb._handle_sfide()
    # With a real advice file → rendered board + age footer
    fdir = tmp_path / "data" / "fantacalcio"
    fdir.mkdir(parents=True)
    import json as _json
    (fdir / "xi_advice.json").write_text(_json.dumps(_mini_adv()))
    out = tb._handle_xi()
    assert "Skorupski" in out and "aggiornata" in out


# ── /formazioni + score_opponent_xi tool wiring ──────────────────────────


def test_formazioni_command_asks_for_the_screenshot(tmp_path, monkeypatch):
    import scripts.pipeline.telegram_bot as tb

    monkeypatch.setattr(tb, "PROJECT_ROOT", tmp_path)
    out = tb._handle_formazioni()
    assert "screenshot" in out and "avversari" in out


def test_opponent_xi_tool_is_registered_and_dispatches(monkeypatch):
    import json as _json

    import scripts.pipeline.telegram_bot as tb

    assert any(t["name"] == "score_opponent_xi" for t in tb._TG_TOOLS)
    assert "score_opponent_xi" in tb._TG_TOOL_HANDLERS
    # handler passes through to the advisor with all fields
    captured = {}

    def fake(team, players, module=None, bench_names=None):
        captured.update(team=team, players=players, module=module,
                        bench=bench_names)
        return {"p_win": 0.5}

    import scripts.fantacalcio.xi_advisor as xa
    monkeypatch.setattr(xa, "score_observed_xi", fake)
    out = tb._tool_score_opponent_xi({"team": "X", "players": ["A", "B"],
                                      "module": "3-4-3", "bench": ["C"]})
    assert _json.loads(out) == {"p_win": 0.5}
    assert captured == {"team": "X", "players": ["A", "B"],
                        "module": "3-4-3", "bench": ["C"]}


def test_reply_keyboard_has_formazioni_and_all_buttons_mapped():
    import scripts.pipeline.telegram_bot as tb

    kb = tb._reply_keyboard()
    labels = [b["text"] for row in kb["keyboard"] for b in row]
    assert "📸 Formazioni" in labels
    for lb in labels:
        assert tb._REPLY_BUTTON_MAP[lb].startswith("/")


# ── Monte Carlo H2H ──────────────────────────────────────────────────────


def _mc_side(names_exp, module="4-4-2", bench=None, team="AAA",
             sd=0.0, p_play=1.0, voto=6.0):
    xi = [{"nome": f"p{i}", "R": r, "team": team, "exp": e, "sd": sd,
           "exp_voto": voto, "voto_sd": 0.0, "p_play": p_play}
          for i, (r, e) in enumerate(names_exp)]
    return {"module": module, "xi": xi, "bench": bench or []}


def _flat_xi(total, module="3-4-3"):
    # 11 players summing to `total`, module WITHOUT modifier rights (3 D)
    roles = ["P"] + ["D"] * 3 + ["C"] * 4 + ["A"] * 3
    return _mc_side([(r, total / 11.0) for r in roles], module=module)


def test_mc_deterministic_thresholds_and_draw_band():
    from scripts.fantacalcio.xi_advisor import h2h_mc

    # 70 fp vs 60 fp, zero variance: 1 goal vs 0 — certain win
    r = h2h_mc(_flat_xi(70.0), _flat_xi(60.0), n=500, seed=1)
    assert r["p_win"] == 1.0 and r["p_draw"] == 0.0
    # 67 vs 66: both 1 goal — certain draw despite the fp gap
    r2 = h2h_mc(_flat_xi(67.0), _flat_xi(66.0), n=500, seed=1)
    assert r2["p_draw"] == 1.0
    assert r2["e_pts"] == 1.0


def test_mc_shared_club_cancels_variance():
    from scripts.fantacalcio.xi_advisor import h2h_mc

    def side(team):
        s = _flat_xi(66.0)
        for x in s["xi"]:
            x["team"] = team
            x["sd"] = 2.0
        return s

    same = h2h_mc(side("Inter"), side("Inter"), n=3000, seed=7)
    diff = h2h_mc(side("Inter"), side("Milan"), n=3000, seed=7)
    # identical rosters, same club: shared shock cancels a chunk of the
    # difference variance -> more draws than with independent clubs
    assert same["p_draw"] > diff["p_draw"]


def test_mc_bench_chain_substitutes_absent_starter():
    from scripts.fantacalcio.xi_advisor import h2h_mc

    a = _flat_xi(66.0)
    a["xi"][5]["p_play"] = 0.0                    # a C never plays
    bench = [{"nome": "sub", "R": "C", "team": "BBB", "exp": 6.0,
              "sd": 0.0, "exp_voto": 6.0, "voto_sd": 0.0, "p_play": 1.0}]
    a_no_bench = {**a, "bench": []}
    a_bench = {**a, "bench": bench}
    b = _flat_xi(63.0)
    r_nb = h2h_mc(a_no_bench, b, n=400, seed=3)
    r_wb = h2h_mc(a_bench, b, n=400, seed=3)
    # without the sub the side loses a player's 6 points (66-6=60 -> 0 goals,
    # loses to 63? no: 63 -> 0 goals too -> draw); with the sub back at 66 -> 1-0
    assert r_wb["p_win"] == 1.0
    assert r_nb["p_win"] == 0.0 and r_nb["p_draw"] == 1.0


def test_mc_modifier_applies_only_with_four_defenders():
    from scripts.fantacalcio.xi_advisor import h2h_mc

    def side(module, nd, voto):
        roles = ["P"] + ["D"] * nd + ["C"] * (10 - nd - 3) + ["A"] * 3
        s = _mc_side([(r, 6.0) for r in roles], module=module, voto=voto)
        return s

    # GK+top3 D at voto 7.0 -> tier +6; total 66 + 6 = 72 -> 2 goals
    with_mod = h2h_mc(side("4-3-3", 4, 7.0), _flat_xi(60.0), n=300, seed=5)
    # 3 D: no modifier rights -> 66 -> 1 goal
    no_mod = h2h_mc(side("3-4-3", 3, 7.0), _flat_xi(60.0), n=300, seed=5)
    assert with_mod["my_mu"] > no_mod["my_mu"] + 4


# ── market → Elo blend ───────────────────────────────────────────────────


def test_market_elo_blend_shifts_pair_and_preserves_mean(tmp_path, monkeypatch):
    import json as _json

    import scripts.fantacalcio.xi_advisor as xa

    up = tmp_path / "data" / "upcoming"
    up.mkdir(parents=True)
    # heavy home favourite: market says home much stronger than the table
    (up / "odds_full.json").write_text(_json.dumps({"matches": {
        "Alpha vs Beta": {"h2h": {"home": 1.30, "draw": 5.5, "away": 9.0}}}}))
    monkeypatch.setattr(xa, "ROOT", tmp_path)
    elo = {"Alpha": 1500.0, "Beta": 1500.0}
    fx = {"Alpha": {"opp": "Beta", "home": True},
          "Beta": {"opp": "Alpha", "home": False}}
    adj = xa._apply_market_elo(elo, fx)
    assert adj["Alpha"] > 1500.0 > adj["Beta"]
    assert round(adj["Alpha"] + adj["Beta"], 6) == 3000.0
    # w=0.7 of the market delta: table delta 0, so blend = 0.7 * d_mkt
    import math
    inv = [1 / 1.30, 1 / 5.5, 1 / 9.0]
    p = (inv[0] + 0.5 * inv[1]) / sum(inv)
    d_mkt = -400 * math.log10(1 / p - 1) - xa.HOME_ELO_EDGE
    assert abs((adj["Alpha"] - adj["Beta"]) - 0.7 * d_mkt) < 1e-6


def test_market_elo_blend_survives_missing_or_bad_odds(tmp_path, monkeypatch):
    import scripts.fantacalcio.xi_advisor as xa

    monkeypatch.setattr(xa, "ROOT", tmp_path)      # no odds file at all
    elo = {"Alpha": 1520.0, "Beta": 1480.0}
    fx = {"Alpha": {"opp": "Beta", "home": True},
          "Beta": {"opp": "Alpha", "home": False}}
    assert xa._apply_market_elo(elo, fx) == elo


# ── goal-ladder inference (verify_goal_ladder) ──────────────────────────


def _ladder_sched(tmp_path, fixtures):
    """One-competition schedule whose round 3 holds the given fixtures."""
    import json
    sched = {"competitions": {"Coppa": {"rounds": [
        {"league_round": 1, "sa_round": 3, "fixtures": fixtures}]}}}
    (tmp_path / "sched.json").write_text(json.dumps(sched))


def _fx(home, away, fph, fpa, score):
    return {"home": home, "away": away, "fp_home": fph, "fp_away": fpa,
            "score": score, "girone": None}


def _ladder_env(monkeypatch, tmp_path):
    import scripts.fantacalcio.pred_ledger as pl
    monkeypatch.setattr(pl, "LEDGER", tmp_path / "led.json")
    monkeypatch.setattr(pl, "SCHEDULE", tmp_path / "sched.json")
    return pl


def test_goal_ladder_verified_and_change_gated(monkeypatch, tmp_path):
    """66+6 world: away-side cells pin base 66 / step 6 exactly; the alert
    fires once (verified) and the identical re-check stays silent."""
    pl = _ladder_env(monkeypatch, tmp_path)
    _ladder_sched(tmp_path, [
        _fx("A", "B", 66.0, 65.5, "1-0"),
        _fx("C", "D", 72.0, 71.5, "2-1"),
        _fx("E", "F", 78.0, 66.0, "3-1"),
    ])
    msg = pl.verify_goal_ladder()
    assert msg and "verificata" in msg
    led = pl._load()
    assert led["goal_ladder"]["configured_ok"] is True
    assert led["goal_ladder"]["n_obs"] == 6
    assert pl.verify_goal_ladder() is None  # unchanged verdict = silence


def test_goal_ladder_refutes_wrong_step(monkeypatch, tmp_path):
    """TRUE POSITIVE: cells from a 66+4 league must refute the configured
    66+6 — and the feasible window must contain the real step 4."""
    pl = _ladder_env(monkeypatch, tmp_path)
    _ladder_sched(tmp_path, [
        _fx("A", "B", 70.0, 65.5, "2-0"),   # 66+6 says 1 goal at 70 fp
        _fx("C", "D", 74.0, 66.0, "3-1"),
    ])
    msg = pl.verify_goal_ladder()
    assert msg and "SMENTITA" in msg
    lad = pl._load()["goal_ladder"]
    assert lad["configured_ok"] is False
    assert lad["step_range"][0] <= 4.0 <= lad["step_range"][1]
    # standing mismatch, same evidence: no re-alert
    assert pl.verify_goal_ladder() is None
    # NEW contradicting cell -> n_obs grows -> alerts again
    _ladder_sched(tmp_path, [
        _fx("A", "B", 70.0, 65.5, "2-0"),
        _fx("C", "D", 74.0, 66.0, "3-1"),
        _fx("E", "F", 78.0, 65.0, "4-0"),
    ])
    assert pl.verify_goal_ladder() is not None


def test_goal_ladder_no_fit_names_the_importer(monkeypatch, tmp_path):
    """Contradictory cells (same fp, different goals) fit NO ladder — the
    alert must blame the cell mapping, not a rules change."""
    pl = _ladder_env(monkeypatch, tmp_path)
    _ladder_sched(tmp_path, [
        _fx("A", "B", 70.0, 70.0, "1-3"),
    ])
    msg = pl.verify_goal_ladder()
    assert msg and "import_rosters" in msg


def test_goal_ladder_skips_forfeit_and_unplayed(monkeypatch, tmp_path):
    """A sub-20 fp side (forfeit) and a score-less fixture yield no
    observations; with zero observations the check is silent."""
    pl = _ladder_env(monkeypatch, tmp_path)
    _ladder_sched(tmp_path, [
        _fx("A", "B", 0.0, 0.0, None),        # unplayed
        _fx("C", "D", 0.0, 71.0, "0-1"),      # home forfeited
    ])
    assert pl._ladder_observations() == [(71.0, 1, False)]
    _ladder_sched(tmp_path, [_fx("A", "B", 0.0, 0.0, None)])
    assert pl.verify_goal_ladder() is None


# ── porta inviolata (+1 fielded GK, zero gol subiti) ────────────────────


def test_parse_emits_gol_subiti_count():
    from scripts.fantacalcio.live_scores import parse
    df = parse(_row("gk", "p", "", "6,5", "4,5", (("Gol subiti", "2"),)))
    assert int(df.iloc[0].gs) == 2
    df0 = parse(_row("gk", "p", "", "7", "8"))
    assert int(df0.iloc[0].gs) == 0


def _cs_env(gk_gs):
    from scripts.fantacalcio.tracker import _pick
    roster = [
        {"id": 1, "nome": "Gk", "R": "P", "team": "Inter", "proj": 10.0},
        *[{"id": 10 + i, "nome": f"D{i}", "R": "D", "team": "Inter",
           "proj": 5.0} for i in range(4)],
        *[{"id": 20 + i, "nome": f"C{i}", "R": "C", "team": "Inter",
           "proj": 5.0} for i in range(4)],
        *[{"id": 30 + i, "nome": f"A{i}", "R": "A", "team": "Inter",
           "proj": 5.0} for i in range(2)],
    ]
    votes = {p["id"]: {"voto": 6.0, "fantavoto": 6.0, "bonus": 0.0,
                       "cards": 0.0, "gs": None} for p in roster}
    votes[1]["gs"] = gk_gs
    return _pick(roster, votes, [(6.0, 1)], [5.0, 4.5, 4.5], "proj")


def test_pick_credits_clean_sheet_gk():
    """TRUE POSITIVE pair: gs=0 earns exactly +1 over gs=2; a gs=None round
    (cached before the column existed) must behave like NO clean sheet."""
    total_cs = _cs_env(0)["total"]
    total_conceded = _cs_env(2)["total"]
    total_unknown = _cs_env(None)["total"]
    assert total_cs == total_conceded + 1.0
    assert total_unknown == total_conceded


def test_advise_total_carries_cs_expectation():
    import scripts.fantacalcio.xi_advisor as xa
    roster = [
        {"id": 1, "nome": "Gk", "R": "P", "team": "Inter", "level": 6.0,
         "voto": 6.0, "p_play": 1.0},
        *[{"id": 10 + i, "nome": f"D{i}", "R": "D", "team": "Inter",
           "level": 6.0, "voto": 6.0, "p_play": 1.0} for i in range(3)],
        *[{"id": 20 + i, "nome": f"C{i}", "R": "C", "team": "Inter",
           "level": 6.0, "voto": 6.0, "p_play": 1.0} for i in range(4)],
        *[{"id": 30 + i, "nome": f"A{i}", "R": "A", "team": "Inter",
           "level": 6.0, "voto": 6.0, "p_play": 1.0} for i in range(3)],
    ]
    fixtures = {"Inter": {"opp": "Nessuno FC", "home": 1, "ts": 0}}
    adv = xa.advise(roster, fixtures, {}, {})
    # unpriced fixture -> measured base rate, folded once into both totals
    assert adv["p_cs"] == round(xa.CS_BASE_RATE, 3)
    bare = sum(x["exp_slot"] for x in adv["xi"]) + adv["modifier"]
    assert abs(adv["total"] - (bare + xa.CS_BASE_RATE)) < 0.02


def test_mc_totals_shift_by_p_cs():
    import numpy as np

    import scripts.fantacalcio.xi_advisor as xa
    xi = [{"nome": "Gk", "R": "P", "team": "Inter", "exp": 6.0, "sd": 0.01,
           "exp_voto": 6.0, "voto_sd": 0.01, "p_play": 1.0}] +          [{"nome": f"X{i}", "R": r, "team": "Inter", "exp": 6.0, "sd": 0.01,
           "exp_voto": 6.0, "voto_sd": 0.01, "p_play": 1.0}
          for i, r in enumerate("DDDCCCCAAA")]
    rng = np.random.default_rng(7)
    zc = {"Inter": rng.standard_normal(4000)}
    base = xa._side_totals({"xi": xi, "bench": [], "module": "3-4-3",
                            "p_cs": 0.0}, zc, np.random.default_rng(1), 4000)
    with_cs = xa._side_totals({"xi": xi, "bench": [], "module": "3-4-3",
                               "p_cs": 0.5}, zc, np.random.default_rng(1), 4000)
    delta = float(with_cs.mean() - base.mean())
    assert 0.4 < delta < 0.6      # ~ +p_cs * 1.0 on the mean
