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
