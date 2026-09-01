#!/usr/bin/env python3
"""Render data/fantacalcio/auction_board.json into a self-contained phone page.

Usage:
    python3 scripts/fantacalcio/build_auction_page.py [--out PATH]

The page embeds every player with room/model/walk-away from the board, the
last-season voti line, the foreign-league line (Understat) and a set of manual
caps + notes (OVERRIDES below) for the players the report discussed. All state
(what you paid, who took whom) lives in the browser's localStorage.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "data/fantacalcio/auction_board.json"
VOTI = ROOT / "data/fantacalcio/voti/stats_2025_26.parquet"
VOTI_DIR = ROOT / "data/fantacalcio/voti"
HIST_SEASONS = ["2023_24", "2024_25", "2025_26"]
UNDERSTAT = ROOT / "data/parsed/understat_players.parquet"
TEMPLATE = Path(__file__).with_name("auction_page_template.html")
DEFAULT_OUT = ROOT / "data/fantacalcio/asta_2026.html"

# nome -> (cap, hold, tier, note). tier 1 core / 2 plan / 3 fallback / 0 avoid
OVERRIDES: dict[str, tuple[int, int, int, str]] = {
    "Martinez L.": (124, 140, 1, "17g/6a, MFV 8.25. The anchor — 60% of the pot goes to 1st, concentrate here."),
    "Dimarco": (89, 140, 1, "7g/17a, MFV 7.64. Most important non-striker: bonus + modifier every week."),
    "Simeone": (28, 50, 1, "11g in 2273', MFV 7.09, Abate's #9. Best value in the whole attack."),
    "Vlasic": (27, 44, 1, "8g/3a, 37 apps, 5/5 pens. Steal at room."),
    "Esposito F.P.": (32, 51, 1, "7g/3a, MFV 7.02, and 7 npg off 9.9 npxG (unlucky). Take him and let Thuram (room 94, 1447') go."),
    "Orsolini": (60, 75, 3, "Declining: MFV 8.19 → 6.76, FV≥6.5 only 32% of rounds, 4/6 pens. Take only ≤ 60 — he replaces Frattesi + Schmid."),
    "Davis K.": (32, 41, 3, "Riser: MFV 6.38 → 7.37 in 30 apps, 4/4 pens — Udinese rigorista. Fine at ≤ 32."),
    "Da Cunha": (22, 28, 3, "4/4 pens, 35 apps two straight years, and the league's strongest finisher: FV 6.21 → 7.58 half-to-half. ≤ 28."),
    "Pellegrini Lo.": (10, 20, 3, "Riser: MFV 5.88 → 6.58. Room 10 for model 22 — cheap 6th mid."),
    "Dybala": (23, 38, 3, "2g off 5.5 xG — last season was finishing luck, not decline. Model 41 at room 23. Value if he falls to you."),
    "Zappacosta": (9, 29, 2, "35 apps, 3g/1a, 2631'. Modifier starter."),
    "Ekkelenkamp": (15, 26, 2, "5g/3a, 2602'."),
    "Conceicao": (21, 28, 2, "3g off 8.5 xG — unlucky, not blunt. FV sd 1.21 = metronome, 50% of rounds ≥6.5. Yıldız out till Nov 25 → starts."),
    "Svilar": (23, 25, 2, "38 apps, 3001'. Carnesecchi 18 / Butez 20 / Vicario 15 are the same tier."),
    "Frattesi": (19, 28, 2, "Lazio starter per sources; broken 2025-26 at Inter (14 apps, 0g)."),
    "Schmid": (11, 22, 2, "4g/8a in 3004' Bundesliga (Werder). Midfield sleeper."),
    "Adams A.": (18, 30, 2, "10g/3a in 2048' La Liga (Sevilla). 5th striker with upside."),
    "Bisseck": (12, 12, 2, "Akanji + Stones arrived: minutes at risk. Don't go past 12."),
    "Monterisi": (6, 9, 2, "2305'. Modifier body."),
    "Odgaard": (8, 10, 2, "Rotation, 1949'."),
    "Franjic": (2, 5, 2, "2005'."),
    "Douvikas": (35, 45, 3, "14g last year as the only #9. Kean (€40M) now splits the slot → ~1500'. Board's need 12.8 is a mirage — but Como's attack (4th, 71 pts in 25-26) keeps him relevant."),
    "Kean": (60, 70, 3, "xG flips the story: 13.9 npxG vs 6 scored — elite chances, broken finishing. Como finished 4th (71 pts) in 25-26 and open vs ranks 10-2-16-12-21. Real buy, starts over Douvikas."),
    "Ramos G.": (78, 90, 3, "PSG: 11/10/6 Ligue 1 goals in 1432/1047/1293'. €74M, undisputed Milan #9. Fallback if Lautaro >140."),
    "Malen": (95, 100, 3, "14g/2a in 18 games, MFV 8.97 — half-season sample. Let the table pay 122."),
    "Hojlund": (76, 82, 3, "12g/5a, MFV 7.44. Lukaku back ~Jan → trade him before Feb 25."),
    "Scamacca": (40, 44, 3, "10g in 1318', MFV 7.55. Splits with Krstović, injury record."),
    "Krstovic": (32, 40, 3, "Unluckiest finisher in the league: 10 npg off 15.7 npxG. MFV 7.19 despite it. Splits with Scamacca."),
    "Rowe": (13, 20, 3, "3g/3a at Bologna. Atalanta record signing, Sarri's LW vs Raspadori."),
    "Kessie": (20, 25, 3, "5g/3a in 26 Saudi games. Sarri mezzala, may take pens. FVM guessed."),
    "Carnesecchi": (18, 25, 3, "MV 6.36, best keeper rating. Equal to Svilar."),
    "Maignan": (16, 18, 3, ""),
    "Vicario": (15, 20, 3, "New Juve #1 (Di Gregorio sold, Perin gone). Board price 1 is the blind cap — real ~20. FVM guessed."),
    "Doekhi": (8, 16, 3, "5 goals in 3060' for Union Berlin — set-piece CB."),
    "Scalvini": (10, 18, 3, ""),
    "Tiago Gabriel": (7, 16, 3, "2209'."),
    "Bremer": (18, 19, 3, "4g/3a, MFV 6.81."),
    "Pavlovic": (17, 17, 3, "5 goals."),
    "Beto": (15, 20, 3, "8g and 9g last two Everton seasons. 50/50 with Pellegrino → halve. FVM guessed."),
    "Pellegrino M.": (12, 15, 3, "9g in 37 at Parma, 5 yellows. 50/50 with Beto."),
    "Woltemade": (20, 30, 3, "12g/2a Stuttgart, 8g/3a Newcastle. Kolo Muani starts first. FVM guessed."),
    "Castro S.": (8, 12, 3, "Malen's backup at Roma, and faded hard: FV 7.21 → 5.86 in H2 at Bologna."),
    "Paz N.": (50, 60, 3, "Luckiest finisher in the league: 12 goals off 7.4 npxG (+4.6). Elite minutes and rating, but room 84 pays for a repeat xG says won't come."),
    "Piccoli": (11, 25, 3, "Model 36 at room 11, 4g off 6.8 xG (unlucky), +0.92 FV in H2. Cheap 4th striker."),
    "Adams C.": (12, 22, 3, "6g/3a, FV 6.64, Torino second striker. Model 30 at room 12."),
    "Bernardeschi": (8, 18, 3, "4g/2a, MFV 6.56. Model 30 at room 8 — bench value at Bologna."),
    "Berardi": (11, 25, 3, "MFV 7.19 in 26 apps, 46% of rounds ≥6.5, 2/3 pens — but Sassuolo + injury history. ≤ 25."),
    "Butez": (18, 22, 3, "38 apps, 29 conceded at Como."),
    "Zielinski": (15, 20, 3, "FV ≥ 6.5 in 57% of rounds — elite consistency. But 5 goals off 1.3 npxG: don't price repeats. Fine at room, no more."),
    "Milinkovic-Savic V.": (4, 12, 3, "Napoli minutes lean his way (Meret 1060' projected). Model 19 at room 4 — best cheap keeper."),
    "Yildiz": (0, 20, 0, "Injured until ~Nov 25 — you'd pay ~65 for half a season. Don't."),
    "Calhanoglu": (0, 45, 0, "5 npg off 1.3 npxG — the non-pen goals were luck. 1726' projected. Room 77 pays a mirage."),
    "Pulisic": (0, 35, 0, "Hardest fader in the league: FV 8.67 → 5.80 half-to-half, sd 3.0, 1014' projected."),
    "De Ketelaere": (30, 36, 3, "Heaviest assist under-performance: 5a off 10.7 xA, 64 key passes. Creation is elite; Atalanta rotation is the only risk."),
    "Mancini": (12, 18, 3, "+1.03 FV in H2; Roma take 30% of their shots from set pieces — header threat from corners."),
    "Gnonto": (5, 8, 0, "0g/1a at Leeds. Fourth option at Fiorentina."),
    "Mora": (8, 12, 0, "18yo, hype price (room 35)."),
    "Mastantuono": (8, 10, 0, "1g in 1026' for Real Madrid at 18. Give him time."),
    "Moreira": (6, 10, 0, "4g/7a + 2g/7a Ligue 1. Only if he starts."),
    "Kolo Muani": (0, 15, 0, "1g at Tottenham last year, shares the slot with Woltemade. Don't."),
    "Thuram": (0, 60, 0, "Room 94 for 1447'. Let it go."),
    "McTominay": (0, 80, 0, "Injured till ~Oct 1: 5 rounds lost."),
    "Wesley": (0, 17, 0, "Room 28 > model 17."),
    "Akanji": (0, 7, 0, "MFV 6.41, no bonus."),
    "Molina N.": (0, 6, 0, "732' projected."),
    "Theate": (0, 5, 0, "1 goal in two Bundesliga seasons."),
    "Balerdi": (0, 4, 0, "0 goals in two seasons."),
    "Mbangula": (0, 3, 0, "3g/2a at Werder."),
    "Ngonge": (0, 3, 0, ""),
    "Hutchinson": (0, 5, 0, "1g/5a Forest."),
    "Sarr P.M.": (0, 3, 0, "2g/4a Tottenham."),
}

_TR = str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ı": "i", "ć": "c", "č": "c", "š": "s", "ž": "z"})


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).translate(_TR)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()


def _rig(v) -> tuple[int, int]:
    """Parse 'rig' which is '3 / 4' (scored/taken) in recent seasons, a number in old ones."""
    s = str(v)
    if "/" in s:
        a, b = s.split("/")
        try:
            return int(a), int(b)
        except ValueError:
            return 0, 0
    try:
        return int(float(s)), int(float(s))
    except (ValueError, TypeError):
        return 0, 0


def _season_line(r, label: str) -> str:
    if not r.pg:
        return ""
    if r.R == "P":
        line = f"{label}: {int(r.pg)} apps · MV {r.mv:.2f} · {int(r.gs)} conceded"
        rp = _rig(getattr(r, "rp", 0))[0]
        return line + (f" · {rp} pens saved" if rp else "")
    sc, tk = _rig(r.rig)
    line = f"{label}: {int(r.pg)} apps · MV {r.mv:.2f} · MFV {r.mfv:.2f} · {int(r.gol)}g {int(r.ass)}a"
    if tk:
        line += f" · pens {sc}/{tk}"
    if r.amm >= 6:
        line += f" · {int(r.amm)} yc"
    return line


def _history(hist: dict[str, pd.DataFrame], nome: str) -> list[str]:
    out = []
    for season, df in hist.items():
        row = df[df.nome == nome]
        if not row.empty:
            label = season[2:4] + "-" + season[5:7]
            line = _season_line(row.iloc[0], label)
            if line:
                out.append(line + f" ({row.iloc[0].team})")
    return out


def _round_consistency() -> pd.DataFrame:
    """Per-slug 2025-26 round profile: apps, mean FV, % rounds ≥6.5, bonus rounds."""
    frames = [pd.read_parquet(f) for f in sorted(VOTI_DIR.glob("round_2025_26_*.parquet"))]
    r = pd.concat(frames, ignore_index=True)
    r = r[r.played.astype(bool) & r.fantavoto.notna()]
    g = r.groupby("slug").agg(apps=("fantavoto", "size"), fv=("fantavoto", "mean"),
                              ge65=("fantavoto", lambda s: (s >= 6.5).mean()),
                              bonus=("bonus", lambda s: (s > 0).sum()))
    return g.reset_index()


def _cons_line(cons: pd.DataFrame, nome: str) -> str:
    raw = str(nome).split(" ")
    initials = "".join(t.rstrip(".") for t in raw if t.endswith("."))
    surname = _norm(" ".join(t for t in raw if not t.endswith(".")))
    parts = surname.split(" ")

    def _hit(slug: str) -> bool:
        toks = slug.split("-")
        if not all(p in toks for p in parts):
            return False
        others = [t for t in toks if t not in parts]
        return not initials or any(t.startswith(initials[0].lower()) for t in others) or not others

    m = cons[cons.slug.map(_hit)]
    if len(m) != 1:
        return ""
    r = m.iloc[0]
    return f"25-26 rounds: {int(r.apps)} played · FV ≥ 6.5 in {r.ge65:.0%} · bonus in {int(r.bonus)}"


def _last_season_line(voti: pd.DataFrame, nome: str) -> str:
    hit = voti[voti.nome == nome]
    if hit.empty:
        return ""
    r = hit.iloc[0]
    if not r.pg:
        return ""
    return f"25-26: {int(r.pg)} apps · MV {r.mv:.2f} · MFV {r.mfv:.2f} · {int(r.gol)}g {int(r.ass)}a · {int(r.amm)} yc"


def _foreign_line(us: pd.DataFrame, nome: str, league: str | None) -> str:
    raw = str(nome).split(" ")
    initials = "".join(t.rstrip(".") for t in raw if t.endswith("."))  # "Adams A." -> "A", "Sarr P.M." -> "PM"
    surname = _norm(" ".join(t for t in raw if not t.endswith(".")))    # "Kolo Muani", "Beto", "Ramos"
    if len(surname) < 3:
        return ""
    cand = us[(us.season >= "2023-2024") & (us.season <= "2025-2026") & (us.minutes > 300) & (us.league != "ITA-Serie A")]

    def _hit(n: str) -> bool:
        if n == surname:
            return True
        if not n.endswith(" " + surname):
            return False
        return not initials or n.startswith(initials[0].lower())

    cand = cand[cand.n.map(_hit)]
    names = cand.player.unique()
    if len(names) != 1:
        return ""
    cand = cand.sort_values("season").tail(2)
    parts = [f"{r.season[2:4]}-{r.season[7:9]} {r.team}: {int(r.goals)}g {int(r.assists)}a in {int(r.minutes)}'"
             for r in cand.itertuples()]
    return cand.iloc[-1].league.split("-")[-1] + " · " + " | ".join(parts)


def build(out: Path, fragment_out: Path | None = None) -> Path:
    board = json.loads(BOARD.read_text())
    voti = pd.read_parquet(VOTI)
    us = pd.read_parquet(UNDERSTAT)
    us["n"] = us.player.map(_norm)
    hist_frames = {s: pd.read_parquet(VOTI_DIR / f"stats_{s}.parquet") for s in HIST_SEASONS}
    cons = _round_consistency()
    squad_ids = {s["id"] for s in board["squad"]}
    by_id = {p["id"]: p for p in board["players"]}
    squad_by_id = {s["id"]: s for s in board["squad"]}  # need_loss / walkaway / alternatives live here
    alts_by_id = {s["id"]: s.get("alternatives") or [] for s in board["squad"]}
    name_to_id = {p["nome"]: p["id"] for p in board["players"]}

    rows = []
    for p in board["players"]:
        if p["status"] == "DEPARTED":
            continue
        sq = squad_by_id.get(p["id"], {})
        room = round(p.get("market_price") or 0)
        model = round(p.get("model_price") or 0)
        walk = sq.get("walkaway")
        need = sq.get("need_loss") or 0
        ov = OVERRIDES.get(p["nome"])
        if ov:
            cap, hold, tier, note = ov
        else:
            cap = max(1, min(room, model))
            hold = max(cap, round(walk) if walk else model)
            tier = 2 if p["id"] in squad_ids else 4
            note = ""
            if p.get("blind"):
                note = "Blind: no recent record, priced at replacement."
        alts = []
        for a in alts_by_id.get(p["id"], []):
            if isinstance(a, dict):
                aid = a.get("id") if a.get("id") in by_id else name_to_id.get(a.get("nome"))
            else:
                aid = name_to_id.get(a)
            if aid is not None and aid not in alts:
                alts.append(aid)
        last = _last_season_line(voti, p["nome"])
        hist = _history(hist_frames, p["nome"])
        # foreign record only for players the board flags foreign or who have no Serie A line
        # (otherwise a namesake abroad — Oscar Højlund for Rasmus — gets attached)
        fo = _foreign_line(us, p["nome"], p.get("fo_league")) if (p.get("foreign") or not last) else ""
        rows.append({
            "id": p["id"], "n": p["nome"], "r": p["R"], "t": p["team"],
            "room": room, "model": model, "cap": cap, "hold": hold,
            "need": round(need, 1), "min": round(p.get("proj_min") or 0),
            "tier": tier, "note": note, "inj": (p.get("inj_note") or "")[:70],
            "st": p["status"], "est": bool(p.get("estimated")),
            "hist": hist,
            "cons": _cons_line(cons, p["nome"]),
            "fo": fo,
            "alts": alts,
        })
    data = {
        "generated": board["generated_at"], "budget": board["settings"]["budget"],
        "roster": board["settings"]["roster"], "players": rows,
    }
    fragment = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    full = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<meta name="apple-mobile-web-app-capable" content="yes">'
            f'</head><body>\n{fragment}\n</body></html>\n')
    out.write_text(full)
    if fragment_out:
        fragment_out.write_text(fragment)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fragment", type=Path, default=None, help="also write the body-only version (for the artifact)")
    a = ap.parse_args()
    print(build(a.out, a.fragment), len(json.loads(BOARD.read_text())["players"]), "players")
