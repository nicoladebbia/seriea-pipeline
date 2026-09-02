"""What the papers think of each Serie A club — scored daily, and it LEARNS.

Two layers, deliberately cheap and local (zero paid API — the May 2026 Groq
sentiment lesson: a per-headline LLM loop burned $73/mo for a signal nothing
used; this is the free replacement, display-only):

  1. LEXICON pulse: every headline mentioning a club is scored with a curated
     Italian football lexicon (word-boundary stems). Per-club pulse is an
     exponentially-decayed sum (half-life PULSE_HALFLIFE_D days), so "what
     people think" tracks the news cycle, not the season average.
  2. LEARNING layer: every scored headline is parked as pending on the club's
     NEXT Serie A round; when that round's result lands in the fixtures file,
     the headline's words train a tiny Naive Bayes (win vs loss, draws
     dropped). Each settled giornata makes the word weights more Italian-
     football-shaped than any hand lexicon. NB influence is blended in by
     evidence volume: weight n_labeled/(n_labeled+NB_PRIOR_N), so it starts
     silent and earns its voice.

State lives in team_pulse.json and is updated incrementally on every tracker
run — nothing is recomputed from scratch, which is what "learns each time"
means here.
"""
from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data" / "fantacalcio" / "team_pulse.json"

# Club match tokens: official name + the short forms the papers actually use.
TEAM_TOKENS = {
    "Atalanta": ["Atalanta"], "Bologna": ["Bologna"], "Cagliari": ["Cagliari"],
    "Como": ["Como"], "Cremonese": ["Cremonese"], "Fiorentina": ["Fiorentina", "Viola"],
    "Frosinone": ["Frosinone"], "Genoa": ["Genoa"], "Inter": ["Inter"],
    "Juventus": ["Juventus", "Juve"], "Lazio": ["Lazio"], "Lecce": ["Lecce"],
    "Milan": ["Milan"], "Napoli": ["Napoli"], "Parma": ["Parma"],
    "Roma": ["Roma", "giallorossi"], "Sassuolo": ["Sassuolo"],
    "Torino": ["Torino", "Toro"], "Udinese": ["Udinese"],
    "Venezia": ["Venezia"], "Verona": ["Verona", "Hellas"],
}
# Stems, word-boundary-anchored on the left. Weights are a PRIOR the NB layer
# gradually overrides — do not hand-tune past sign and rough size.
LEXICON = {
    r"vittori|vince|vinc|batte|trionf|show|espugna": 2,
    r"rinnov|recuper|torna|convocat|doppietta|tripletta|capolista|colpo|esalt|elogi|super|perla|gioiell|decisiv": 1,
    r"crisi|esoner|contestazion|fischi|umiliazion|tracoll|disfatta": -2,
    r"infortun|lesion|ko\b|salta|squalific|sconfitt|perde|flop|polemic|tegola|emergenz|rischi|dubbi|addio|rottur|delus|stop\b|forfait": -1,
}
_LEX = [(re.compile(r"\b(?:" + pat + r")", re.I), w) for pat, w in LEXICON.items()]
_TEAM_RX = {t: re.compile(r"\b(?:" + "|".join(map(re.escape, toks)) + r")\b", re.I)
            for t, toks in TEAM_TOKENS.items()}
_WORD_RX = re.compile(r"[a-zà-ù]{4,}", re.I)

PULSE_HALFLIFE_D = 7.0
NB_PRIOR_N = 50.0
MAX_PENDING = 400


def lex_score(text: str) -> int:
    return max(-3, min(3, sum(w * len(rx.findall(text)) for rx, w in _LEX)))


def _words(text: str) -> list[str]:
    return sorted({w.lower() for w in _WORD_RX.findall(text)})[:40]


def _load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"teams": {}, "nb": {"pos": {}, "neg": {}, "n_pos": 0, "n_neg": 0},
                "pending": [], "seen": []}


def _nb_logodds(nb: dict, words: list[str]) -> float:
    n_pos, n_neg = nb["n_pos"], nb["n_neg"]
    if n_pos < 3 or n_neg < 3:
        return 0.0
    v = len(set(nb["pos"]) | set(nb["neg"])) or 1
    tp = sum(nb["pos"].values()) + v
    tn = sum(nb["neg"].values()) + v
    lo = 0.0
    for w in words:
        lo += math.log((nb["pos"].get(w, 0) + 1) / tp) \
            - math.log((nb["neg"].get(w, 0) + 1) / tn)
    return max(-3.0, min(3.0, lo))


def _results_by_round() -> dict[tuple[str, int], int]:
    """(club, sa_round) -> +1 win / -1 loss / 0 draw, from the fixtures file."""
    from config.team_names import normalize_team as NT
    from scripts.utils.match_timing import _sofascore_fixture_files
    out: dict[tuple[str, int], int] = {}
    try:
        path = next(p for p, lg in _sofascore_fixture_files() if lg == "serie_a")
        raw = json.loads(path.read_text())
    except (OSError, ValueError, StopIteration):
        return out
    for x in raw:
        if (x.get("status") or {}).get("type") != "finished":
            continue
        rnd = (x.get("roundInfo") or {}).get("round")
        hs = (x.get("homeScore") or {}).get("current")
        as_ = (x.get("awayScore") or {}).get("current")
        h = NT((x.get("homeTeam") or {}).get("name", "")) or ""
        a = NT((x.get("awayTeam") or {}).get("name", "")) or ""
        if rnd is None or hs is None or as_ is None or not h or not a:
            continue
        sgn = 0 if hs == as_ else (1 if hs > as_ else -1)
        out[(h, int(rnd))] = sgn
        out[(a, int(rnd))] = -sgn
    return out


def update(items: list[dict], next_round: int | None) -> dict:
    """Fold today's headlines in, label anything whose round has settled."""
    st = _load()
    now = datetime.now(UTC)
    seen = set(st.get("seen", []))

    # 1) decay every club's pulse by elapsed time
    for t in st["teams"].values():
        try:
            dt_d = (now - datetime.fromisoformat(t["updated_at"])).total_seconds() / 86400
        except (KeyError, ValueError):
            dt_d = 0.0
        t["pulse"] = t.get("pulse", 0.0) * 0.5 ** (dt_d / PULSE_HALFLIFE_D)
        t["updated_at"] = now.isoformat()

    # 2) new headlines -> lexicon score + park for labeling
    for it in items:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        text = f"{it.get('title', '')} {it.get('desc', '')}"
        clubs = [c for c, rx in _TEAM_RX.items() if rx.search(text)]
        if not clubs:
            continue
        sc = lex_score(text)
        ws = _words(text)
        for c in clubs:
            t = st["teams"].setdefault(
                c, {"pulse": 0.0, "n_items": 0, "updated_at": now.isoformat(),
                    "recent": []})
            t["pulse"] += sc + _nb_logodds(st["nb"], ws) \
                * (st["nb"]["n_pos"] + st["nb"]["n_neg"]) \
                / (st["nb"]["n_pos"] + st["nb"]["n_neg"] + NB_PRIOR_N)
            t["n_items"] += 1
            t["recent"] = ([{"title": it.get("title", "")[:110],
                             "score": sc, "source": it.get("source")}]
                           + t.get("recent", []))[:3]
            if next_round is not None:
                st["pending"].append({"club": c, "sa_round": int(next_round),
                                      "words": ws})

    # 3) learning: label pending items whose round has a result
    res = _results_by_round()
    still = []
    labeled = 0
    for pnd in st["pending"][-MAX_PENDING:]:
        key = (pnd["club"], pnd["sa_round"])
        if key not in res:
            still.append(pnd)
            continue
        sgn = res[key]
        if sgn == 0:
            labeled += 1
            continue                      # draws teach nothing directional
        bucket = st["nb"]["pos"] if sgn > 0 else st["nb"]["neg"]
        for w in pnd["words"]:
            bucket[w] = bucket.get(w, 0) + 1
        st["nb"]["n_pos" if sgn > 0 else "n_neg"] += 1
        labeled += 1
    st["pending"] = still
    st["seen"] = sorted(seen)[-2000:]
    st["generated_at"] = now.isoformat()
    st["labeled_this_run"] = labeled
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False))
    return st


def refresh(next_round: int | None = None) -> dict:
    """Fetch the same feeds news.py uses and fold them in. Best-effort."""
    from curl_cffi import requests as rq

    from scripts.fantacalcio.news import FEEDS, _parse_feed
    items = []
    for source, url in FEEDS.items():
        try:
            r = rq.get(url, impersonate="chrome124", timeout=25)
            if r.status_code == 200:
                items.extend(_parse_feed(r.text, source))
        except Exception:
            continue
    return update(items, next_round)


def summary() -> dict:
    st = _load()
    teams = [{"team": c, "pulse": round(t.get("pulse", 0.0), 2),
              "n_items": t.get("n_items", 0), "recent": t.get("recent", [])}
             for c, t in st.get("teams", {}).items()]
    teams.sort(key=lambda x: -x["pulse"])
    nb = st.get("nb", {})
    return {"generated_at": st.get("generated_at"), "teams": teams,
            "nb_trained_on": {"wins": nb.get("n_pos", 0),
                              "losses": nb.get("n_neg", 0)},
            "pending_labels": len(st.get("pending", []))}


if __name__ == "__main__":
    st = refresh()
    s = summary()
    print(f"labeled {st.get('labeled_this_run', 0)}, pending {s['pending_labels']}, "
          f"nb {s['nb_trained_on']}")
    for t in s["teams"][:8]:
        print(f"  {t['team']:12s} pulse {t['pulse']:+5.2f} (n={t['n_items']})")
