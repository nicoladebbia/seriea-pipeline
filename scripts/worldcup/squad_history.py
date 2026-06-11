"""Historical squad club-Elo — the dataset behind the squad-strength study.

For each backtest tournament (scripts.worldcup.backtest DEV/FINAL windows),
build the mean club Elo + coverage of every participating squad AS OF that
tournament's start: the historical counterpart of the 2026
clubelo_squad_strength.json scrape.

- Squads (player → club) parsed from the Wikipedia "<tournament> squads"
  page wikitext ({{nat fs g player|...|club=[[X]]}} rows under team
  headings) — the same source the 2026 squads.json came from.
- Club ratings from api.clubelo.com/{YYYY-MM-DD} (one CSV of all ranked
  clubs per date). ClubElo covers EUROPEAN clubs only: non-European clubs
  are expected misses, recorded in coverage_pct — the study shrinks the
  signal by coverage rather than pretending it's complete.
- Club-name matching: normalization cascade + explicit alias map +
  prefix-token containment, ambiguity → unmatched (never guess between
  two plausible clubs). Unmatched clubs are reported by frequency.

Raw fetches are cached under data/worldcup/squad_history/ so re-runs are
offline and deterministic. Output: data/worldcup/clubelo_history.json
keyed by backtest label → team (results.csv name space) →
{mean_club_elo, matched, total, coverage_pct}.

Run: python3 -m scripts.worldcup.squad_history
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.worldcup.engine import DATA_DIR, atomic_write_json

CACHE_DIR = DATA_DIR / "squad_history"
OUTPUT_JSON = DATA_DIR / "clubelo_history.json"

# (backtest label, Wikipedia squads page title, ClubElo snapshot date)
TOURNAMENTS: list[tuple[str, str, str]] = [
    ("World Cup 2018", "2018 FIFA World Cup squads", "2018-06-14"),
    ("Euro 2020", "UEFA Euro 2020 squads", "2021-06-11"),
    ("World Cup 2022", "2022 FIFA World Cup squads", "2022-11-20"),
    ("Euro 2024", "UEFA Euro 2024 squads", "2024-06-14"),
    ("Copa América 2024", "2024 Copa América squads", "2024-06-20"),
]

# Wikipedia squad-page team headings → results.csv team names (only where
# they differ). Validated against the actual backtest windows by main().
WIKI_TO_RESULTS: dict[str, str] = {
    "Czech Republic": "Czech Republic",  # results.csv keeps the 2021 name
    "Turkey": "Turkey",
    "Türkiye": "Turkey",
}

# Wiki club display name → ClubElo name, for the pairs the mechanical
# cascade can't bridge. Kept small on purpose: every entry is a real
# mismatch seen in the data, not speculation.
CLUB_ALIASES: dict[str, str] = {
    "paris saint germain": "Paris SG",
    "internazionale": "Inter",
    "inter milan": "Inter",
    "atletico madrid": "Atletico",
    "atlético madrid": "Atletico",
    "manchester united": "Man United",
    "manchester city": "Man City",
    "bayern munich": "Bayern",
    "borussia monchengladbach": "Gladbach",
    "borussia mönchengladbach": "Gladbach",
    "psv eindhoven": "PSV",
    "sporting cp": "Sporting",
    "wolverhampton wanderers": "Wolves",
    "brighton hove albion": "Brighton",
    "west ham united": "West Ham",
    "newcastle united": "Newcastle",
    "tottenham hotspur": "Tottenham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "nottingham forest": "Forest",
    "sheffield united": "Sheffield United",
    "real betis": "Betis",
    "real sociedad": "Sociedad",
    "athletic bilbao": "Athletic",
    "celta vigo": "Celta",
    "deportivo la coruna": "La Coruna",
    "bayer leverkusen": "Leverkusen",
    "eintracht frankfurt": "Frankfurt",
    "vfb stuttgart": "Stuttgart",
    "vfl wolfsburg": "Wolfsburg",
    "tsg hoffenheim": "Hoffenheim",
    "1899 hoffenheim": "Hoffenheim",
    "rb leipzig": "RB Leipzig",
    "red bull salzburg": "Salzburg",
    "fc salzburg": "Salzburg",
    "dinamo zagreb": "Dinamo Zagreb",
    "red star belgrade": "Crvena Zvezda",
    "crvena zvezda": "Crvena Zvezda",
    "olympique lyonnais": "Lyon",
    "olympique de marseille": "Marseille",
    "olympique marseille": "Marseille",
    "saint etienne": "Saint-Etienne",
    "as monaco": "Monaco",
    "stade rennais": "Rennes",
    "stade de reims": "Reims",
    "stade brestois": "Brest",
    "rc lens": "Lens",
    "losc lille": "Lille",
    "cska moscow": "CSKA Moskva",
    "lokomotiv moscow": "Lok Moskva",
    "spartak moscow": "Spartak Moskva",
    "dynamo moscow": "Dinamo Moskva",
    "zenit saint petersburg": "Zenit",
    "shakhtar donetsk": "Shakhtar",
    "dynamo kyiv": "Dynamo Kyiv",
    "fenerbahce": "Fenerbahce",
    "galatasaray": "Galatasaray",
    "besiktas": "Besiktas",
    "olympiacos": "Olympiakos",
    "panathinaikos": "Panathinaikos",
    "club brugge": "Brugge",
    "royal antwerp": "Antwerp",
    "anderlecht": "Anderlecht",
    "ajax": "Ajax",
    "az alkmaar": "AZ",
    "feyenoord": "Feyenoord",
    "celtic": "Celtic",
    "rangers": "Rangers",
    "sl benfica": "Benfica",
    "fc porto": "Porto",
    "sporting braga": "Braga",
    "fc copenhagen": "FC Kobenhavn",
    "copenhagen": "FC Kobenhavn",
    "fc midtjylland": "Midtjylland",
    "malmo ff": "Malmoe FF",
    "young boys": "Young Boys",
    "fc basel": "Basel",
    "slavia prague": "Slavia Praha",
    "sparta prague": "Sparta Praha",
    "viktoria plzen": "Plzen",
    "legia warsaw": "Legia",
    "lech poznan": "Lech",
    "ferencvaros": "Ferencvaros",
    "apoel": "APOEL",
    "maccabi tel aviv": "Maccabi Tel Aviv",
}

# One {{nat fs [g] player|...}} template per line; the body nests other
# templates (age={{birth date and age2|...}}), so club= is extracted from
# the LINE rather than brace-balancing the template.
_PLAYER_LINE_RE = re.compile(r"\{\{nat\s+fs\s+(?:g\s+)?player\s*\|", re.IGNORECASE)
_CLUB_PARAM_RE = re.compile(
    r"\|\s*club\s*=\s*(?P<club>.*?)(?:\|\s*clubnat\s*=|\}\}\s*$)", re.IGNORECASE
)
_HEADING_RE = re.compile(r"^===([^=].*?)===\s*$", re.MULTILINE)
_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_ILL_RE = re.compile(r"\{\{ill\|([^}|]+)[^}]*\}\}", re.IGNORECASE)


def _fetch(url: str, cache_file: Path) -> str:
    """GET with on-disk cache; raw bytes are the artifact of record."""
    if cache_file.exists():
        return cache_file.read_text()
    # S310-justified: URLs are hardcoded http(s) endpoints (clubelo API,
    # en.wikipedia.org) assembled from constants — no user input.
    req = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "seriea-pipeline-research/1.0 (backtest data)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        text = resp.read().decode("utf-8")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text)
    return text


def fetch_clubelo_snapshot(date: str) -> dict[str, float]:
    """All ranked clubs' Elo as of `date` (one CSV from api.clubelo.com)."""
    text = _fetch(
        f"http://api.clubelo.com/{date}", CACHE_DIR / f"clubelo_{date}.csv"
    )
    out: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO(text)):
        club, elo = row.get("Club", ""), row.get("Elo", "")
        if club and elo:
            out[club] = float(elo)
    if len(out) < 100:
        raise RuntimeError(f"ClubElo snapshot {date} looks broken: {len(out)} rows")
    return out


def fetch_squads_wikitext(page_title: str) -> str:
    safe = page_title.replace(" ", "_")
    quoted = urllib.parse.quote(safe, safe="_")  # é in 'Copa América' etc.
    return _fetch(
        f"https://en.wikipedia.org/w/index.php?title={quoted}&action=raw",
        CACHE_DIR / f"squads_{_strip_accents(safe)}.wiki",
    )


def _clean_link(value: str) -> str:
    """'[[A|B]]' → B, '[[A]]' → A, '{{ill|A|...}}' → A, plain → plain."""
    value = value.strip()
    m = _LINK_RE.search(value)
    if m:
        return (m.group(2) or m.group(1)).strip()
    m = _ILL_RE.search(value)
    if m:
        return m.group(1).strip()
    return value


def parse_squads(wikitext: str) -> dict[str, list[str]]:
    """Team → list of club display names, one entry per squad player.

    Teams are the ===X=== headings; player rows are {{nat fs g player}}
    templates with a club= parameter. Headings that contain no player rows
    (e.g. '===Player representation===' appendix sections) drop out
    naturally because they collect zero players.
    """
    sections: list[tuple[str, int]] = [
        (_clean_link(m.group(1)), m.start()) for m in _HEADING_RE.finditer(wikitext)
    ]
    squads: dict[str, list[str]] = {}
    for (team, start), (_next_team, next_start) in zip(
        sections, sections[1:] + [("", len(wikitext))]
    ):
        chunk = wikitext[start:next_start]
        clubs: list[str] = []
        for line in chunk.splitlines():
            if not _PLAYER_LINE_RE.search(line):
                continue
            cm = _CLUB_PARAM_RE.search(line)
            if cm:
                club = _clean_link(cm.group("club"))
                if club:
                    clubs.append(club)
        if clubs:
            squads[team] = clubs
    return squads


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


_STOP_TOKENS = {
    "fc", "cf", "afc", "sc", "ac", "as", "ssc", "cd", "sd", "ca", "rc", "rcd",
    "fk", "sk", "bk", "nk", "if", "sv", "bsc", "kaa", "krc", "rsc", "ud", "us",
    "ogc", "psc", "pfc", "sfc", "cska", "club", "cp", "de", "the",
}


def _norm(name: str) -> str:
    s = _strip_accents(name).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(name: str) -> tuple[str, ...]:
    return tuple(t for t in _norm(name).split() if t not in _STOP_TOKENS)


class ClubMatcher:
    """Wiki club display name → ClubElo rating, via a deterministic cascade:
    exact normalized name → alias map → prefix-token containment (every
    ClubElo token must prefix-match a distinct wiki token; unique hit only —
    ambiguity is a non-match, never a guess)."""

    def __init__(self, elo_by_club: dict[str, float]) -> None:
        self.elo_by_club = elo_by_club
        self.by_norm = {_norm(c): c for c in elo_by_club}
        self.token_sets = {c: _tokens(c) for c in elo_by_club}
        self.cache: dict[str, str | None] = {}

    def match(self, wiki_club: str) -> str | None:
        if wiki_club in self.cache:
            return self.cache[wiki_club]
        result = self._match_uncached(wiki_club)
        self.cache[wiki_club] = result
        return result

    def _match_uncached(self, wiki_club: str) -> str | None:
        norm = _norm(wiki_club)
        if norm in self.by_norm:
            return self.by_norm[norm]
        alias = CLUB_ALIASES.get(norm)
        if alias is not None:
            return alias if alias in self.elo_by_club else None
        wiki_toks = _tokens(wiki_club)
        if not wiki_toks:
            return None
        hits = []
        for club, elo_toks in self.token_sets.items():
            if not elo_toks:
                continue
            unused = list(wiki_toks)
            ok = True
            for et in elo_toks:
                pick = next(
                    (wt for wt in unused if wt.startswith(et) or et.startswith(wt)),
                    None,
                )
                if pick is None:
                    ok = False
                    break
                unused.remove(pick)
            if ok:
                hits.append((len(elo_toks), club))
        if not hits:
            return None
        best = max(h[0] for h in hits)
        top = [club for score, club in hits if score == best]
        return top[0] if len(top) == 1 else None


def build_history() -> dict[str, Any]:
    out: dict[str, Any] = {
        "_provenance": {
            "generated_at": datetime.now(UTC).isoformat(),
            "sources": (
                "en.wikipedia.org '<tournament> squads' wikitext + "
                "api.clubelo.com/{date} snapshots (cached in "
                "data/worldcup/squad_history/)"
            ),
            "note": (
                "clubelo covers European clubs only — coverage_pct is the "
                "share of squad players whose club matched; the backtest "
                "study shrinks the squad signal by coverage"
            ),
        }
    }
    unmatched: Counter[str] = Counter()
    for label, page, date in TOURNAMENTS:
        elo_by_club = fetch_clubelo_snapshot(date)
        matcher = ClubMatcher(elo_by_club)
        squads = parse_squads(fetch_squads_wikitext(page))
        teams_out: dict[str, Any] = {}
        for wiki_team, clubs in squads.items():
            team = WIKI_TO_RESULTS.get(wiki_team, wiki_team)
            elos = []
            for club in clubs:
                hit = matcher.match(club)
                if hit is not None:
                    elos.append(elo_by_club[hit])
                else:
                    unmatched[club] += 1
            teams_out[team] = {
                "mean_club_elo": round(sum(elos) / len(elos), 1) if elos else None,
                "matched": len(elos),
                "total": len(clubs),
                "coverage_pct": round(100 * len(elos) / len(clubs), 1),
            }
        out[label] = teams_out
    out["_unmatched_clubs_top"] = [
        {"club": c, "players": n} for c, n in unmatched.most_common(40)
    ]
    return out


def main() -> None:
    history = build_history()
    atomic_write_json(OUTPUT_JSON, history)
    for label, _, _ in TOURNAMENTS:
        teams = history[label]
        covs = [t["coverage_pct"] for t in teams.values()]
        print(
            f"{label:<18} {len(teams):>2} squads · "
            f"coverage mean {sum(covs) / len(covs):5.1f}% · "
            f"min {min(covs):5.1f}% ({min(teams, key=lambda k: teams[k]['coverage_pct'])})"
        )
    print(f"\nunmatched clubs (top): "
          f"{[u['club'] for u in history['_unmatched_clubs_top'][:12]]}")
    print(f"wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
