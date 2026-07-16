"""Tests for the richer TM parse + Wikipedia second source + cross-source merge.

Locks the defects/behaviors found and fixed 2026-07-14:
  1. TM fee-vs-market-value: a row showing BOTH a market value and a fee must
     store the FEE as fee_eur (the old select_one grabbed the first td.rechts =
     market value, e.g. Højlund's €44m fee was stored as his €60m value).
  2. TM captures position, nationality, from_club/to_club on the /plus/1 layout.
  3. Wikipedia rowspan date-carry: rows that inherit a rowspan-merged date (4
     cells, not 5) must be kept, not dropped.
  4. Cross-source merge: n_sources flag, and loan-return rows must NOT receive a
     (wrong-event) Wikipedia date.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scraper.transfermarkt import _parse_transfers_page
from scraper.wiki_transfers import _clean_fee, _derive_window, _parse_date, _parse_wiki_page


# --- 1 & 2: TM parser (synthetic HTML matching the verified /plus/1 layout) ---

# A minimal arrivals table with the two-column money layout: td.rechts[0] is the
# market value, td.rechts[-1] is the fee. Includes the inline name/position table,
# a nationality flag img, and a club link with a crest img (full-name alt).
_TM_HTML = """
<html><body>
<h2>Arrivals</h2>
<table class="items"><tbody>
<tr>
  <td class="zentriert">1</td>
  <td class="hauptlink">
    <table class="inline-table"><tr><td><a href="/spieler/1">Rasmus Højlund</a></td></tr>
    <tr><td>Centre-Forward</td></tr></table>
  </td>
  <td class="zentriert"><img class="flaggenrahmen" title="Denmark"/></td>
  <td class="zentriert">23</td>
  <td class="rechts">&#8364;60.00m</td>
  <td class="zentriert"><a href="/manchester-united/startseite/verein/985"><img title="Manchester United"/>Man Utd</a></td>
  <td class="rechts">&#8364;44.00m</td>
</tr>
</tbody></table>
</body></html>
"""


def test_tm_fee_not_market_value() -> None:
    # The FEE (€44m) must be stored as fee_eur, NOT the market value (€60m).
    arrivals, _ = _parse_transfers_page(_TM_HTML, "Napoli")
    assert len(arrivals) == 1
    row = arrivals[0]
    assert row["fee_eur"] == 44_000_000.0, "fee must be €44m, not the €60m market value"
    assert row["market_value_at_transfer"] == 60_000_000.0


def test_tm_new_detail_fields() -> None:
    arrivals, _ = _parse_transfers_page(_TM_HTML, "Napoli")
    row = arrivals[0]
    assert row["position"] == "Centre-Forward"
    assert row["nationality"] == "Denmark"
    assert row["from_club"] == "Manchester United"  # full name from crest alt
    assert row["to_club"] == "Napoli"


def test_tm_single_money_column_is_the_fee() -> None:
    # A row with only ONE td.rechts (no separate market value) → that value is the
    # fee, and market_value_at_transfer is None (not mis-read).
    html = _TM_HTML.replace('<td class="rechts">&#8364;60.00m</td>', "")
    arrivals, _ = _parse_transfers_page(html, "Napoli")
    row = arrivals[0]
    assert row["fee_eur"] == 44_000_000.0
    assert row["market_value_at_transfer"] is None


# --- 3: Wikipedia rowspan date-carry ---

_WIKI_HTML = """
<table class="wikitable">
<tr><th>Date</th><th>Name</th><th>Moving from</th><th>Moving to</th><th>Fee</th></tr>
<tr><td>2 January 2026</td><td>Manor Solomon</td><td>Tottenham</td><td>Fiorentina</td><td>Loan[5]</td></tr>
<tr><td>Second Signing</td><td>Ajax</td><td>Bologna</td><td>Undisclosed[4]</td></tr>
</table>
"""


def test_wiki_rowspan_date_carry() -> None:
    # The 2nd row has only 4 cells (its date is rowspan-merged from the row above).
    # It must be KEPT with the inherited date (both rows touch a Serie A club).
    rows = _parse_wiki_page(_WIKI_HTML)
    names = {r["player_name"] for r in rows}
    assert "Manor Solomon" in names
    assert "Second Signing" in names  # the 4-cell inherited-date row (→ Bologna)
    sec = next(r for r in rows if r["player_name"] == "Second Signing")
    assert sec["transfer_date"] == "2026-01-02"  # inherited from Solomon's row


def test_wiki_keeps_promoted_clubs() -> None:
    # A row into a PROMOTED club (Venezia) from a non-SA seller must be KEPT.
    # Regression guard: the club set once held last season's league (Cremonese/
    # Pisa/Verona) and dropped the promoted clubs' incoming transfers.
    html = """
    <table class="wikitable">
    <tr><th>Date</th><th>Name</th><th>Moving from</th><th>Moving to</th><th>Fee</th></tr>
    <tr><td>3 July 2026</td><td>New Signing</td><td>Palermo</td><td>Venezia</td><td>€2m</td></tr>
    <tr><td>4 July 2026</td><td>Relegated Move</td><td>Cremonese</td><td>Bari</td><td>Free</td></tr>
    </table>
    """
    rows = _parse_wiki_page(html)
    names = {r["player_name"] for r in rows}
    assert "New Signing" in names        # → Venezia (promoted, in the league)
    assert "Relegated Move" not in names  # Cremonese→Bari, both now Serie B


def test_wiki_clean_fee() -> None:
    assert _clean_fee("Undisclosed[nb 1]") == ("Undisclosed", None)
    assert _clean_fee("Free[2]") == ("Free", None)
    assert _clean_fee("Loan[3]") == ("Loan", None)
    txt, val = _clean_fee("€44m")
    assert val == 44_000_000.0


def test_wiki_derive_window() -> None:
    assert _derive_window(_parse_date("21 May 2026")) == "summer"   # May folded into summer
    assert _derive_window(_parse_date("3 June 2026")) == "summer"
    assert _derive_window(_parse_date("2 January 2026")) == "winter"
    assert _derive_window(_parse_date("15 October 2026")) is None   # out-of-window


# --- 4: cross-source merge (uses tmp parquets, no live scrape) ---


def test_merge_flags_and_skips_loan_return_date(tmp_path, monkeypatch) -> None:
    import scraper.wiki_transfers as wt
    monkeypatch.setattr(wt, "WIKI_DIR", tmp_path)
    # TM spine: one real arrival + one loan-return row for the same player.
    pd.DataFrame([
        {"player_name": "Rasmus Højlund", "transfer_type": "in", "fee_text": "€44.00m"},
        {"player_name": "Marco Brescianini", "transfer_type": "in",
         "fee_text": "End of loan30/06/2026"},
        {"player_name": "Unmatched Player", "transfer_type": "in", "fee_text": "€5.00m"},
    ]).to_parquet(tmp_path / "transfers_2026_2027.parquet", index=False)
    # Wikipedia: matches both named players (Brescianini's date is the WRONG event).
    pd.DataFrame([
        {"player_name": "Rasmus Højlund", "transfer_date": "2026-06-03", "window": "summer"},
        {"player_name": "Marco Brescianini", "transfer_date": "2026-01-09", "window": "winter"},
    ]).to_parquet(tmp_path / "wiki_transfers_2026_2027.parquet", index=False)

    df = wt.enrich_transfers_with_wiki("2026-2027")
    hoj = df[df["player_name"] == "Rasmus Højlund"].iloc[0]
    bre = df[df["player_name"] == "Marco Brescianini"].iloc[0]
    unm = df[df["player_name"] == "Unmatched Player"].iloc[0]

    assert hoj["n_sources"] == 2 and hoj["transfer_date"] == "2026-06-03"
    # Brescianini cross-confirmed (n_sources=2) but NO date (loan-return = wrong event)
    assert bre["n_sources"] == 2 and bre["transfer_date"] is None
    assert unm["n_sources"] == 1 and unm["transfer_date"] is None
