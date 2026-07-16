"""Cross-source reconciliation of live match scores.

Compares the Odds API score (authoritative, always primary) against goals counted
from Sofascore live events, and records the outcome on each match under
``reconciliation``. Called by ``scripts/data/live_monitor.py`` once per polling
cycle; the return value is only logged, never gated on.

Rebuilt 2026-07-16 against the 123 stored output blocks in ``data/live/*.json``
(see AUGUST_RUNBOOK.md §3b). Replay-verified: ``tests/test_live_reconciliation.py``
regenerates every stored block exactly.

Two deliberate deviations from the recovered spec, both evidence-backed:

* The Odds API score is read from the **last scored snapshot**, not from
  ``_last_home_score``/``_last_away_score`` as the spec claimed. Those keys are
  absent from 45 of the 123 persisted blocks; the snapshot reproduces all 123.
* ``timing_mismatch`` is **not implemented**. Its trigger is unrecovered: two
  hypothesis families were swept against the oracle and both refuted (see
  ``_TIMING_MISMATCH_NOT_IMPLEMENTED``). It is ``info`` severity and does not
  affect ``scores_agree``, so omitting it is the safe default.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# Odds API statuses that imply an upper bound on the match clock. A Sofascore
# event past the bound means the Odds API status feed is lagging.
# Derived from a single oracle case (first_half + events at 90'); no stored block
# exercises any other status, so only these two are checked.
_STATUS_MAX_MINUTE = {"first_half": 45, "half_time": 45}

# Why timing_mismatch is absent, so nobody re-runs the search:
#   Truth flags 5 of 104 Sofascore goals in dual-source matches.
#   (1) "goal aligns if an odds_api score change falls within TOL minutes",
#       TOL swept 0..15: flags 98 at 0' and 26 at 15'. Wrong shape, not a mistuning.
#   (2) (1) plus a coverage guard skipping goals before the first snapshot minute
#       (motivated by Cagliari-Napoli: goal at 2', snapshots start at 9' with the
#       score already 0-1, and truth does NOT flag it): best case TOL=6 catches
#       5/5 but still drags in 39 false positives.
# Both refuted against the full oracle. Do not ship a tolerance window.
_TIMING_MISMATCH_NOT_IMPLEMENTED = True


def _odds_api_score(match: dict[str, Any]) -> list[int] | None:
    """Latest Odds API score, read from the most recent snapshot carrying one."""
    for snap in reversed(match.get("snapshots") or []):
        score = snap.get("score")
        if score:
            return [int(score[0]), int(score[1])]
    return None


def _odds_api_status(match: dict[str, Any]) -> str | None:
    for snap in reversed(match.get("snapshots") or []):
        if snap.get("status"):
            return str(snap["status"])
    return match.get("status")


def _goal_events(match: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in (match.get("live_events") or []) if e.get("type") == "goal"]


def _sofascore_score(match: dict[str, Any]) -> list[int]:
    """Goals counted from Sofascore live events, grouped by the side credited.

    ``is_home`` marks the *scorer's* side, so an own goal credits the opponent.
    """
    home = away = 0
    for event in _goal_events(match):
        credited_home = bool(event.get("is_home"))
        if event.get("goal_type") == "ownGoal":
            credited_home = not credited_home
        if credited_home:
            home += 1
        else:
            away += 1
    return [home, away]


def _latest_event_minute(match: dict[str, Any]) -> int:
    return max((e.get("minute") or 0) for e in (match.get("live_events") or [])) if match.get("live_events") else 0


def reconcile_match(match: dict[str, Any]) -> dict[str, Any] | None:
    """Build the reconciliation block for one match, or None if unreconcilable."""
    odds_score = _odds_api_score(match)
    if odds_score is None:
        return None

    sources = ["odds_api"]
    all_scores: dict[str, list[int]] = {"odds_api": odds_score}
    discrepancies: list[dict[str, Any]] = []

    # Sofascore joins the comparison only once it has actually delivered events —
    # a bare sofascore_id is not enough (2 oracle blocks have the id and no events).
    if match.get("live_events"):
        sources.append("sofascore")
        ss_score = _sofascore_score(match)
        all_scores["sofascore"] = ss_score

        if ss_score != odds_score:
            discrepancies.append({
                "type": "score_mismatch",
                "severity": "critical",
                "source": "sofascore",
                "reported": ss_score,
                "expected": odds_score,
                "message": (
                    f"All sources disagree: sofascore reports {ss_score[0]}-{ss_score[1]}, "
                    f"using odds_api {odds_score[0]}-{odds_score[1]} as primary"
                ),
            })

        status = _odds_api_status(match)
        max_minute = _STATUS_MAX_MINUTE.get(status or "")
        latest_minute = _latest_event_minute(match)
        if max_minute is not None and latest_minute > max_minute:
            discrepancies.append({
                "type": "status_mismatch",
                "severity": "info",
                "odds_api_status": status,
                "latest_event_minute": latest_minute,
                "message": (
                    f"Odds API says '{status}' but Sofascore has events at "
                    f"{latest_minute}' — status may be lagging"
                ),
            })

    scores_agree = not any(d["type"] == "score_mismatch" for d in discrepancies)
    severities = {d["severity"] for d in discrepancies}
    severity = "critical" if "critical" in severities else "info" if severities else "ok"

    return {
        "scores_agree": scores_agree,
        "sources_checked": sources,
        "discrepancies": discrepancies,
        "score_used": {
            "home": odds_score[0],
            "away": odds_score[1],
            # Odds API is primary on any disagreement; "consensus" only when
            # more than one source was checked and they agreed.
            "source": "consensus" if len(sources) > 1 and scores_agree else "odds_api",
        },
        "all_scores": all_scores,
        "severity": severity,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def reconcile_all_matches(matchday: dict[str, Any]) -> int:
    """Reconcile every match in ``matchday`` in place.

    Returns the number of discrepancy *items* found (not the number of matches
    carrying them); the caller only logs it.
    """
    total = 0
    for key, match in (matchday.get("matches") or {}).items():
        try:
            block = reconcile_match(match)
        except Exception as exc:  # noqa: BLE001 - one bad match must not sink the cycle
            log.warning("Reconciliation failed for %s: %s", key, exc)
            continue
        if block is None:
            continue
        match["reconciliation"] = block
        total += len(block["discrepancies"])
    return total
