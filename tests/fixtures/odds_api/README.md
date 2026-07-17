# The Odds API specimen

`event_envelope.json` is a **real** Odds API event envelope, not a hand-written
mock. Provenance: extracted 2026-07-16 from `data/cache/odds_totals_69392f7a4131.json`,
a bulk `/odds` response this codebase fetched and cached itself (Serie A,
2026-04-17 Sassuolo v Como). Its 25-bookmaker `bookmakers` array was stripped —
that is the **only** edit, and stripping it is what turns the `/odds` envelope
into the `/events` shape.

It exists because the Odds API key is deactivated for the off-season
(`DEACTIVATED_KEY`, HTTP 401), so `fetch_upcoming_matches` could not be built or
tested against a live call. This is the closest thing to a live specimen
available, and it pins every field the parser reads.

## What it establishes

`tests/test_fetch_upcoming_matches.py` maps this record through the real
`_event_to_match` and asserts the output. That makes the parser's schema claim
tested against genuine API data rather than asserted.

The four fields the parser depends on — `id`, `home_team`, `away_team`,
`commence_time` — are confirmed twice: here, and by the working consumer at
`odds_fetcher.py:776-783` that already extracts the same four from `/events`.

## What it does NOT establish

That live `/events` matches this envelope. It should — `/events` is this minus
`bookmakers` — but that is unconfirmed until the key is reactivated and real
fixtures exist (mid-August). See AUGUST_RUNBOOK.md.

Re-capture from a live `/events` call once the key is active, rather than
hand-editing this file.
