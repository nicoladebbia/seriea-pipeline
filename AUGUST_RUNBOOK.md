# AUGUST 2026 REACTIVATION RUNBOOK

Single source for bringing the betting pipeline back for Serie A 2026-27.
Written 2026-06-11. Execute top-to-bottom; each step has a verification.
The traps are real — read the ⚠️ notes before acting.

## 0. Current state you're starting from (set 2026-06-11)

- All launchd jobs are LOADED (the June-1 hibernation was reversed by the WC
  reload). `wc-refresh` exits instantly outside the tournament window after
  Jul 20 — harmless to leave loaded.
- `BETTING_DRY_RUN_FROM_MORNING=true` is ALREADY in morning+evening plists
  (file-level edit 2026-06-11) but only takes effect when launchd RE-loads
  them → step 4 makes it live.
- CLV kill-switch is live in code (apply_intelligence_filters): any market
  whose rolling 50-bet CLV goes negative stops getting bets automatically.
- Player floor engine is 19 markets, all validated; Player_Props betting
  stays GATED until the forward sample (step 8) proves edge.
- Odds API key is LAPSED (401) and plan possibly cancelled — step 2.

## 1. Confirm PR #2 (O/U 1.5 shrinkage) is merged — DO THIS FIRST

```bash
gh pr view 2 --json state,mergedAt
git log --oneline main | grep -i "shrinkage\|8af99e4" | head -3
```
If still open: merge it BEFORE reactivating, or the per-line shrinkage
(`line_shrinkage: {1.5: 0.6}` in betting_unified) is not live.

## 2. Odds API plan + key

- Re-subscribe / verify plan at the-odds-api.com dashboard (billing is
  Nicola-only). Update the key in `config/api_keys.json` if rotated.
- Verify: `python3 -c "from scripts.data.odds_fetcher import fetch_and_save_odds"`
  then one manual fetch; confirm no 401 and credits debit as expected.

## 3. Promoted teams 2026-27 — BEFORE first pipeline run

Three teams come up from Serie B. For each:
- [ ] `config/team_names.py` (or the normaliser map) — all spelling variants
- [ ] `scraper/weather.py` TEAM_TO_CITY — city entry (the EPL lesson: missing
      lookup = silently zero rows)
- [ ] Elo seed — `features/` Elo step seeds new teams; verify default-seed
      path fires (promoted teams typically seed ~1450 or league-min)
- [ ] `scripts/betting/player_predictions.py` TEAM_MAP if Sofascore spells
      them differently
- Verify: run one incremental pipeline, grep logs for the three team names,
  confirm fixtures + odds + features rows exist for each.

## 4. Reload launchd (arms the T-30 timing mode)

```bash
for p in ~/Library/LaunchAgents/com.seriea-pipeline.*.plist; do
  launchctl unload "$p" 2>/dev/null; launchctl load "$p"
done
launchctl list | grep seriea-pipeline | awk '$2 != 0 && $2 != "-" {print}'
```
⚠️ RunAtLoad is still TRUE on morning/evening → the reload itself fires one
immediate pipeline run each (wake-storm pattern). That's ONE duplicate run,
absorbed by the 60-min odds cache — acceptable. Do the reload once, not
repeatedly.
- Verify timing mode is live: next morning run's log must show bets as
  CANDIDATES ("no journal write"), and `run_pre_kickoff` commits them at
  T-30. If bets journal-write straight from the morning run, the env var
  didn't reach the job (check `launchctl print` for the env block).

## 5. O/U 1.5 shrinkage verification (the 4-point check)

After the first 2-3 matchweeks:
1. O/U 1.5 stakes SMALLER than pre-June journal entries (Kelly on ~4.6%
   deflated edge, not ~8%).
2. ~24% of marginal O/U 1.5 signals no longer become bets.
3. O/U 2.5 bets UNCHANGED (same stakes/selection as before). If 2.5 changed,
   something is wrong — investigate before continuing.
4. O/U 1.5 ROI regressing toward ~2.5% is the fix WORKING.
⚠️ THE TRAP: do NOT revert the shrinkage because ROI dropped from +10% to
~2.5%. The +10% was variance (top-3 bets = 72% of profit). Only revert if
the EXCLUDED 24% of bets turn out to be systematic winners (settle them on
paper and check).

## 6. Scanner + monitors back on (+ Betfair first contact)

- **Betfair feed first contact**: put `app_key` + `session_token` under
  `"betfair"` in `config/api_keys.json` (login at identitysso.betfair.it),
  then `python3 -m scripts.data.betfair_feed --probe`. The module was built
  2026-06-11 against the documented API-NG contract but NEVER live-tested —
  the probe IS the contract verification. It also prints the real Serie A
  competition id; if ≠ 81, fix SERIE_A_COMPETITION_ID. Only after the probe
  passes: schedule `--fetch` snapshots (toward-the-close history per match →
  the sharpest CLV benchmark we can get).

- `odds_edge_monitor.py` (soft lines, CLV-validated) — confirm its schedule
  fires; it also runs `scan_arbitrage` (cross-book arbs) for free.
- Health monitor: `curl -s localhost:5001/api/data-freshness | python3 -m json.tool`
  → `ok=true`, no CRITICAL.
- CLV kill-switch needs no activation (in-code), but verify after ~2 weeks:
  `python3 -c "from scripts.betting.clv_tracker import get_market_clv_gate; print(get_market_clv_gate('O/U 2.5','serie_a'))"`
  — expect a growing n and a sane clv_mean, not a permanently-insufficient 0.

## 7. Prop forward sample (the only path to un-gating Player_Props)

- Prop settlement now runs on the match clock automatically (scheduler
  settlement_check → settle_props), and per-event prop odds fetch costs
  6 cr/event — confirm `player_prop_odds.py` is on its schedule.
- Check The Odds API per-event player markets for Serie A: shots / SoT /
  goalscorer are known-present; **passes / tackles / duels / interceptions
  availability is UNVERIFIED** — probe one event and record which of the 19
  floor markets have real lines.
- Gate decision (NOT before ~100 settled props per market): positive ROI vs
  the line + the floor's prob beating the implied prob on a real sample →
  only then flip `betting_rules.json` Player_Props enabled_leagues.

## 8. EPL stays gated

Dev track only. Retrain on completed 2025-26, run shadow/paper all season,
collect CLV. The bar to enable real EPL bets = the same bar Serie A met:
positive CLV z-score on a real sample + model accuracy above floor. 52.5%
accuracy + 5 historical bets is NOT that bar. Do not re-litigate without
new evidence.

## 8b. ⚠️ KNOWN BLOCKER — fix weekly_retrain BEFORE the first matchweek retrain

The 2026-06-11 dry-run exposed three defects (logs/retrain.log, task #7):
1. `catboost_no_odds` aux retrain crashes on NaN in y_true with the completed
   2025-26 frame — the production weekly retrain WILL fail in August.
2. `--dry-run` still overwrites production xg_home/xg_away in place (aux
   section has no gate). xG models were refreshed 2026-06-11 ungated —
   re-validate or accept explicitly.
3. Teardown hang after the draw-detector phase (main-thread cond_wait) —
   the job never exits; needs a join fix or timeout.
Promote decision from that dry-run: candidate REJECTED (acc 51.7% vs 53.3%,
ECE worse) — incumbent catboost_no_odds.cbm (Apr 24) stays production.

## 9. First-week sanity watchlist

- `logs/launchd-morning-err.log` — no duplicate STARTING lines at odd hours
  (wake-storm); odds spend roughly = one fetch per scheduled run.
- `data/pipeline_state.json:last_odds_fetch` bumping on every fetch.
- Bankroll ledger invariant: journal-derived balance == bankroll.json.
- Telegram digest arriving; track-record page grading new matches.
- Calibration: live ECE on the first ~100 1X2 predictions < 0.06.

## Quick reference — what was built in June (off-season)

| Thing | Where | State |
|---|---|---|
| 19-market floor engine | scripts/betting/player_predictions.py | live, betting-gated |
| CLV kill-switch | clv_tracker.get_market_clv_gate + apply_intelligence_filters | live in code |
| T-30 candidate mode | morning/evening plists env var | arms at step 4 reload |
| Prop auto-settlement | scheduler settlement_check | live at reload |
| Result-pinned WC sim | scripts/worldcup/ | WC-only, sunsets Jul 20 |
| Betfair feed / lineup scraper | NOT BUILT — master plan WS3.3/3.7 | next build block |
