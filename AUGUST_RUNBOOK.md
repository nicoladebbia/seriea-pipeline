# AUGUST 2026 REACTIVATION RUNBOOK

Single source for bringing the betting pipeline back for Serie A 2026-27.
Written 2026-06-11. Execute top-to-bottom; each step has a verification.
The traps are real — read the ⚠️ notes before acting.

## 0. Current state you're starting from (set 2026-06-11)

- Only THREE jobs are loaded (2026-06-12, Nicola's request — the SA/EPL
  pipeline was spamming parlay/digest messages off-season): `wc-refresh`,
  `telegram-bot`, `web-dashboard`. Everything else was `launchctl unload`ed
  with plists intact on disk → step 4's reload restores the full set.
  `wc-refresh` exits instantly outside the tournament window after Jul 20.
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

## 3a. 🚨 Two plists hardcode `--season 2025-2026` — bump BEFORE §4 reload

Found 2026-07-16 while exercising the jobs. Both of these pin the season in
`ProgramArguments`, so on reload they refresh the **completed 2025-26 season**
and never touch 2026-27. They exit 0 and log success — green, and useless.
The modules are fine; the *invocation* is the bug.

- [ ] `scripts/pipeline/com.seriea-pipeline.scrape-epl-current.plist`
      → `--season 2025-2026` → `2026-2027`
- [ ] `scripts/pipeline/com.seriea-pipeline.refresh-understat.plist`
      → `--season 2025-2026` → `2026-2027`
- [ ] Edit the **installed** copies in `~/Library/LaunchAgents/` too, or
      reinstall from the repo copies — launchd reads the installed ones.
      Edit XML form only; never `plutil -convert json` a launchd plist.
- [ ] `scripts/pipeline/refresh_weekly_data.py:27` — `CURRENT_SEASON =
      "2025-2026"`. Same trap, module constant instead of a plist arg: it is
      what Step 2/3 pass to all five parsers, so the whole weekly FBref path
      refreshes the completed season until this is bumped.

Check every plist for a pinned season while you're in here:
```bash
grep -l "2025-2026" ~/Library/LaunchAgents/com.seriea-pipeline.*.plist
```

## 3b. 🚨 PHANTOM MODULES — 15 missing files. FIX BEFORE §4 RELOAD

**Discovered 2026-07-16.** 15 modules that live jobs invoke **do not exist on
disk and were never committed**. Hibernation hides this: nothing has run since
2026-06-01, so the §4 reload is the moment it all detonates.

### Mechanism (so it doesn't recur)

These were written locally, ran fine, and were **never `git add`ed** — no
`.gitignore` involvement (checked: `git check-ignore` says NOT ignored; only
`/data/` and `*.pyc` are). They were swept **after 2026-06-01** by a cleanup
deleting "untracked" files. Evidence they were alive on Jun 1:
`logs/launchd-weekly-data-refresh.log` reports **"11/13 steps OK"** at
2026-06-01 07:59 with **zero** `No module named` errors, and
`data/parsed/player_stats.parquet` is stamped Jun 1 07:59.

### The list (verified 2026-07-16 — file absent AND never in git history)

**Plist-invoked (job dies on first tick):**
| Job | Module | Note |
|---|---|---|
| `sofascore-watcher` | `scripts.data.sofascore_watcher` | spec recovered — see §3c |
| `refresh-understat` | `scripts.data.refresh_understat_players` | writer of `understat_players.parquet` (11.7k rows, LIVE — read by `features/understat_features.py`, `features/lineup_xg.py`, `web/app.py`) |
| `transfer-refresh` | `scripts.data.refresh_transfers` | **PR #7 (draft) adds this one** — land it |

**`refresh_weekly_data.py` (weekly-data-refresh.plist, Mon 04:00) — the whole FBref path:**
`scrape_fbref_missing` (Step 2) · `parse_all_player_stats` · `parse_all_lineups`
· `parse_all_events` · `parse_all_goalkeeper_stats` · `parse_all_shots` (Step 3)
· `fallback_sofascore_to_fbref` (:219). Steps 2–3 are subprocess calls, so they
fail **loudly** (exit 1) but `run()` records the step and continues.

> ✅ **Step 3's five parsers are REBUILT** (2026-07-16). They only read cached
> HTML, so they work with FBref blocked. **`--append` replaces only the parsed
> *matches*, never a whole season** — `lineups`/`events` hold hundreds of matches
> under the canonical `date_Home_Away` id that FBref parsing does not own, and
> `player_stats` holds the `sofascore_fallback` rows.
>
> **What was actually verified** (state it precisely — the earlier wording here
> claimed more than was proven):
> - **2025-2026 — the only season the job parses — is exact for all five.** Each
>   module's `main()` was run through the real CLI (`--season 2025-2026 --append`)
>   with `OUTPUT_PATH` redirected to a scratch copy; the written file matched the
>   stored parquet under `assert_frame_equal`, column order included. Every live
>   parquet was md5-confirmed unchanged.
> - **2024-2025 re-parse:** `lineups` and `events` reproduce; `player_stats`
>   **does not** (see below); `shots` reproduces (it is that module's only season).
> - Rows from *other* seasons in a passing full-file check were **preserved by the
>   merge, not re-parsed** — that is not evidence those seasons re-parse.
>
> ⚠️ **`--season` is REQUIRED on all five (no all-seasons default).** The stored
> 2024-25 `player_stats` slice was written by an earlier parser against an HTML
> capture that no longer exists: it holds **coarse** positions (`DF`) and the
> prefixed `misc_`/`possession_`/`defense_` columns, where every other season —
> including 2025-26, which we reproduce byte-exact — holds **fine** positions
> (`CB`) and the unprefixed `fouls`/`fouled`/… columns. Today's cached 2024-25
> HTML parses to fine positions, so the stored slice is **not reproducible from
> surviving inputs**. A bare run would have silently rewritten 10,096 position
> values in a *training* season. It now fails loudly instead. Do **not** "fix"
> this by mapping fine→coarse: that invents provenance and would corrupt 2025-26.
> (DATA_CATALOG.md's step table showed the bare form until 2026-07-16 — corrected,
> since following the doc was itself the path into this.)
>
> ⚠️ **`parse_all_shots` has nothing to parse for 2025-26**: those cached reports
> carry no `shots_all` table (0/40 sampled vs 40/40 for 2024-25 — they carry only
> the summary player table, which is also why 2025-26 has 29 filled columns where
> 2024-25 has 133). Step 3's shots call therefore exits 1 until the HTML is
> re-downloaded. Not a parser bug; matches what DATA_CATALOG.md says about
> `shots.parquet`.
>
> ⚠️ **`parse_all_lineups` lost `--include-epl` / `--epl-only`** — DATA_CATALOG.md
> documented both on the original (swept) module; the rebuild reconstructs only
> the Serie A path `refresh_weekly_data.py` invokes. **Existing EPL rows are
> safe** (the merge replaces only parsed match_ids — 271,530 rows / 6,427
> other-convention matches verified surviving a 2025-26 run). What is missing is
> *refreshing* EPL lineups from `{season}_epl/` HTML. The oracle for building it
> exists locally (`data/raw/html/2025_2026_epl/` + the EPL rows already in
> `lineups.parquet`), so this is buildable offline — it was left out to keep the
> rebuild scoped to what the live job runs. Old flag now errors loudly.
>
> Still missing: **`scrape_fbref_missing`** (Step 2 — needs FBref reachable, so
> it is genuinely August work) and **`fallback_sofascore_to_fbref`** (:219, see
> below).

**Imported by live code (raises ImportError at the call site):**
| Module | Caller | Status |
|---|---|---|
| `scripts.data.fetch_upcoming_matches` | `scripts/pipeline/scheduler.py:923` | ❌ not rebuilt — source unknown |
| `scripts.data.odds_tracker` | `run_full_pipeline.py:362`, `betting_unified.py:3565`, `web/app.py:5506,5836` | ✅ **REBUILT** — see caveat below |
| `scripts.data.live_sofascore` | `scripts/data/live_monitor.py:1140` | ❌ not rebuilt — Sofascore 403 |
| `scripts.data.live_reconciliation` | `scripts/data/live_monitor.py:1181` | ❌ not rebuilt — semantics unrecoverable |
| `scripts.data.backfill_historical_odds` | `run_full_pipeline.py:1337` (subprocess) | ⏳ buildable — source reachable |

#### Triage of these five (2026-07-16)

**`odds_tracker` — REBUILT.** Highest value of the five: every caller wraps the
import in a bare `except`, so its absence is **silent** — no snapshots written,
and `clv_tracker` (`:217` globs `bookmakers_*.json`, `:296` globs `extra_*.json`)
loses closing odds. CLV is the only rock-solid edge signal in this repo, so the
failure mode is "the edge signal quietly dies", not "a job errors".

Reconstructed from 1,098 surviving snapshots + 17 timestamped log lines + the
last `odds_movement.json`. Verified: reproduces that file **exactly** (85/85
fields, 5 matches) and all **17/17** logged summaries, through the real
entrypoint, with live data md5-confirmed untouched.

⚠️ **The one thing NOT pinned — confirm on a live run.** The 48h window is
uniquely identified (24h/36h/72h are refuted; it also reproduces the stored
`snapshots_count`/`hours_tracked` exactly). The **thresholds are not**: sweeping
the plane against all 17 runs leaves `LINE ∈ [0.09, 0.11]` and
`STEAM ∈ [0.125, 0.155]`. Shipped values are the natural round ones (0.10 /
0.15). Only 4 of the 17 runs are informative — the other 13 are the degenerate
all-zero case any threshold satisfies. `is_steam_move` feeds a **0.8-weight**
component in `features/market_intelligence.py` → `betting_unified`, so a wrong
threshold shifts betting scores for matches moving in that narrow band.

⚠️ **Do NOT try to confirm this on the first run after reload — it cannot work.**
All 3,294 existing snapshots will be ~2 months stale, so every one falls outside
the 48h window. The first runs see only the snapshot just written → all movements
`0.0` → `line_moves: 0, steam_moves: 0` **whatever the thresholds are**. That is
the degenerate case that validates nothing (it's why only 4 of the original 17
runs were informative). A `0/0` summary post-reload is **expected and healthy**,
not a bug — do not "fix" it.

The confirmation needs **both** preconditions:
1. **≥48h of fresh post-reload snapshots** have accumulated (i.e. ≥2 snapshots
   for the same match spanning >0h), and
2. **at least one match actually moved ≥0.10** in-window — otherwise there is
   still nothing to separate.

Only then does comparing a live `summary` against the constants mean anything.
The cheap way to get a real check: once (1) holds, recompute `max(|dH|,|dD|,|dA|)`
per match straight from `data/odds_snapshots/` and confirm the count crossing
0.10/0.15 equals the logged `line_moves`/`steam_moves` — same method as the
original sweep, which is reproducible from the snapshots alone.
Also unverified: `direction`'s `home_drifting`/`away_drifting` branches (the only
surviving per-match output is the all-`stable` run) and the
`implied_prob_shift_*` formula (**0 consumers** — grep-verified, so harmless).

**`backfill_historical_odds` — buildable, not yet built.** Reads
football-data.co.uk (**HTTP 200**, 196KB — reachable), *not* the Odds API, so the
dead key doesn't block it (`DATA_CATALOG.md:179`). Tracked helper
`scraper/historical.py` already implements the import; oracle = the odds columns
in `matches.parquet`. It writes to `matches.parquet` (ground truth), so treat it
as a destructive path and dry-run first.

**`live_sofascore` — blocked, do not build.** Needs Sofascore; **both** tiers are
403 (api *and* www — the CLAUDE.md HTML fallback is currently dead too). Only its
*output* (`data/live/*.json`) survives, not the raw inputs, so there is nothing to
verify a parser against. Per the never-parse-an-unverified-source rule: recorded,
not guessed.

**`live_reconciliation` — genuine phantom, and a name trap.** ⚠️
`scripts/analysis/live_reconciliation.py` **exists and is tracked** but is a
*different module* that merely shares the name — it has no
`reconcile_all_matches`. Repointing the import swaps `ModuleNotFoundError` for
`ImportError`; it does **not** fix anything. The contract is
`reconcile_all_matches(matchday) -> int` (discrepancy count), but what counts as a
"discrepancy" is not recoverable from any artifact.

**`fetch_upcoming_matches` — source unknown.** The only artifact,
`data/upcoming/matches.json`, is a *synthetic fallback*: `"source": "manual"`,
templated venues (`"Milan Stadium"`), and a `fetched_at` a week **after** the
matches it lists. It records the fallback, not the real fetch — so it can't
serve as an oracle.

⚠️ **NOT a phantom:** `scripts.pipeline.matchday_update` — `web/app.py:6517` is a
comment recording its *intentional* removal. Don't "restore" it.

### `fallback_sofascore_to_fbref` — mapping recovered, selection rule NOT

Investigated 2026-07-16 and **deliberately not rebuilt**. It needs no network
(`data/external/sofascore/player_match_stats.parquet` covers 64/64 of its
matches locally), so it is not blocked — but **it has no reproducible oracle**,
which is a different and more interesting problem.

It owns the `data_source="sofascore_fallback"` rows of `player_stats.parquet`
(2025-26: 2,020 rows across 64 matches, keyed by Sofascore numeric id — zero
overlap with FBref's 8-hex ids). The **column mapping is recovered and exact**
(row counts match per match, e.g. 32↔32):

| player_stats | ← player_match_stats.parquet |
|---|---|
| `player` | `player_name` |
| `shirtnumber` | `shirt_number` |
| `shots` | `total_shots` |
| `fouled` | `was_fouled` |
| `team` / `is_home` / `position` / `minutes` / `goals` / `assists` / `shots_on_target` / `fouls` / `interceptions` / `tackles` / `season` | same name |
| `data_source` | literal `"sofascore_fallback"` |

**Why it was not rebuilt:** the *selection* rule can't be recovered. For 2025-26
Sofascore has 380 SA matches and FBref parsing covers 260, leaving 120
uncovered — yet only **64** carry fallback rows, and **3 fallback ids are not in
the uncovered set at all**. So the stored set is an artifact of *run history*
(what was missing on the day each weekly run fired), not a function of today's
data. Any rule written now would produce a different set (~120) and write it
into a live parquet. That is inventing policy, not restoring a module — so the
finding is recorded instead. Decide the intended rule explicitly in August, then
build to it; the transform above can be verified exactly against the 64.

### Source reachability (measured 2026-07-16 — gates what can be rebuilt)

| Source | State | Consequence |
|---|---|---|
| FBref | Cloudflare-blocked | `scrape_fbref_missing` can't fetch new HTML. **But 309 cached EPL match reports** in `data/raw/html/2025_2026_epl/` parse fine offline. |
| Sofascore | **HTTP 403** (IP ban, still active) | `sofascore_watcher` standings scrape unverifiable. |
| Understat | **HTTP 200 but JS shell** (`playersData` count = 0 via plain HTTP) | needs Selenium — catalog confirms. Not an IP ban. |

### Prevention rule (worth more than the modules)

Nothing checks that an invoked module exists. This one-liner would have caught
all 15 on day one — run it before any reload, and consider wiring it into the
health check:

```bash
# Every scripts.* module referenced in code or plists must exist on disk.
# Resolves packages, and skips a trailing segment that is a symbol not a module.
{ grep -rhoE '\bscripts\.[a-z_][a-z0-9_.]*' --include="*.py" scripts/ web/ ;
  grep -rhoE '\bscripts\.[a-z_][a-z0-9_.]*' ~/Library/LaunchAgents/com.seriea-pipeline.*.plist ; } \
| sed 's/\.$//' | sort -u | while read m; do
    p="$(echo "$m" | tr '.' '/')"
    [ -f "$p.py" ] && continue          # module
    [ -d "$p" ] && continue             # package
    par="${p%/*}"; [ -f "$par.py" ] && continue   # symbol imported from a module
    echo "MISSING: $m"
  done
```
**Known limits — triage each hit, don't act blind (tested 2026-07-16, 22 hits):**
- **False positives:** stale *docstring* examples. 9 of the 22 hits (e.g.
  `scripts.learning_loop`, `scripts.feedback_analyzer`, `scripts.scrape_sofascore`)
  are files that moved into subpackages while their docstrings still say the old
  `python3 -m scripts.X`. The file exists; the doc is wrong. Same for
  `scripts.pipeline.matchday_update` (a comment about its removal).
- **False negatives:** it CANNOT see modules named via f-string. That is exactly
  how `refresh_weekly_data.py` builds the 5 `parse_all_*` parsers
  (`f"scripts.data.{parser_name}"`), so those never appear. Check Step 3's list
  by hand.
- Don't use `\s` in `sed` on macOS (BSD sed lacks it) — it silently fails to
  strip and every line reports MISSING.

**And: `git status` before ending any session that adds a script.** Untracked
load-bearing code is the actual root cause here, not the cleanup that swept it.

### ⚠️ Do NOT create stub modules to silence this

A stub that imports but has no `__main__` turns a **loud** `No module named`
(exit 1, visible) into a **silent no-op that `run()` logs as OK**. Missing is
strictly better than fake. Rebuild properly or leave absent.

## 3c. `sofascore_watcher` — recovered spec (rebuild from this)

The module is gone but its behaviour is fully documented. **This spec was
reverse-engineered 2026-07-16 from the plist comment + a 55-tick log + the
surviving state file; the logs can rotate away, so this is now the record.**

- **Spec** (from `scripts/pipeline/com.seriea-pipeline.sofascore-watcher.plist`,
  whose XML comment survived — the `~/Library/LaunchAgents` copy lost it):
  every 10 min; refresh standings each tick (cheap HTML scrape); full API
  refresh when a match just ended (kickoff +110–140min), a match kicks off soon
  (kickoff −30 to −90min), or **every 18th tick** (~3h heartbeat). It replaced
  `sofascore-refresh` (3h) and `match-end-refresh` (10m), neither now installed.
- **Proven behaviour** (`logs/sofascore_watcher-err.log`, 55 ticks 2026-04-30
  14:42 → 05-01 00:18, zero errors). `full=True` fired at exactly ticks 18/36/54
  — the heartbeat, confirmed. Its three log lines were:
  `standings json refresh {league}: wrote {n} teams` ·
  `running full API refresh — reason: periodic` ·
  `watcher tick={n} done in {t}s (full={bool}, post={n}, pre={n})`
- **Collaborators** (from logger names in that log): `scripts.data.scrape_sofascore`
  (fixtures cache) and `scraper.sofascore_events` (incidents). It did **NOT**
  import `web.app` — zero `Auto-settle scheduler started` lines.
- **State:** `data/external/sofascore/.watcher_tick.json` — `{"tick": N,
  "updated_at": ISO}`. Survives at tick 3055 (2026-06-01), so the counter
  persists across the per-tick process launches. Ticks ran 1.1–5.9s.
- **Writes:** `data/upcoming/standings.json` (serie_a) and
  `standings_premier_league.json` — the convention in
  `ensemble_prediction_engine.py:1238-1240`. 20 teams each.

**Open question for August (the one real design fork):** the standings HTML
scraper lived inside the watcher and is gone. The only surviving one is
`web/app.py:_live_standings_via_html` (sentinel-checked, breaker-guarded) — but
**importing `web.app` starts the auto-settle thread** (measured: 0.22s import,
logs "Auto-settle scheduler started"). It sleeps 300s first and is a daemon, so
it's harmless for a 1–6s tick — **but a post-matchday full refresh rate-limits
at 2s/match and can exceed 300s, which would fire the ledger settler and spend
Odds API credits from inside a watcher tick.** So: either extract the scraper
out of `web/app.py` into a shared module (both callers import it), or give the
watcher its own. **Requires a live (non-403) Sofascore to verify — that's why
it wasn't built on 2026-07-16.**

## 4. Reload launchd (arms the T-30 timing mode)

🚨 **Do §3b first.** 3 of these jobs (`sofascore-watcher`, `refresh-understat`,
`transfer-refresh`) point at modules that no longer exist and will die on their
first tick, and `weekly-data-refresh` will fail its whole FBref path. The
`launchctl list` check below shows nonzero exit codes — that's what they mean.

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

## 8b. ✅ RESOLVED 2026-07-15 — weekly_retrain defects fixed BEFORE August

The 2026-06-11 dry-run exposed three defects. All three closed 2026-07-15:

1. ✅ **`catboost_no_odds` crashes on NaN y_true** — cause: 13 final-round 2025-26
   matches have result="U" (scores never parsed, pipeline hibernated mid-May); "U"
   maps to NaN. FIX: loud-exclude of unlabeled rows in `retrain_no_odds_catboost.py`
   (logs count + names each, never silent). Commit f240272. In August with a complete
   season it's a no-op. See memory project_jul15_retrain_datagap.
2. ✅ **`--dry-run` overwrote xg_home/xg_away** — cause: `retrain_xg_models` never passed
   dry-run, and `train_unified.py` had no dry-run flag. FIX (two layers): added `--dry-run`
   to train_unified (honoured by xg_only mode → skips save_model), and `retrain_xg_models`
   now forwards it. VERIFIED: `retrain_xg_models(dry_run=True)` leaves both .cbm mtimes
   byte-unchanged. (The separate "aux xG has no quality GATE on the real run" is still
   true and still just an accept-explicitly item — not fixed, not a crash.)
3. ✅ **"Teardown hang / never exits"** — MISDIAGNOSIS confirmed by measurement. Four
   non-destructive probes (import; draw-detector isolated; minimal in-process train;
   full `train_universal(validate=True, all models)` + walk-forward CV) ALL exited clean
   with ZERO lingering threads. The draw-detector "hang" was just SLOWNESS: the added
   2025-26 season made it a 3-fold ablation, ~200s total — under the parent timeout=600,
   not a deadlock. The real signature (`cond_wait` pool-join at teardown) was NOT
   reproducible on 2026-07 code. HARDENING SHIPPED anyway: `main()` now does
   flush + `os._exit(code)` (branch-independent — exits clean whether the rare hang
   recurs or not; no atexit registered, all state persisted in-body, exit code preserved).

Promote decision from the ORIGINAL 2026-06-11 dry-run: candidate REJECTED. The 2026-07-15
loud-exclude dry-run also REJECTED (acc 0.489 < 0.50, ll 1.008 > 1.00) — the 2025-26 fold
is the anchor because the season is incomplete (367/380). Incumbent stays production.
August's normal backfill→rebuild→retrain on 380/380 fixes this cleanly.

## 8c. Backfill the 13 missing 2025-26 scores — BEFORE the first retrain

The 2025-26 season is incomplete in the data: 13 final-round matches
(2026-05-03 → 05-11, rounds 35-36) were PLAYED but their scores never got
parsed — the pipeline hibernated mid-May. In `matches.parquet` they are the
only rows with NaN `home_score`/`away_score`; downstream they surface as
`result="U"` in `features_serie_a.parquet` (380 rows, 367 labelled).

**The scores are already captured** — `data/parsed/staged_sa_2025-26_missing_scores.json`
(scraped 2026-07-16 from Sofascore season 76457, VERIFIED: 342 known-true
matches.parquet rows agreed, 0 disagreements). Re-scraping in August is fine
too; the staged file exists because the source had been unreachable for a
month and might have been again.

Do NOT hand-write the parquet. Apply through the normal flow:
1. Write the 13 scores into `matches.parquet` via the normal parser path.
2. `build_features(use_cache=False)` — a FULL rebuild. This is destructive
   (overwrites `features_serie_a.parquet`) and it is REQUIRED: 20 real-result
   matches (05-17 → 05-24) had elo/form computed while the 13 upstream rows
   were unlabelled, so the corruption is ≥33 rows, not 13. Model `--rollback`
   covers `data/models/` ONLY, not features — take a backup first.
3. Retrain: `python3 -m scripts.models.retrain_no_odds_catboost --walkforward-final --n-seeds 3 --fit-calibrator`
4. Verify: `features_serie_a.parquet` 2025-26 must show 380/380 labelled, 0 "U".

⚠️ Separately, 30 rows in `matches.parquet` 2025-26 HAVE scores but a null
`result` (e.g. Roma 2-0 Lazio on 05-17). The features build derives `result`
from the scores so this is currently harmless, but anything reading
`matches.parquet:result` directly would see 43 nulls, not 13. Worth a look
during the backfill.

The 2026-07-15 loud-exclude retrain REJECTED (acc 0.489 < 0.50) precisely
because the 2025-26 fold is the newest, most-weighted AND the compromised
one. On complete data this fold stops being the anchor.

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

## Quick reference — what was hardened mid-June (scanner audit session)

| Thing | Where | State |
|---|---|---|
| Scanner reads BettingConfig enabled-flags | odds_edge_monitor.py `_build_edge_thresholds` | live; only O/U_Over surfaces as actionable, 1X2/DC/O/U_Under → watchlist (flagged "no proven edge") |
| 1X2 Home now evaluated (was silently dropped) | odds_edge_monitor.py selection loop | live; routed to watchlist (1X2 disabled), not actionable |
| De-vig fixes (no phantom edges) | odds_edge_monitor.py 1X2 fallback + O/U pair + live | live; fallback de-vigs best-line triplet, O/U skips when no real under price, live path de-vigs + ±50% clamp; `no_sharp_ref` flag added |
| LIVE_MONITORING gate (default OFF) | odds_edge_monitor.scan_live_value + run_full_pipeline | live; in-play polling off until `LIVE_MONITORING=1`. **August: leave OFF unless betting live.** |
| O/U Over 2.5 validation logging | prediction_tracker.score_predictions | live; logs `model_over_2_5` (Poisson from xG, NOT the bogus `over_25` boolean flag), `home/away_xg`, `total_goals`, `over_2_5_hit` |

### CRITICAL August check — validate O/U Over before trusting it
The scanner's ONLY enabled market (O/U Over) has NEVER been validated on real
data — the "+10% ROI" is a BettingConfig comment, not a reproduced number. The
June scanner backtest couldn't test it because the tracker logged 1X2 probs
only. That gap is now closed (tracker logs O/U prob + outcome as of mid-June):
1. After ~4-6 weeks of resumed Serie A, backtest `model_over_2_5` vs
   `over_2_5_hit` in `prediction_tracker.json:scored`: skill score
   `1 - brier/baseline_brier > 0` AND realized ROI at the de-vigged edge.
2. If skill ≤ 0 or ROI negative over 40+ bets, O/U Over is NOT the earner the
   config claims — gate it like the others. Do NOT keep betting on faith.
3. Until that backtest exists, treat O/U Over bets as PROVISIONAL.
