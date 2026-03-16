# Research Evidence

## Topic: Free/Affordable Sources for Detailed Per-Match, Per-Player Serie A Statistics
**Date:** 2026-02-09
**Accessed:** 2026-02-09

## Findings

### 1. Understat (understat.com)

**Coverage:**
- xG, npxG, xA: YES - per player per match
- SCA, GCA: NO
- Passes: NO
- Carries: NO
- Take-ons: NO
- Touches, tackles, blocks: NO
- PPDA/Pressing: YES - team level only via special endpoints

**Data Access:**
Understat provides free access to Serie A data via web scraping. Multiple Python libraries are available: `understatapi` (PyPI), `understat` (async Python library), and `worldfootballR` (R package). These APIs support per-match granularity with functions like `get_player_matches()` returning xG, xA, goals, assists, key passes, npxG, xGChain, and xGBuildup. [Source: https://understat.com, accessed: 2026-02-09]

**API/Scraping:** Scraping required (no official API, but third-party libraries available)

**Cost:** Free

**Serie A Coverage:** 2014/15 season onwards, including 2025/2026

**Per-Match Per-Player Granularity:** YES for xG metrics

[Source: https://pypi.org/project/understatapi/, accessed: 2026-02-09]
[Source: https://understat.readthedocs.io/en/latest/classes/understat.html, accessed: 2026-02-09]

---

### 2. FotMob (fotmob.com)

**Coverage:**
- xG, npxG, xA: YES (xG and xGOT available)
- SCA, GCA: Unknown
- Passes: YES (opposition half passes mentioned)
- Carries: Unknown
- Take-ons: Unknown
- Touches, tackles, blocks: Likely YES (detailed match stats available)
- PPDA/Pressing: Unknown

**Data Access:**
FotMob provides match statistics through an unofficial API endpoint at `https://www.fotmob.com/api/matchDetails`. The `soccerdata` Python library can access FotMob data with automatic local caching. However, detailed player-level APIs are not publicly available - implementations typically rely on search endpoints and provide browser links for complete player information. [Source: https://medium.com/@lmirandam07/fotmob-fixtures-data-extraction-with-python-f1ee03d2dfe, accessed: 2026-02-09]

**API/Scraping:** Unofficial API + scraping (no public player detail endpoints)

**Cost:** Free (unofficial access)

**Serie A Coverage:** Current season (2025/2026)

**Per-Match Per-Player Granularity:** LIMITED - match-level stats available, but player detail endpoints not public

[Source: https://soccerdata.readthedocs.io/en/latest/datasources/FotMob.html, accessed: 2026-02-09]
[Source: https://github.com/jamtur01/raycast_gfb, accessed: 2026-02-09]

---

### 3. WhoScored (whoscored.com)

**Coverage:**
- xG, npxG, xA: YES
- SCA, GCA: YES (likely - powered by Opta data)
- Passes: YES (passes completed, pass completion %)
- Carries: Likely YES
- Take-ons: YES (dribbles attempted/completed)
- Touches, tackles, blocks: YES (tackles won, interceptions)
- PPDA/Pressing: Unknown (not mentioned in search results)

**Data Access:**
WhoScored displays detailed match statistics compiled from Opta's event feed, including individual player stats for goals, assists, shots, tackles, interceptions, dribbles, passes, and key passes. Data can be accessed via scraping using tools like WebHarvy (30-60 seconds per match) or the `soccerdata` Python library. Match statistics pages have multiple tabs that must be scraped separately. [Source: https://www.whoscored.com/regions/108/tournaments/5/seasons/10732/stages/24500/playerstatistics/italy-serie-a-2025-2026, accessed: 2026-02-09]

**API/Scraping:** Scraping required (no official API)

**Cost:** Free (via scraping, subject to terms of service)

**Serie A Coverage:** Current season (2025/2026) and historical data

**Per-Match Per-Player Granularity:** YES - comprehensive per-match player statistics

[Source: https://soccerdata.readthedocs.io/en/latest/datasources/WhoScored.html, accessed: 2026-02-09]
[Source: https://www.webharvy.com/blog/scrape-whoscored-live-scores/, accessed: 2026-02-09]

---

### 4. Sofascore (sofascore.com)

**Coverage:**
- xG, npxG, xA: YES (xG available via Opta data)
- SCA, GCA: Unknown
- Passes: YES
- Carries: Unknown
- Take-ons: YES (likely)
- Touches, tackles, blocks: YES
- PPDA/Pressing: NO (PPDA typically requires paid subscriptions like Wyscout/Opta)

**Data Access:**
Sofascore provides an unofficial API at `https://www.sofascore.com/api/v1` with Serie A having league ID 23. Multiple Python wrappers exist: `sofascore-wrapper` (PyPI), `sofascore_scraper` (GitHub), and integration in `soccerdata` library. Match data includes possession, shots, passes, tackles, and can be filtered by "per90", "perMatch", or "total". [Source: https://www.sofascore.com/tournament/football/italy/serie-a/23, accessed: 2026-02-09]

**API/Scraping:** Unofficial API + Python wrappers available

**Cost:** Free (unofficial access)

**Serie A Coverage:** Current season (2025/2026) and historical data

**Per-Match Per-Player Granularity:** YES - detailed player statistics per match

[Source: https://pypi.org/project/sofascore-wrapper/, accessed: 2026-02-09]
[Source: https://github.com/tunjayoff/sofascore_scraper, accessed: 2026-02-09]
[Source: https://soccerdata.readthedocs.io/en/latest/reference/sofascore.html, accessed: 2026-02-09]

---

### 5. TransferMarkt (transfermarkt.us)

**Coverage:**
- xG, npxG, xA: NO
- SCA, GCA: NO
- Passes: NO
- Carries: NO
- Take-ons: NO
- Touches, tackles, blocks: NO
- PPDA/Pressing: NO

**Data Access:**
TransferMarkt focuses on market values, transfer history, contract details, and basic performance statistics (goals, assists, appearances) rather than advanced per-match metrics. Data can be accessed via `worldfootballR` R package, Apify scraper, or open datasets on GitHub with 93,000+ player profiles. Performance data is automatically calculated from match sheets but lacks granular per-match advanced statistics. [Source: https://www.transfermarkt.us/serie-a/startseite/wettbewerb/IT1, accessed: 2026-02-09]

**API/Scraping:** Scraping + third-party datasets

**Cost:** Free (via scraping/datasets)

**Serie A Coverage:** Current season (2025/2026) and historical data

**Per-Match Per-Player Granularity:** NO - primarily season totals and transfer data, not detailed match statistics

[Source: https://jaseziv.github.io/worldfootballR/articles/extract-transfermarkt-data.html, accessed: 2026-02-09]
[Source: https://github.com/salimt/football-datasets, accessed: 2026-02-09]

---

### 6. Football-data.co.uk

**Coverage:**
- xG, npxG, xA: NO
- SCA, GCA: NO
- Passes: NO
- Carries: NO
- Take-ons: NO
- Touches, tackles, blocks: NO
- PPDA/Pressing: NO

**Data Access:**
Football-data.co.uk provides CSV downloads of match results with betting odds, final/half-time scores, corners, and cards. The site has published data since 2001 with weekly updates via Travis-CI. Data covers the last 10 seasons of Serie A but focuses on match-level outcomes and betting markets rather than player statistics. [Source: https://football-data.co.uk/data.php, accessed: 2026-02-09]

**API/Scraping:** Direct CSV download

**Cost:** Free

**Serie A Coverage:** Last 10 seasons including current season

**Per-Match Per-Player Granularity:** NO - match outcomes and betting odds only, no player statistics

[Source: https://datahub.io/core/italian-serie-a, accessed: 2026-02-09]
[Source: https://github.com/probberechts/soccerdata, accessed: 2026-02-09]

---

### 7. StatsBomb Open Data (GitHub)

**Coverage:**
- xG, npxG, xA: YES - event-level granularity
- SCA, GCA: YES (likely - event data includes shot-creating actions)
- Passes: YES (pass attempts, completions, progressive passes)
- Carries: YES
- Take-ons: YES
- Touches, tackles, blocks: YES
- PPDA/Pressing: Can be calculated from event data

**Data Access:**
StatsBomb Open Data provides event-level JSON data from 3,000+ matches across 18 tournaments including Serie A. Data includes competitions.json, match files, and detailed event/lineup data with xG metrics (shot_statsbomb_xg, shot_statsbomb_xg2). Python (`statsbombpy`) and R (`StatsBombR`) libraries available for easy access. Free for public use in research projects and genuine football analytics interest. [Source: https://github.com/statsbomb/open-data, accessed: 2026-02-09]

**API/Scraping:** Official API + GitHub repository (JSON files)

**Cost:** Free (for research/non-commercial use)

**Serie A Coverage:** Selected seasons available (check competitions.json for specific coverage)

**Per-Match Per-Player Granularity:** YES - event-level data providing comprehensive per-match player statistics

[Source: https://github.com/statsbomb/statsbombpy, accessed: 2026-02-09]
[Source: https://medium.com/@lucascarrasquillaparra/complete-guide-on-working-with-the-statsbomb-open-data-dataset-a57c26d5852b, accessed: 2026-02-09]

**Note:** StatsBomb 360 data provides even more detailed spatial tracking for selected matches. Coverage is selective - not all Serie A seasons/matches may be available.

---

### 8. API-Football (api-football.com)

**Coverage:**
- xG, npxG, xA: Limited/Unknown in free tier
- SCA, GCA: NO
- Passes: YES (likely basic pass counts)
- Carries: NO
- Take-ons: NO
- Touches, tackles, blocks: YES (basic match statistics)
- PPDA/Pressing: NO

**Data Access:**
API-Football free tier provides 100 requests/day with access to all endpoints including countries, seasons, leagues, standings, teams, livescore, fixtures, events, lineups, top scorers, players, and statistics. Serie A has league ID 135. Statistics and predictions included in free tier, but limited to current season. No credit card required. [Source: https://www.api-football.com/, accessed: 2026-02-09]

**API/Scraping:** Official RESTful API

**Cost:** Free tier: 100 requests/day; Paid plans available for higher limits and historical data

**Serie A Coverage:** Current season in free tier; historical data requires paid plan

**Per-Match Per-Player Granularity:** YES for basic statistics; advanced metrics (xG) availability unclear in free tier

[Source: https://www.api-football.com/pricing, accessed: 2026-02-09]
[Source: https://medium.com/@bouabdallaoui.yassine/football-apis-made-easy-the-easiest-way-to-fetch-any-player-stats-318aa4146b1d, accessed: 2026-02-09]

---

### 9. Other Notable Mentions (Not Free)

**Opta Sports (Stats Perform):**
- Coverage: ALL 7 categories (xG, SCA, GCA, passes, carries, take-ons, touches/tackles/blocks, PPDA)
- Access: Commercial licensing required - no free access
- Former "Opta Playground" developer program discontinued (404 error)
- WhoScored uses Opta data, so scraping WhoScored provides indirect access
- [Source: https://www.statsperform.com/opta/, accessed: 2026-02-09]

**Wyscout (Hudl):**
- Coverage: ALL 7 categories with video integration
- Access: 15-day free trial for clubs/scouts; academic access through SoccerEDU partnerships
- Covers 600+ leagues with 5 years of historical data
- PPDA available (typically requires paid subscription)
- [Source: https://wyscout.com/, accessed: 2026-02-09]
- [Source: https://trybeem.com/blog/wyscout-free-trial/, accessed: 2026-02-09]

---

## Summary Matrix

| Source | xG/xA | SCA/GCA | Passes | Carries | Take-ons | Touches/Tackles | PPDA | Cost | Per-Match | API |
|--------|-------|---------|--------|---------|----------|-----------------|------|------|-----------|-----|
| Understat | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | Partial | Free | ✓ | Unofficial |
| FotMob | ✓ | ? | ✓ | ? | ? | ✓ | ? | Free | Limited | Unofficial |
| WhoScored | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ? | Free | ✓ | Scraping |
| Sofascore | ✓ | ? | ✓ | ? | ✓ | ✓ | ✗ | Free | ✓ | Unofficial |
| TransferMarkt | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Free | ✗ | Scraping |
| Football-data | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Free | ✗ | CSV |
| StatsBomb | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Calc | Free* | ✓ | Official |
| API-Football | ? | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | Free† | ✓ | Official |
| Opta | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Paid | ✓ | Commercial |
| Wyscout | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Trial/Paid | ✓ | Commercial |

*Free for research/non-commercial use, selective coverage
†Free tier limited to 100 requests/day and current season

---

## Recommended Sources for Serie A Pipeline

### Primary Sources (Free):
1. **Understat** - Best free source for xG, npxG, xA per match per player
2. **WhoScored** - Most comprehensive free source via scraping (Opta-powered)
3. **StatsBomb Open Data** - Highest quality when Serie A coverage available

### Secondary Sources (Free):
4. **Sofascore** - Good alternative for match statistics with unofficial API
5. **FotMob** - Supplementary source for match context

### Not Recommended for Advanced Stats:
- TransferMarkt (no advanced stats)
- Football-data.co.uk (match outcomes only)
- API-Football free tier (limited advanced metrics)

---

## Implementation Notes

**Current Pipeline Status:**
- Already using Understat for xG (confirmed working for all 20 Serie A teams)
- FBref blocked via 403 errors (no longer viable)
- player_xg_profiles.json has 634 players with xG data

**Gaps to Fill:**
- SCA, GCA: Must scrape WhoScored or use StatsBomb (if Serie A coverage exists)
- Progressive passes/carries: WhoScored scraping required
- PPDA: Can calculate from Understat team data or use paid sources

**Technical Approach:**
- Use `soccerdata` Python library for unified access to multiple sources
- Implement WhoScored scraping with rate limiting (30-60 sec/match)
- Check StatsBomb Open Data for Serie A season availability
- Cache all scraped data locally to minimize repeated requests

---

## Assumptions

- WhoScored provides SCA/GCA metrics (assumed based on Opta integration, not explicitly confirmed in search results)
- FotMob player detail API limitations may be bypassed with creative scraping approaches
- StatsBomb Open Data Serie A coverage varies by season (specific seasons not listed in search results)
- API-Football free tier xG availability unclear - may require testing to confirm

---

## Sources

1. [Understat Serie A xG Table 2025/2026](https://understat.com/league/Serie_A)
2. [understatapi PyPI](https://pypi.org/project/understatapi/)
3. [Understat Python Documentation](https://understat.readthedocs.io/en/latest/classes/understat.html)
4. [worldfootballR - Extract Understat Data](https://jaseziv.github.io/worldfootballR/articles/extract-understat-data.html)
5. [FotMob Serie A 2025/2026 Stats](https://www.fotmob.com/leagues/55/stats/serie)
6. [FotMob Data Extraction with Python](https://medium.com/@lmirandam07/fotmob-fixtures-data-extraction-with-python-f1ee03d2dfe)
7. [soccerdata - FotMob Documentation](https://soccerdata.readthedocs.io/en/latest/datasources/FotMob.html)
8. [WhoScored Serie A Player Statistics](https://www.whoscored.com/regions/108/tournaments/5/seasons/10732/stages/24500/playerstatistics/italy-serie-a-2025-2026)
9. [soccerdata - WhoScored Documentation](https://soccerdata.readthedocs.io/en/latest/datasources/WhoScored.html)
10. [Scrape WhoScored Live Scores](https://www.webharvy.com/blog/scrape-whoscored-live-scores/)
11. [WhoScored Scraper - ScrapeLead](https://scrapelead.io/news/whoscored-web-data-scraper/)
12. [Sofascore Serie A 2025/2026](https://www.sofascore.com/tournament/football/italy/serie-a/23)
13. [sofascore-wrapper PyPI](https://pypi.org/project/sofascore-wrapper/)
14. [SofaScore Scraper GitHub](https://github.com/tunjayoff/sofascore_scraper)
15. [soccerdata - Sofascore Documentation](https://soccerdata.readthedocs.io/en/latest/reference/sofascore.html)
16. [TransferMarkt Serie A](https://www.transfermarkt.us/serie-a/startseite/wettbewerb/IT1)
17. [worldfootballR - Extract Transfermarkt Data](https://jaseziv.github.io/worldfootballR/articles/extract-transfermarkt-data.html)
18. [Football Datasets GitHub](https://github.com/salimt/football-datasets)
19. [Football-data.co.uk Data Page](https://football-data.co.uk/data.php)
20. [Italian Serie A Dataset - DataHub](https://datahub.io/core/italian-serie-a)
21. [soccerdata - GitHub](https://github.com/probberechts/soccerdata)
22. [StatsBomb Open Data GitHub](https://github.com/statsbomb/open-data)
23. [statsbombpy GitHub](https://github.com/statsbomb/statsbombpy)
24. [Complete Guide to StatsBomb Open Data](https://medium.com/@lucascarrasquillaparra/complete-guide-on-working-with-the-statsbomb-open-data-dataset-a57c26d5852b)
25. [API-Football Official Site](https://www.api-football.com/)
26. [API-Football Pricing](https://www.api-football.com/pricing)
27. [Football APIs Made Easy](https://medium.com/@bouabdallaoui.yassine/football-apis-made-easy-the-easiest-way-to-fetch-any-player-stats-318aa4146b1d)
28. [Opta Data - Stats Perform](https://www.statsperform.com/opta/)
29. [Guide to Football/Soccer APIs](https://www.jokecamp.com/blog/guide-to-football-and-soccer-data-and-apis/)
30. [Wyscout Official Site](https://wyscout.com/)
31. [Wyscout Free Trial Guide](https://trybeem.com/blog/wyscout-free-trial/)
32. [Wyscout Features - SoccerEDU](https://www.socceredu.com/en-US/blog/wyscout)
33. [Football Stats: PPDA and Packing](https://medium.com/@buildingblocks/football-stats-ppda-and-packing-a750a0df18ef)
34. [Where to Get Free Football Data](https://mckayjohns.substack.com/p/where-to-get-free-football-data)
