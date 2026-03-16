#!/bin/bash
# Scrape FBref match reports for seasons 2017-2024
# (2024-2025 already done)
#
# Expected runtime: ~4-5 hours at 6s delay
# Safe to leave running overnight
#
# Progress is logged to scrape_all.log
# Can be safely interrupted and restarted (registry tracks downloads)

set -e
cd "$(dirname "$0")"

LOG="scrape_all.log"
echo "=== Scrape started at $(date) ===" | tee -a "$LOG"

SEASONS=(
    "2017-2018"
    "2018-2019"
    "2019-2020"
    "2020-2021"
    "2021-2022"
    "2022-2023"
    "2023-2024"
)

for season in "${SEASONS[@]}"; do
    echo "" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"
    echo "=== SEASON: $season — $(date) ===" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"

    echo "--- Scraping fixtures for $season ---" | tee -a "$LOG"
    python3 cli.py scrape-fixtures --season "$season" 2>&1 | tee -a "$LOG"

    echo "--- Downloading match reports for $season ---" | tee -a "$LOG"
    python3 cli.py scrape-matches --season "$season" 2>&1 | tee -a "$LOG"

    echo "=== $season DONE at $(date) ===" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "=== ALL SCRAPING DONE at $(date) ===" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# Re-parse everything to rebuild player_stats.parquet and shots.parquet
echo "" | tee -a "$LOG"
echo "--- Parsing all downloaded matches ---" | tee -a "$LOG"
python3 cli.py parse 2>&1 | tee -a "$LOG"

# Rebuild features with the new data
echo "" | tee -a "$LOG"
echo "--- Rebuilding feature table ---" | tee -a "$LOG"
python3 cli.py features 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== FULL PIPELINE COMPLETE at $(date) ===" | tee -a "$LOG"
