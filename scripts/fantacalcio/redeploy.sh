#!/bin/zsh
# Fanta redeploy -- the single path from a code change to the live app.
# Nicola 2026-09-03: "every time you change the system, it updates
# automatically on the app." Triggered by .git/hooks/post-commit for any
# commit touching the fanta engine, the bot, the scheduler, or web/app.py
# (hook is unversioned; see memory fanta-max-automation). Manual run is fine.
#
# Order matters: tests GATE the rebuild (deep-test before activating) --
# red tests mean the artifacts and services keep serving the last good code.
set -u -o pipefail
ROOT="/Users/nicoladebbia/Projects/seriea-pipeline"
LOG="$ROOT/logs/fanta_redeploy.log"
cd "$ROOT" || exit 1
{
  echo "=== redeploy $(date '+%F %T') trigger=${1:-manual} ${2:-}"
  if ! python3 -m pytest tests/test_fanta_tracker.py -q 2>&1 | tail -2; then
    echo "TESTS RED -- artifacts NOT rebuilt, services NOT restarted"
    python3 -c "
from scripts.pipeline.notify import notify
notify('Fanta redeploy BLOCCATO: test rossi - il sistema serve ancora il codice precedente',
       title='Fanta redeploy', level='warning', category='system')" || true
    exit 1
  fi
  # normal build path: tracker + advice + rivals; the XI push stays
  # change-gated inside, so an unchanged recommendation redeploys silently
  python3 -m scripts.fantacalcio.tracker 2>&1 | tail -4
  /bin/launchctl kickstart -k gui/501/com.seriea-pipeline.telegram-bot \
    && echo "telegram-bot restarted"
  if [[ "${2:-}" == "--web" ]]; then
    /bin/launchctl kickstart -k gui/501/com.seriea-pipeline.web-dashboard \
      && echo "web-dashboard restarted"
  fi
  python3 -c "
import json, datetime
d = json.load(open('data/fantacalcio/xi_advice.json'))
print('advice:', d['module'], 'exp', d['exp_total'], 'gen', d['generated_at'])"
  echo "=== done $(date '+%F %T')"
} >> "$LOG" 2>&1
