"""Force a real Telegram send of the 'Lineups Confirmed' card against current data.

Use this to verify that enrichment (shirt numbers, G/A, xG, match preview)
renders correctly before an actual lineup arrives live.

Usage:
    python3 -m scripts.diagnostics.fire_lineup_card          # real send
    python3 -m scripts.diagnostics.fire_lineup_card --dry    # print HTML only, no Telegram

Effects:
    - Deletes data/.lineups_dedup.json so the send is not suppressed.
    - Sends one Telegram message to your configured bot chat.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true",
                        help="Print the Telegram HTML instead of sending")
    args = parser.parse_args()

    from config.settings import DATA_DIR
    dedup = DATA_DIR / ".lineups_dedup.json"
    if dedup.exists():
        dedup.unlink()
        print(f"cleared: {dedup}")

    from scripts.pipeline import notify as N

    if args.dry:
        # Monkey-patch notify() to capture HTML instead of sending
        captured: dict = {}
        orig = N.notify

        def _capture(message, title, level, category,
                     priority="", tg_html="", tg_reply_markup=None):
            captured["tg_html"] = tg_html
            captured["message"] = message
            captured["title"] = title
            return {"captured": True}

        N.notify = _capture
        try:
            result = N.notify_lineups_confirmed()
        finally:
            N.notify = orig

        html = captured.get("tg_html", "")
        if not html:
            print("No HTML produced. Possible reasons:")
            print("  - No confirmed lineups in data/upcoming/confirmed_lineups.json")
            print("  - All matches have <11 players on either side")
            return 1
        print("\n" + "=" * 75)
        print("Telegram HTML (would be sent):")
        print("=" * 75)
        print(html)
        print("=" * 75)
        print(f"result: {result}")
        return 0

    # Real send
    result = N.notify_lineups_confirmed()
    print(f"fired: {result}")
    if not result:
        print("No lineups met the confirmation threshold (>=11 per side).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
