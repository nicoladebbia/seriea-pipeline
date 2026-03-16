# Decision Trees

Read this before modifying code in the pipeline.

## "Should I modify this code?"
```
Is it in scripts/ensemble_prediction_engine.py?
  → YES: High risk. Backup first. Run backtest BEFORE and AFTER. Reject if LL regresses > 0.002.
  → NO: Is it in features/*.py?
    → YES: Verify shift(1) for leakage. Rebuild features.parquet. Check coverage > 80%.
    → NO: Is it in scripts/ultimate_betting_system.py?
      → YES: Verify edge caps (8% DC/O/U, 20% draws). Run backtest_betting_system.py.
      → NO: Standard change. Run pytest after.
```

## Anti-Patterns (NEVER DO THESE)
- **DO NOT** use future data in features — always `shift(1)`
- **DO NOT** ship model changes without backtest confirmation
- **DO NOT** trust edges > 8% — model overconfidence (adverse selection)
- **DO NOT** modify `.env` or `config/api_keys.json` programmatically
- **DO NOT** run `pip install` — use existing virtualenv
- **DO NOT** add Cards/Corners bets — heuristic Poisson, no real odds, zero provable edge
- **DO NOT** inflate DC fallback odds — was Bug B2, must stay at `* 1.00`
- **DO NOT** change `ENSEMBLE_WEIGHTS` without also updating `ENSEMBLE_WEIGHTS_WITH_DEEP`
