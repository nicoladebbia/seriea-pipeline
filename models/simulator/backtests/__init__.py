"""Historical P&L backtest harness (Phase 3b).

Entry point: `BacktestHarness` in harness.py.

Every predictor wraps the same interface:

    class Predictor(Protocol):
        name: str
        version: str
        def predict(self, df: pd.DataFrame, market: str) -> np.ndarray: ...

The harness walk-forwards by season, applies the predictor, resolves odds via
the fallback chain, applies stake policies, and emits a per-market,
per-threshold, per-source, per-stake-policy report with bootstrap CIs.

No live betting, no API calls. Read-only over historical data.
"""
