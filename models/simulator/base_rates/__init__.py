"""Rate estimators: per-team Poisson rates for shots, SOT, corners, cards.

Each estimator exposes the same interface:

    class Estimator:
        def fit(self, train_df: pd.DataFrame) -> None: ...
        def predict(self, eval_df: pd.DataFrame) -> (np.ndarray, np.ndarray)
            # returns (home_rate, away_rate)

They consume `features_serie_a.parquet` and are consumed by `engine/simulator.py`
to populate the corresponding MatchSimulation arrays.
"""
