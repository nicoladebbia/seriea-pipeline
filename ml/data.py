"""Data loading, feature tier selection, and time-series splitting."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from features.build import get_ml_feature_columns
from ml.config import (
    CLASS_LABELS,
    MATCH_DATE_COL,
    ODDS_COLUMN_PATTERNS,
    RESULT_COL,
    SEASON_COL,
    FeatureConfig,
    ValidationConfig,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature-category detection helpers
# ---------------------------------------------------------------------------

def _is_rolling_feature(col: str) -> bool:
    """True for rolling average features (e.g., home_roll_5_goals_scored)."""
    return "_roll_" in col

def _is_h2h_feature(col: str) -> bool:
    return col.startswith("h2h_")

def _is_elo_feature(col: str) -> bool:
    return "elo" in col.lower()

def _is_xg_trend_feature(col: str) -> bool:
    return "xg_" in col and ("trend" in col or "diff" in col or "over" in col or "under" in col)

def _is_referee_feature(col: str) -> bool:
    return col.startswith("ref_") or "referee" in col.lower()

def _is_odds_feature(col: str) -> bool:
    return any(col.startswith(p) or col.endswith(p.rstrip("_")) for p in ODDS_COLUMN_PATTERNS)

def _is_player_aggregate_feature(col: str) -> bool:
    """Team-aggregated player stats (e.g., home_agg_goals, away_agg_tackles)."""
    return "_agg_" in col

def _is_advanced_player_feature(col: str) -> bool:
    """Advanced player ratio/dependency features (adv_roll5_*)."""
    return "adv_roll5_adv_" in col

def _is_advanced_shot_feature(col: str) -> bool:
    """Advanced shot creation/finishing features (advshot_roll5_*)."""
    return "advshot_roll5_" in col

def _is_gk_feature(col: str) -> bool:
    return "_gk_" in col or "psxg" in col.lower()

def _is_shot_quality_feature(col: str) -> bool:
    return "_shot_" in col and "quality" not in col.lower() or "_avg_xg_per_shot" in col

def _is_manager_feature(col: str) -> bool:
    return "manager_tenure" in col or "manager_new" in col or "manager_changed" in col

def _is_market_feature(col: str) -> bool:
    return any(kw in col for kw in ("squad_value", "transfer_", "net_spend", "market_value"))

def _is_line_velocity_feature(col: str) -> bool:
    return col.startswith("line_vel_") or col == "steam_move_flag"

def _is_squad_value_feature(col: str) -> bool:
    return "squad_value" in col or "avg_player_value" in col

def _is_sofascore_feature(col: str) -> bool:
    return col.startswith(("home_ss_idx_", "away_ss_idx_", "ss_idx_diff_"))


# ---------------------------------------------------------------------------
# Domain-aware imputation
# ---------------------------------------------------------------------------

_H2H_DEFAULTS = {
    "h2h_matches_played": 0,
    "h2h_home_wins": 0,
    "h2h_away_wins": 0,
    "h2h_draws": 0,
    "h2h_home_goals_avg": 0.0,
    "h2h_away_goals_avg": 0.0,
    "h2h_home_win_rate": 1 / 3,  # uninformative prior
    "h2h_last_result": 0,  # neutral (draw-like)
}


def _smart_impute(df: pd.DataFrame, cols: list[str], fit_mask: pd.Series | None = None) -> pd.DataFrame:
    """Impute NaN values using category-aware strategies.

    Called BEFORE the NaN-threshold filter so more features can qualify
    as universal.  Strategies by feature category:
      - H2H:             domain defaults (0 counts, 1/3 rates)
      - Rolling stats:   0.0  (season-start = no history, neutral)
      - Elo:             forward-fill within team, else 1500 (league mean)
      - xG trends:       0.0  (no over/under-performance = neutral)
      - Referee:         per-column median (league-average tendency)
      - Player agg / GK / Shot quality: per-season median
      - Manager:         0 tenure, 0 flags (unknown = no data)
      - Market/transfer: per-season median
      - Odds:            NOT imputed (genuinely missing for some seasons)
      - Everything else: 0.0  (safe neutral for tree-based models)
    """
    X = df.copy()

    # Filter to numeric columns only — object columns (team names, etc.) can't be imputed
    cols = [c for c in cols if X[c].dtype.kind in ('f', 'i', 'u', 'b')]

    # Pre-compute per-season medians for features that need them.
    # When fit_mask is provided, compute stats only from fitting (train) rows
    # to prevent test-set information from leaking into imputation.
    fit_df = X[fit_mask] if fit_mask is not None else X
    season_col = SEASON_COL if SEASON_COL in X.columns else None
    if season_col:
        season_medians = fit_df.groupby(season_col)[cols].median()
    else:
        season_medians = None

    for col in cols:
        if not X[col].isna().any():
            continue

        # Skip odds — genuinely missing for some seasons
        if _is_odds_feature(col):
            continue

        if _is_h2h_feature(col):
            X[col] = X[col].fillna(_H2H_DEFAULTS.get(col, 0.0))

        elif _is_elo_feature(col):
            # Forward-fill within the column, then default to league mean
            X[col] = X[col].ffill().fillna(1500.0)

        elif _is_xg_trend_feature(col):
            X[col] = X[col].fillna(0.0)

        elif _is_rolling_feature(col):
            X[col] = X[col].fillna(0.0)

        elif _is_referee_feature(col):
            X[col] = X[col].fillna(fit_df[col].median())

        elif _is_manager_feature(col):
            X[col] = X[col].fillna(0.0)

        elif _is_advanced_player_feature(col) or _is_advanced_shot_feature(col):
            # Advanced ratio features: per-season median (only exist for FBref seasons)
            if season_medians is not None and col in season_medians.columns:
                for season_val in X[season_col].dropna().unique():
                    mask = (X[season_col] == season_val) & X[col].isna()
                    if mask.any():
                        med = season_medians.loc[season_val, col] if season_val in season_medians.index else np.nan
                        if pd.notna(med):
                            X.loc[mask, col] = med
            if X[col].isna().any():
                X[col] = X[col].fillna(fit_df[col].median())

        elif _is_player_aggregate_feature(col) or _is_gk_feature(col) or _is_shot_quality_feature(col):
            # Per-season median: these features exist only for FBref seasons
            if season_medians is not None and col in season_medians.columns:
                for season_val in X[season_col].dropna().unique():
                    mask = (X[season_col] == season_val) & X[col].isna()
                    if mask.any():
                        med = season_medians.loc[season_val, col] if season_val in season_medians.index else np.nan
                        if pd.notna(med):
                            X.loc[mask, col] = med
            # Remaining NaN (seasons with no data at all): global median from fit data
            if X[col].isna().any():
                X[col] = X[col].fillna(fit_df[col].median())

        elif _is_market_feature(col):
            if season_medians is not None and col in season_medians.columns:
                for season_val in X[season_col].dropna().unique():
                    mask = (X[season_col] == season_val) & X[col].isna()
                    if mask.any():
                        med = season_medians.loc[season_val, col] if season_val in season_medians.index else np.nan
                        if pd.notna(med):
                            X.loc[mask, col] = med
            if X[col].isna().any():
                X[col] = X[col].fillna(fit_df[col].median())

        elif _is_sofascore_feature(col):
            # Sofascore indices are rank-normalized [0,1]. 0.0 = worst in dataset
            # which is a false signal for missing data. Use 0.5 (midpoint) instead.
            X[col] = X[col].fillna(0.5)

        else:
            # Generic fallback: 0.0 (neutral for tree-based models)
            X[col] = X[col].fillna(0.0)

    return X


def _add_availability_flags(df: pd.DataFrame, ml_cols: list[str]) -> list[str]:
    """Add binary indicator columns for feature-group availability.

    Returns the list of new indicator column names added.
    """
    indicators: list[str] = []

    # FBref player aggregate data
    agg_cols = [c for c in ml_cols if _is_player_aggregate_feature(c)]
    if agg_cols:
        df["_has_player_agg"] = (~df[agg_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_player_agg")

    # GK quality data
    gk_cols = [c for c in ml_cols if _is_gk_feature(c)]
    if gk_cols:
        df["_has_gk_data"] = (~df[gk_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_gk_data")

    # Shot quality data
    shot_cols = [c for c in ml_cols if _is_shot_quality_feature(c)]
    if shot_cols:
        df["_has_shot_data"] = (~df[shot_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_shot_data")

    # Advanced player ratio features
    adv_cols = [c for c in ml_cols if _is_advanced_player_feature(c)]
    if adv_cols:
        df["_has_adv_player"] = (~df[adv_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_adv_player")

    # Advanced shot features
    advshot_cols = [c for c in ml_cols if _is_advanced_shot_feature(c)]
    if advshot_cols:
        df["_has_adv_shots"] = (~df[advshot_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_adv_shots")

    # Betting odds
    odds_cols = [c for c in ml_cols if _is_odds_feature(c)]
    if odds_cols:
        df["_has_odds"] = (~df[odds_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_odds")

    # Line movement / velocity data (34% NaN in 2025-26)
    line_cols = [c for c in ml_cols if _is_line_velocity_feature(c)]
    if line_cols:
        df["_has_line_movement"] = (~df[line_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_line_movement")

    # Squad value data (19.7% NaN in 2025-26)
    sv_cols = [c for c in ml_cols if _is_squad_value_feature(c)]
    if sv_cols:
        df["_has_squad_value"] = (~df[sv_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_squad_value")

    # Sofascore indices (100% NaN pre-2022, ~1.5% after)
    ss_cols = [c for c in ml_cols if _is_sofascore_feature(c)]
    if ss_cols:
        df["_has_sofascore"] = (~df[ss_cols].isna().all(axis=1)).astype(np.int8)
        indicators.append("_has_sofascore")

    return indicators


def _latest_season(seasons: pd.Series) -> str:
    """Return the latest season from a Series of season strings."""
    return sorted(seasons.unique())[-1]


class DataLoader:
    """Load features.parquet and partition features into tiers."""

    def __init__(self, features_path: str):
        self.features_path = features_path
        self.df: pd.DataFrame | None = None
        self.universal_features: list[str] = []
        self.all_ml_features: list[str] = []

    def load(self) -> pd.DataFrame:
        self.df = pd.read_parquet(self.features_path)
        ml_cols = get_ml_feature_columns(self.df)

        cfg = FeatureConfig()

        # --- NaN tier report (diagnostic) ---
        nan_pcts = self.df[ml_cols].isna().mean()
        # Save pre-imputation NaN rates for per-league filtering later.
        # After imputation, 100%-NaN features become 100%-zero and escape the
        # NaN threshold filter. This lets get_universal_dataset() catch them.
        self._pre_imputation_nan = nan_pcts.copy()
        tier1 = (nan_pcts < 0.10).sum()
        tier2 = ((nan_pcts >= 0.10) & (nan_pcts < 0.30)).sum()
        tier3 = ((nan_pcts >= 0.30) & (nan_pcts < 0.60)).sum()
        tier4 = (nan_pcts >= 0.60).sum()
        log.info(
            "NaN tiers BEFORE imputation — Tier1(<10%%): %d, Tier2(10-30%%): %d, "
            "Tier3(30-60%%): %d, Tier4(>60%%): %d",
            tier1, tier2, tier3, tier4,
        )

        # --- Add availability indicator flags BEFORE imputation ---
        indicator_cols = _add_availability_flags(self.df, ml_cols)
        if indicator_cols:
            log.info("Added %d availability indicator features: %s", len(indicator_cols), indicator_cols)

        # --- Smart imputation BEFORE NaN threshold filter ---
        self.df = _smart_impute(self.df, ml_cols)

        # Recalculate NaN rates after imputation
        nan_pcts_after = self.df[ml_cols].isna().mean()
        tier1_after = (nan_pcts_after < 0.10).sum()
        tier2_after = ((nan_pcts_after >= 0.10) & (nan_pcts_after < 0.30)).sum()
        log.info(
            "NaN tiers AFTER imputation — Tier1(<10%%): %d (+%d), Tier2(10-30%%): %d",
            tier1_after, tier1_after - tier1, tier2_after,
        )

        # Include indicator columns in the ML feature set
        all_ml_with_indicators = sorted(ml_cols + indicator_cols)

        # Apply NaN threshold on POST-imputation rates
        nan_pcts_final = self.df[all_ml_with_indicators].isna().mean()
        self.universal_features = sorted(
            nan_pcts_final[nan_pcts_final < cfg.universal_nan_threshold].index.tolist()
        )
        self.all_ml_features = sorted(all_ml_with_indicators)

        log.info(
            "Loaded %d matches, %d universal features, %d total ML features",
            len(self.df), len(self.universal_features), len(self.all_ml_features),
        )
        return self.df

    def get_universal_dataset(
        self, league: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
        """Return (X, y, feature_names) using universal features only.

        Args:
            league: If specified, filter rows to this league only (e.g.
                "serie_a", "premier_league"). None = all leagues (backward
                compatible -- existing Serie A behavior unchanged since
                min_train_season already filters).
        """
        if self.df is None:
            self.load()

        mask = self.df[RESULT_COL].isin(CLASS_LABELS)

        # League filter: restrict to a single league when requested
        if league is not None:
            if "league" not in self.df.columns:
                log.warning("League filter requested (%s) but no 'league' column in data", league)
            else:
                league_mask = self.df["league"] == league
                n_league = league_mask.sum()
                if n_league == 0:
                    raise ValueError(
                        f"No rows found for league='{league}'. "
                        f"Available leagues: {sorted(self.df['league'].unique())}"
                    )
                mask = mask & league_mask
                log.info("League filter=%s: %d matches selected", league, n_league)

        # Apply min_train_season filter: drop pre-2017 data to unlock advanced
        # features (xG, pressing, odds velocity, lineup) that have zero coverage
        # in early seasons. With time-decay=0.85, these old matches contribute <4%
        # weight anyway.
        from ml.config import ValidationConfig
        val_cfg = ValidationConfig()
        if val_cfg.min_train_season:
            season_mask = self.df[SEASON_COL] >= val_cfg.min_train_season
            mask = mask & season_mask
            n_dropped = (~season_mask).sum()
            if n_dropped > 0:
                log.info("min_train_season=%s: dropped %d matches before cutoff, %d remaining",
                         val_cfg.min_train_season, n_dropped, mask.sum())

        # Recompute universal features on the FILTERED dataset (league + season)
        # This is critical: features that are 70% NaN on all-data (2005-2026, all leagues)
        # may be only 5% NaN on the filtered training set (e.g., EPL 2017+).
        #
        # ALSO check pre-imputation NaN rates: features that were 100% NaN for this
        # league subset get imputed to 0.0 and escape the post-imputation threshold.
        # Example: away_squad_disruption is 100% NaN for all EPL → imputed to 0 → looks
        # fine post-imputation but carries zero signal. We use the raw NaN rates to catch these.
        from ml.config import FeatureConfig
        feat_cfg = FeatureConfig()
        filtered_df = self.df.loc[mask]
        filtered_nan = filtered_df[self.all_ml_features].isna().mean()

        # Pre-imputation check: recompute raw NaN rates on the league-filtered subset
        # using the saved pre-imputation snapshot. Features that were >95% NaN before
        # imputation for THIS subset are dead signal and should be excluded.
        if hasattr(self, '_pre_imputation_nan') and league is not None:
            raw_df = pd.read_parquet(self.features_path)
            raw_filtered = raw_df.loc[mask]
            avail_cols = [c for c in self.all_ml_features if c in raw_filtered.columns]
            raw_nan = raw_filtered[avail_cols].isna().mean()
            dead_features = set(raw_nan[raw_nan > 0.95].index)
            if dead_features:
                log.warning("Pre-imputation check: %d features are >95%% NaN for league=%s "
                            "(imputed to 0 but carry no signal): %s",
                            len(dead_features), league,
                            sorted(dead_features)[:10])
                # Exclude dead features from the filtered universal set
                filtered_nan = filtered_nan.drop(labels=list(dead_features & set(filtered_nan.index)),
                                                  errors="ignore")

        filtered_universal = sorted(
            filtered_nan[filtered_nan < feat_cfg.universal_nan_threshold].index.tolist()
        )
        n_extra = len(filtered_universal) - len(self.universal_features)
        if n_extra > 0:
            log.info(
                "Recomputed NaN threshold on filtered data: %d features (was %d, +%d rescued)",
                len(filtered_universal), len(self.universal_features), n_extra,
            )
        features_to_use = filtered_universal if filtered_universal else self.universal_features

        X = self.df.loc[mask, features_to_use].copy().reset_index(drop=True)
        y = self.df.loc[mask, RESULT_COL].copy().reset_index(drop=True)
        seasons = self.df.loc[mask, SEASON_COL].copy().reset_index(drop=True)

        # Fill remaining NaN — let tree models handle odds NaN natively
        for col in features_to_use:
            if X[col].isna().any():
                if col in _H2H_DEFAULTS:
                    X[col] = X[col].fillna(_H2H_DEFAULTS[col])
                elif _is_odds_feature(col) or _is_line_velocity_feature(col):
                    pass  # XGBoost/LightGBM/CatBoost handle NaN natively
                else:
                    X[col] = X[col].fillna(0.0)

        # Attach season and league as metadata (not features)
        X["_season"] = seasons.values
        X["_match_date"] = pd.to_datetime(
            self.df.loc[mask, MATCH_DATE_COL], errors="coerce"
        ).values
        if "league" in self.df.columns:
            X["_league"] = self.df.loc[mask, "league"].copy().reset_index(drop=True).values
        else:
            X["_league"] = "serie_a"

        log.info("NaN remaining after imputation: %d", X[features_to_use].isna().sum().sum())
        return X, y, features_to_use

    def get_rich_dataset(
        self, season: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
        """Return (X, y, feature_names) using all features for one season.

        If season is None, uses the latest season in the data.
        """
        if self.df is None:
            self.load()

        if season is None:
            season = _latest_season(self.df[SEASON_COL])

        season_df = self.df[self.df[SEASON_COL] == season].copy()
        mask = season_df[RESULT_COL].isin(CLASS_LABELS)

        X = season_df.loc[mask, self.all_ml_features].copy().reset_index(drop=True)
        y = season_df.loc[mask, RESULT_COL].copy().reset_index(drop=True)

        # Drop columns that are entirely NaN for this season
        all_nan = X.columns[X.isna().all()]
        if len(all_nan) > 0:
            X = X.drop(columns=all_nan)
            features = [c for c in self.all_ml_features if c not in set(all_nan)]
        else:
            features = list(self.all_ml_features)

        X["_season"] = season
        X["_match_date"] = pd.to_datetime(
            season_df.loc[mask, MATCH_DATE_COL], errors="coerce"
        ).values
        if "league" in season_df.columns:
            X["_league"] = season_df.loc[mask, "league"].copy().reset_index(drop=True).values
        else:
            X["_league"] = "serie_a"

        return X, y, features


class TimeSeriesSplitter:
    """Generate walk-forward (expanding window) train/test splits by season."""

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    def generate_splits(
        self, seasons: pd.Series,
    ) -> list[Tuple[list[str], list[str]]]:
        """Return list of (train_seasons, test_seasons) tuples."""
        ordered = sorted(seasons.unique())
        counts = seasons.value_counts()
        min_test = getattr(self.config, "min_test_matches", 0)
        splits = []
        dropped: list[tuple[list[str], int]] = []

        for i in range(self.config.min_train_seasons, len(ordered)):
            if self.config.expanding_window:
                train_seasons = ordered[:i]
            else:
                train_seasons = ordered[
                    max(0, i - self.config.min_train_seasons) : i
                ]
            test_seasons = ordered[i : i + self.config.test_season_size]

            if not (train_seasons and test_seasons):
                continue

            # A season becomes a fold on its first played match. Ten matches
            # cannot measure a model, but every caller averages folds
            # unweighted, so those ten would carry a full fold's vote.
            n_test = int(counts.reindex(test_seasons).fillna(0).sum())
            if min_test and n_test < min_test:
                dropped.append((list(test_seasons), n_test))
                continue

            splits.append((list(train_seasons), list(test_seasons)))

        if dropped and splits:
            log.info(
                "Walk-forward: dropped %d fold(s) under min_test_matches=%d: %s",
                len(dropped), min_test,
                ", ".join(f"{t} (n={n})" for t, n in dropped),
            )
        elif dropped and not splits:
            # Never hand back an empty split list — that aborts training
            # outright, which is a worse failure than a noisy fold.
            log.error(
                "Every walk-forward fold is under min_test_matches=%d (largest "
                "n=%d). Falling back to the unfiltered splits; any metric from "
                "this run is unmeasured.",
                min_test, max(n for _t, n in dropped),
            )
            splits = [(list(t), list(te)) for t, te in (
                (ordered[:i] if self.config.expanding_window
                 else ordered[max(0, i - self.config.min_train_seasons):i],
                 ordered[i : i + self.config.test_season_size])
                for i in range(self.config.min_train_seasons, len(ordered))
            ) if t and te]

        return splits
