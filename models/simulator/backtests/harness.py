"""BacktestHarness — walk-forward, per-market, per-threshold, per-source harness.

Core loop:

    for season in seasons:
        train = df[df.season < season]
        eval  = df[df.season == season]
        predictor.fit(train)              # only if predictor is re-trainable
        for match in eval:
            for market in markets:
                p = predictor.predict(match, market)
                for src_chain in market.odds_chain:
                    odds = resolve_odds(match, src_chain)
                    for thresh in edge_thresholds_pct:
                        if edge(p, odds) >= thresh:
                            for policy in stake_policies:
                                stake = policy.stake(...)
                                profit = settle(actual_outcome, stake, odds)
                                record(market, thresh, policy, src, stake, profit)

At end, compute ROIStats (bootstrap CI, Sharpe, DD, streak) per
(market, threshold, stake_policy, odds_source). Emit a BacktestReport.

Walk-forward granularity: season boundary. Per-match walk-forward requires
thousands of retrains; overkill unless intra-season drift is observed.

Predictor contract (duck-typed):

    class Predictor:
        name: str
        version: str
        markets: list[str]                  # which markets it can predict
        def fit(self, train_df): ...        # optional; may be no-op
        def predict_binary(self, eval_df, market) -> np.ndarray
        def predict_multiclass(self, eval_df, market) -> np.ndarray  # shape (n, K)
        def classes_for(self, market) -> tuple[str, ...]             # multiclass only
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from .odds_fallback import (
    BinaryOdds,
    MulticlassOdds,
    NoOddsAvailable,
    deoverround_binary,
    deoverround_multiclass,
    resolve_odds_binary,
    resolve_odds_multiclass,
)
from .report import BacktestReport, BacktestRunMetadata
from .roi_bootstrap import ROIStats, compute_roi_stats
from .stake_policies import BetInput, DEFAULT_POLICIES, StakePolicy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market definitions — shared across predictors. One schema per market.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryMarket:
    label: str
    target_column: str                         # actual outcome col in df (0/1 or bool)
    odds_chain: tuple[tuple[str, str], ...]    # [(col_yes, col_no), ...] priority order
    kind: str = "binary"


@dataclass(frozen=True)
class MulticlassMarket:
    label: str
    target_column: str
    classes: tuple[str, ...]
    odds_chain: tuple[tuple[str, ...], ...]
    kind: str = "multiclass"


# ---------------------------------------------------------------------------
# Predictor contract
# ---------------------------------------------------------------------------

class Predictor(Protocol):
    name: str
    version: str

    def fit(self, train_df: pd.DataFrame) -> None: ...

    def supports(self, market_label: str) -> bool: ...

    def predict_binary(self, eval_df: pd.DataFrame, market_label: str) -> np.ndarray: ...

    def predict_multiclass(self, eval_df: pd.DataFrame, market_label: str) -> np.ndarray: ...

    def classes_for(self, market_label: str) -> tuple[str, ...]: ...


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@dataclass
class _BetRecord:
    match_id: str
    market: str
    threshold_pct: float
    stake_policy: str
    odds_source: str
    stake_eur: float
    profit_eur: float
    won: bool
    p_predicted: float
    fair_prob: float
    edge_pct: float
    odds: float


class BacktestHarness:
    def __init__(
        self,
        df: pd.DataFrame,
        binary_markets: list[BinaryMarket],
        multiclass_markets: list[MulticlassMarket],
        edge_thresholds_pct: list[float],
        stake_policies: list[StakePolicy] = DEFAULT_POLICIES,
        bankroll_eur: float = 1000.0,
        season_col: str = "season",
        league_col: str = "league",
        league_filter: str = "serie_a",
        seed_salt: int = 0,
    ):
        self.df = df.copy()
        self.binary_markets = {m.label: m for m in binary_markets}
        self.multiclass_markets = {m.label: m for m in multiclass_markets}
        self.edge_thresholds_pct = sorted(edge_thresholds_pct)
        self.stake_policies = stake_policies
        self.bankroll_eur = bankroll_eur
        self.season_col = season_col
        self.league_col = league_col
        self.league_filter = league_filter
        self.seed_salt = seed_salt

    # ----- Target precomputation --------------------------------------------

    def _ensure_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute target columns referenced by markets if not already present."""
        df = df.copy()
        if "home_score" in df.columns and "away_score" in df.columns:
            df["total_goals"] = df["home_score"] + df["away_score"]
            for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
                col = f"over_{str(line).replace('.', '_')}"
                if col not in df.columns:
                    df[col] = (df["total_goals"] > line).astype("float")
            if "btts" not in df.columns:
                df["btts"] = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype("float")
            if "home_clean_sheet" not in df.columns:
                df["home_clean_sheet"] = (df["away_score"] == 0).astype("float")
            if "away_clean_sheet" not in df.columns:
                df["away_clean_sheet"] = (df["home_score"] == 0).astype("float")
        if "home_corners" in df.columns and "away_corners" in df.columns:
            df["total_corners"] = df["home_corners"] + df["away_corners"]
            for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
                col = f"corners_over_{str(line).replace('.', '_')}"
                if col not in df.columns:
                    df[col] = (df["total_corners"] > line).astype("float")
        if "home_yellow_cards" in df.columns and "away_yellow_cards" in df.columns:
            df["total_cards"] = (
                df["home_yellow_cards"].fillna(0) + df["away_yellow_cards"].fillna(0)
                + df.get("home_red_cards", pd.Series(0, index=df.index)).fillna(0)
                + df.get("away_red_cards", pd.Series(0, index=df.index)).fillna(0)
            )
            for line in [2.5, 3.5, 4.5, 5.5]:
                col = f"cards_over_{str(line).replace('.', '_')}"
                if col not in df.columns:
                    df[col] = (df["total_cards"] > line).astype("float")
        return df

    # ----- Main loop --------------------------------------------------------

    def run(self, predictor: Predictor, seasons: list[str]) -> BacktestReport:
        df = self.df[self.df[self.league_col] == self.league_filter].copy()
        df = self._ensure_targets(df)

        records: list[_BetRecord] = []
        all_sources: set[str] = set()
        n_scored = 0

        for season in seasons:
            train = df[df[self.season_col] < season].copy()
            eval_df = df[df[self.season_col] == season].copy()
            if len(eval_df) == 0:
                log.warning("No rows for season=%s — skipping", season)
                continue
            log.info("Season %s: train=%d, eval=%d", season, len(train), len(eval_df))
            try:
                predictor.fit(train)
            except NotImplementedError:
                pass  # stateless predictor

            # BINARY MARKETS
            for mk_label, mk in self.binary_markets.items():
                if not predictor.supports(mk_label):
                    continue
                if mk.target_column not in eval_df.columns or eval_df[mk.target_column].isna().all():
                    continue

                sub = eval_df.dropna(subset=[mk.target_column]).copy()
                if len(sub) == 0:
                    continue

                probs = predictor.predict_binary(sub, mk_label)
                if probs is None or len(probs) == 0:
                    continue

                for i in range(len(sub)):
                    row = sub.iloc[i]
                    try:
                        odds = resolve_odds_binary(row, list(mk.odds_chain))
                    except NoOddsAvailable:
                        continue
                    all_sources.add(odds.source)
                    fair_yes, fair_no = deoverround_binary(odds)
                    p_yes = float(probs[i])
                    target = int(row[mk.target_column])
                    edge_yes = (p_yes - fair_yes) / fair_yes * 100.0
                    edge_no = ((1 - p_yes) - fair_no) / fair_no * 100.0
                    # Pick side with higher edge
                    if edge_yes >= edge_no:
                        side = "yes"; p = p_yes; odds_u = odds.yes
                        fair_u = fair_yes; edge_pct = edge_yes; won = target == 1
                    else:
                        side = "no"; p = 1 - p_yes; odds_u = odds.no
                        fair_u = fair_no; edge_pct = edge_no; won = target == 0

                    for thresh in self.edge_thresholds_pct:
                        if edge_pct < thresh:
                            continue
                        for policy in self.stake_policies:
                            stake = policy.stake(
                                BetInput(p=p, odds=odds_u, edge_pct=edge_pct),
                                self.bankroll_eur,
                            )
                            profit = ((odds_u - 1.0) * stake) if won else (-stake)
                            records.append(_BetRecord(
                                match_id=str(row.get("match_id", f"row_{i}")),
                                market=mk_label,
                                threshold_pct=thresh,
                                stake_policy=policy.label(),
                                odds_source=odds.source,
                                stake_eur=stake,
                                profit_eur=profit,
                                won=won,
                                p_predicted=p,
                                fair_prob=fair_u,
                                edge_pct=edge_pct,
                                odds=odds_u,
                            ))
                n_scored += len(sub)

            # MULTICLASS MARKETS
            for mk_label, mk in self.multiclass_markets.items():
                if not predictor.supports(mk_label):
                    continue
                if mk.target_column not in eval_df.columns or eval_df[mk.target_column].isna().all():
                    continue
                sub = eval_df.dropna(subset=[mk.target_column]).copy()
                if len(sub) == 0:
                    continue
                probs = predictor.predict_multiclass(sub, mk_label)
                if probs is None or len(probs) == 0:
                    continue
                classes = predictor.classes_for(mk_label)

                for i in range(len(sub)):
                    row = sub.iloc[i]
                    target = str(row[mk.target_column])
                    if target not in classes:
                        continue
                    try:
                        odds = resolve_odds_multiclass(row, list(mk.odds_chain), classes)
                    except NoOddsAvailable:
                        continue
                    all_sources.add(odds.source)
                    fair = deoverround_multiclass(odds)
                    best_idx, best_edge = 0, -1e9
                    for ci in range(len(classes)):
                        e = (probs[i, ci] - fair[ci]) / fair[ci] * 100.0
                        if e > best_edge:
                            best_edge = e
                            best_idx = ci
                    p = float(probs[i, best_idx])
                    odds_u = odds.odds[best_idx]
                    fair_u = fair[best_idx]
                    won = classes[best_idx] == target

                    for thresh in self.edge_thresholds_pct:
                        if best_edge < thresh:
                            continue
                        for policy in self.stake_policies:
                            stake = policy.stake(
                                BetInput(p=p, odds=odds_u, edge_pct=best_edge),
                                self.bankroll_eur,
                            )
                            profit = ((odds_u - 1.0) * stake) if won else (-stake)
                            records.append(_BetRecord(
                                match_id=str(row.get("match_id", f"row_{i}")),
                                market=mk_label,
                                threshold_pct=thresh,
                                stake_policy=policy.label(),
                                odds_source=odds.source,
                                stake_eur=stake,
                                profit_eur=profit,
                                won=won,
                                p_predicted=p,
                                fair_prob=fair_u,
                                edge_pct=best_edge,
                                odds=odds_u,
                            ))
                n_scored += len(sub)

        # Assemble report
        meta = BacktestRunMetadata(
            predictor_name=predictor.name,
            predictor_version=predictor.version,
            seasons=list(seasons),
            markets=sorted(set([r.market for r in records])),
            edge_thresholds_pct=list(self.edge_thresholds_pct),
            stake_policies=[p.label() for p in self.stake_policies],
            odds_sources_observed=sorted(all_sources),
            n_matches_scored=n_scored,
            seed_salt=self.seed_salt,
        )
        report = BacktestReport(metadata=meta, markets={})

        # Group records by (market, threshold, stake_policy, odds_source)
        markets_set = sorted(set(r.market for r in records))
        for mk in markets_set:
            report.markets[mk] = {}
            for thresh in self.edge_thresholds_pct:
                thresh_key = f"thresh_{thresh:g}pct"
                report.markets[mk][thresh_key] = {}
                for policy in self.stake_policies:
                    sp_key = policy.label()
                    report.markets[mk][thresh_key][sp_key] = {}
                    # Per-source stats
                    for src in sorted(all_sources) + ["ALL"]:
                        matched = [
                            r for r in records
                            if r.market == mk and r.threshold_pct == thresh
                            and r.stake_policy == sp_key
                            and (src == "ALL" or r.odds_source == src)
                        ]
                        if not matched:
                            continue
                        stakes = np.array([r.stake_eur for r in matched])
                        profits = np.array([r.profit_eur for r in matched])
                        stats = compute_roi_stats(
                            stakes, profits,
                            n_resample=1000,
                            seed_salt=self.seed_salt + hash((mk, thresh_key, sp_key, src)) % 10_000_000,
                        )
                        report.markets[mk][thresh_key][sp_key][src] = stats
        return report


# ---------------------------------------------------------------------------
# Canonical market registry — used by all predictors
# ---------------------------------------------------------------------------

SERIE_A_BINARY_MARKETS: list[BinaryMarket] = [
    BinaryMarket("O/U 2.5", "over_2_5",
                 (("odds_PS_close_over25", "odds_PS_close_under25"),
                  ("odds_B365_close_over25", "odds_B365_close_under25"),
                  ("odds_Avg_over25", "odds_Avg_under25"),
                  ("odds_B365_over25", "odds_B365_under25"))),
    # O/U 1.5, 3.5 lines exist in poisson_over_{1,3}_5 but bookmaker odds are
    # not in matches.parquet today — the harness will report target_available
    # but 0 matched bets. That's correct behavior.
    BinaryMarket("O/U 1.5", "over_1_5",
                 (("odds_PS_close_over15", "odds_PS_close_under15"),
                  ("odds_B365_over15", "odds_B365_under15"))),
    BinaryMarket("O/U 3.5", "over_3_5",
                 (("odds_PS_close_over35", "odds_PS_close_under35"),
                  ("odds_B365_over35", "odds_B365_under35"))),
    BinaryMarket("BTTS", "btts",
                 (("odds_PS_close_btts_yes", "odds_PS_close_btts_no"),
                  ("odds_B365_btts_yes", "odds_B365_btts_no"))),
    # Phase 2 corners/cards — odds blocked until P2 backfill (May 1).
    # Target columns exist (computed in harness._ensure_targets); odds chains
    # are future-ready. Today these markets emit "target_available" but 0 bets.
    BinaryMarket("corners_over_8_5", "corners_over_8_5",
                 (("odds_PS_close_corners_over_85", "odds_PS_close_corners_under_85"),)),
    BinaryMarket("corners_over_9_5", "corners_over_9_5",
                 (("odds_PS_close_corners_over_95", "odds_PS_close_corners_under_95"),)),
    BinaryMarket("corners_over_10_5", "corners_over_10_5",
                 (("odds_PS_close_corners_over_105", "odds_PS_close_corners_under_105"),)),
    BinaryMarket("cards_over_3_5", "cards_over_3_5",
                 (("odds_PS_close_cards_over_35", "odds_PS_close_cards_under_35"),)),
    BinaryMarket("cards_over_4_5", "cards_over_4_5",
                 (("odds_PS_close_cards_over_45", "odds_PS_close_cards_under_45"),)),
    BinaryMarket("cards_over_5_5", "cards_over_5_5",
                 (("odds_PS_close_cards_over_55", "odds_PS_close_cards_under_55"),)),
]

SERIE_A_MULTICLASS_MARKETS: list[MulticlassMarket] = [
    MulticlassMarket(
        "1X2", "result",
        classes=("H", "D", "A"),
        odds_chain=(
            ("odds_PSH", "odds_PSD", "odds_PSA"),
            ("odds_PS_close_H", "odds_PS_close_D", "odds_PS_close_A"),
            ("odds_AvgH", "odds_AvgD", "odds_AvgA"),
            ("odds_B365H", "odds_B365D", "odds_B365A"),
        ),
    ),
]
