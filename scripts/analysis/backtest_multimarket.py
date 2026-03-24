"""Multi-market backtest with lineup-adjusted xG comparison.

Backtests 3 markets (1X2, O/U 2.5, Asian Handicap) against Pinnacle sharp odds.
Compares team-level xG vs lineup-adjusted xG from player profiles.
Sweeps edge thresholds (2-8%) and tags situational context per match.

Usage:
    python3 -m scripts.analysis.backtest_multimarket \
        --seasons 2023-2024 2024-2025 --use-lineup-xg

    python3 -m scripts.analysis.backtest_multimarket \
        --seasons 2023-2024 2024-2025 --no-lineup-xg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from config.settings import DATA_DIR, MODELS_DIR
from storage.paths import features_path

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
FLAT_STAKE = 20.0
MAX_GOALS = 11  # 0..10 for Poisson matrix

SITUATIONAL_TAGS = {
    "is_derby": "derby",
    "is_midweek": "midweek",
    "home_is_promoted": "promoted_home",
    "away_is_promoted": "promoted_away",
    "home_manager_changed": "mgr_change_home",
    "away_manager_changed": "mgr_change_away",
    "home_short_rest": "short_rest_home",
    "away_short_rest": "short_rest_away",
    "home_post_intl_break": "post_intl_home",
    "away_post_intl_break": "post_intl_away",
}

# Column name mapping for odds
ODDS_MAP = {
    "1x2_open": {"H": "odds_PSH", "D": "odds_PSD", "A": "odds_PSA"},
    "1x2_close": {"H": "odds_PSCH", "D": "odds_PSCD", "A": "odds_PSCA"},
    "ou_open": {"over": "odds_P_gt_2.5", "under": "odds_P_lt_2.5"},
    "ou_close": {"over": "odds_PC_gt_2.5", "under": "odds_PC_lt_2.5"},
    "ah_open": {"home": "odds_PAHH", "away": "odds_PAHA"},
    "ah_close": {"home": "odds_PCAHH", "away": "odds_PCAHA"},
    "ah_line_open": "odds_AHh",
    "ah_line_close": "odds_AHCh",
}


# ---------------------------------------------------------------------------
# Poisson goal matrix and market probability derivations
# ---------------------------------------------------------------------------


def poisson_goal_matrix(home_xg: float, away_xg: float) -> np.ndarray:
    """Build an (MAX_GOALS x MAX_GOALS) joint probability matrix."""
    home_pmf = np.array([poisson.pmf(g, home_xg) for g in range(MAX_GOALS)])
    away_pmf = np.array([poisson.pmf(g, away_xg) for g in range(MAX_GOALS)])
    return np.outer(home_pmf, away_pmf)


def ou_probs_from_matrix(matrix: np.ndarray, line: float = 2.5) -> Dict[str, float]:
    """Over/Under probabilities from goal matrix."""
    prob_over = 0.0
    prob_under = 0.0
    n = matrix.shape[0]
    for h in range(n):
        for a in range(n):
            total = h + a
            if total > line:
                prob_over += matrix[h, a]
            elif total < line:
                prob_under += matrix[h, a]
            # total == line => push (ignored for 2.5 since goals are integers)
    total_p = prob_over + prob_under
    if total_p > 0:
        return {"over": prob_over / total_p, "under": prob_under / total_p}
    return {"over": 0.5, "under": 0.5}


def btts_probs_from_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """Both Teams To Score probabilities."""
    n = matrix.shape[0]
    prob_yes = 0.0
    for h in range(1, n):
        for a in range(1, n):
            prob_yes += matrix[h, a]
    return {"yes": prob_yes, "no": 1.0 - prob_yes}


def ah_probs_from_matrix(matrix: np.ndarray, line: float) -> Dict[str, float]:
    """Asian Handicap probabilities.

    `line` is from the home team's perspective (negative = home favorite).
    E.g., line=-0.5 means home -0.5; home wins if they win outright.

    Returns P(home covers), P(away covers). Push probability is split 50/50
    for quarter lines (handled in settlement instead).
    """
    n = matrix.shape[0]
    prob_home = 0.0
    prob_away = 0.0
    prob_push = 0.0

    for h in range(n):
        for a in range(n):
            adj_diff = (h - a) + line  # home goals - away goals + handicap
            if adj_diff > 0:
                prob_home += matrix[h, a]
            elif adj_diff < 0:
                prob_away += matrix[h, a]
            else:
                prob_push += matrix[h, a]

    # For whole/half lines push is impossible. For quarter lines, push is
    # settled as half-win/half-loss. We include push mass split evenly.
    total = prob_home + prob_away + prob_push
    if total > 0:
        # Normalize without push (push refunded), but include push mass
        # split evenly for probability estimation
        prob_home_adj = prob_home + 0.5 * prob_push
        prob_away_adj = prob_away + 0.5 * prob_push
        t = prob_home_adj + prob_away_adj
        return {"home_cover": prob_home_adj / t, "away_cover": prob_away_adj / t}
    return {"home_cover": 0.5, "away_cover": 0.5}


def match_1x2_probs(matrix: np.ndarray) -> Dict[str, float]:
    """1X2 probabilities from goal matrix (for xG-only comparison)."""
    n = matrix.shape[0]
    pH = pD = pA = 0.0
    for h in range(n):
        for a in range(n):
            if h > a:
                pH += matrix[h, a]
            elif h == a:
                pD += matrix[h, a]
            else:
                pA += matrix[h, a]
    t = pH + pD + pA
    return {"prob_H": pH / t, "prob_D": pD / t, "prob_A": pA / t}


# ---------------------------------------------------------------------------
# Odds processing
# ---------------------------------------------------------------------------


def remove_overround(odds_list: List[float]) -> List[float]:
    """Remove overround using power method (Shin's simplified).

    More accurate than simple normalization for 2-way markets.
    Falls back to basic normalization for robustness.
    """
    implied = [1.0 / o for o in odds_list]
    total = sum(implied)
    if total <= 0:
        n = len(odds_list)
        return [1.0 / n] * n
    # Basic normalization (sufficient for Pinnacle's thin margins ~2-3%)
    return [p / total for p in implied]


def compute_clv(opening_odds: float, closing_odds: float) -> float:
    """Closing Line Value: positive = beat the close.

    CLV = (1/closing) - (1/opening)  →  positive means we got a better price.
    """
    if opening_odds <= 1 or closing_odds <= 1:
        return 0.0
    return (1.0 / closing_odds) - (1.0 / opening_odds)


# ---------------------------------------------------------------------------
# Bet settlement
# ---------------------------------------------------------------------------


def settle_1x2(selection: str, home_goals: int, away_goals: int) -> float:
    """Settle a 1X2 bet. Returns 1.0 (win), 0.0 (loss)."""
    actual = "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A")
    return 1.0 if selection == actual else 0.0


def settle_ou(selection: str, home_goals: int, away_goals: int, line: float = 2.5) -> float:
    """Settle Over/Under. Returns 1.0 (win), 0.5 (push), 0.0 (loss)."""
    total = home_goals + away_goals
    if selection == "over":
        if total > line:
            return 1.0
        elif total == line:
            return 0.5  # push (only possible with whole number lines)
        return 0.0
    else:  # under
        if total < line:
            return 1.0
        elif total == line:
            return 0.5
        return 0.0


def settle_ah(selection: str, home_goals: int, away_goals: int, line: float) -> float:
    """Settle Asian Handicap. Handles quarter lines.

    `line` is from home perspective (negative = home favorite).
    `selection` is 'home' or 'away'.

    Quarter lines are split into two half-bets:
      -0.75 → half at -0.5, half at -1.0
      -0.25 → half at  0.0, half at -0.5
    """
    diff = home_goals - away_goals  # positive = home winning

    # Check if quarter line
    frac = abs(line) % 0.5
    is_quarter = abs(frac - 0.25) < 0.01

    if is_quarter:
        # Split into two half-bets
        line1 = line - 0.25  # more favorable to away
        line2 = line + 0.25  # more favorable to home
        r1 = _settle_ah_single(selection, diff, line1)
        r2 = _settle_ah_single(selection, diff, line2)
        return 0.5 * r1 + 0.5 * r2
    else:
        return _settle_ah_single(selection, diff, line)


def _settle_ah_single(selection: str, diff: int, line: float) -> float:
    """Settle single AH line. diff = home_goals - away_goals."""
    adj = diff + line
    if selection == "home":
        if adj > 0:
            return 1.0
        elif adj == 0:
            return 0.5  # push → refund
        return 0.0
    else:  # away
        if adj < 0:
            return 1.0
        elif adj == 0:
            return 0.5
        return 0.0


# ---------------------------------------------------------------------------
# Situational tagging
# ---------------------------------------------------------------------------


def tag_match(row: pd.Series) -> List[str]:
    """Extract situational tags from a match row."""
    tags = []
    for col, label in SITUATIONAL_TAGS.items():
        val = row.get(col, 0)
        if pd.notna(val) and val:
            tags.append(label)

    # Rest advantage (numeric)
    rest_adv = row.get("rest_advantage", 0)
    if pd.notna(rest_adv) and abs(rest_adv) >= 2:
        tags.append("rest_advantage" if rest_adv > 0 else "rest_disadvantage")

    return tags


# ---------------------------------------------------------------------------
# xG sources
# ---------------------------------------------------------------------------


def get_team_level_xg(row: pd.Series) -> Optional[Tuple[float, float]]:
    """Get team-level xG from pre-computed features (same as predict_xg_poisson)."""
    home_atk = row.get("home_xg_attack_strength", np.nan)
    home_def = row.get("home_xg_defense_strength", np.nan)
    away_atk = row.get("away_xg_attack_strength", np.nan)
    away_def = row.get("away_xg_defense_strength", np.nan)

    if any(np.isnan(v) for v in [home_atk, home_def, away_atk, away_def]):
        return None

    # Calibrated against 2023-2025 actual goal distributions (760 matches).
    # Old: league_avg=1.38, hb=1.08, ab=0.92 → total xG 2.817 (actual 2.586), +4.2% Over bias
    # New: league_avg=1.28, hb=1.06, ab=0.94 → total xG 2.614, -0.6% Over bias, brier 0.2538
    league_avg = 1.28
    home_xg = home_atk * away_def * league_avg
    away_xg = away_atk * home_def * league_avg

    home_xg = max(0.4, min(3.5, home_xg))
    away_xg = max(0.3, min(3.0, away_xg))

    # Home advantage adjustment (calibrated)
    home_xg *= 1.06
    away_xg *= 0.94

    return home_xg, away_xg


def get_lineup_xg(
    row: pd.Series,
    lineup_db: pd.DataFrame,
    predictor,  # LineupXGPredictor
) -> Optional[Tuple[float, float]]:
    """Get lineup-adjusted xG using confirmed lineups and player profiles.

    Args:
        row: Match row from features.parquet
        lineup_db: lineups.parquet DataFrame
        predictor: LineupXGPredictor instance

    Returns:
        (home_xg, away_xg) or None if no lineup data for this match
    """
    match_id = row.get("match_id")
    if not match_id or match_id not in lineup_db.index:
        return None

    match_lineups = lineup_db.loc[match_id]
    home_team = row["home_team"]
    away_team = row["away_team"]

    # Get starters for each team
    if isinstance(match_lineups, pd.Series):
        # Single row — shouldn't happen for a match with 22+ players
        return None

    home_starters = match_lineups[
        (match_lineups["is_home"] == True) & (match_lineups["role"] == "starter")
    ]["player_name"].tolist()

    away_starters = match_lineups[
        (match_lineups["is_home"] == False) & (match_lineups["role"] == "starter")
    ]["player_name"].tolist()

    if len(home_starters) < 7 or len(away_starters) < 7:
        return None

    # Use LineupXGPredictor with actual confirmed lineups
    home_pred = predictor.predict_team_xg(
        team=home_team,
        lineup=home_starters,
        use_form=True,
        opponent=away_team,
    )
    away_pred = predictor.predict_team_xg(
        team=away_team,
        lineup=away_starters,
        use_form=True,
        opponent=home_team,
    )

    home_xg = home_pred["predicted_xg"]
    away_xg = away_pred["predicted_xg"]

    # Home advantage adjustment (calibrated)
    home_xg *= 1.06
    away_xg *= 0.94

    home_xg = max(0.4, min(3.5, home_xg))
    away_xg = max(0.3, min(3.0, away_xg))

    return home_xg, away_xg


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------


@dataclass
class BetRecord:
    match_id: str
    season: str
    match_date: str
    home_team: str
    away_team: str
    market: str          # "1X2_H", "OU_Over", "AH_Home", etc.
    selection: str       # "H", "over", "home", etc.
    xg_source: str       # "ensemble", "team_xg", "lineup_xg"
    model_prob: float
    sharp_prob: float
    edge: float
    odds: float
    stake: float
    result_goals: str    # "2-1"
    settlement: float    # 1.0, 0.5, 0.0
    profit: float
    clv: float
    tags: List[str]


class MultiMarketBacktest:
    def __init__(
        self,
        seasons: List[str],
        edge_thresholds: List[float] = None,
        use_lineup_xg: bool = True,
        walk_forward: bool = False,
    ):
        self.seasons = seasons
        self.thresholds = edge_thresholds or EDGE_THRESHOLDS
        self.use_lineup_xg = use_lineup_xg
        self.walk_forward = walk_forward
        self.df: Optional[pd.DataFrame] = None
        self.lineup_db: Optional[pd.DataFrame] = None
        self.lineup_predictor = None

    def load_data(self) -> bool:
        """Load features.parquet and optionally lineups."""
        path = features_path()
        if not path.exists():
            log.error("features.parquet not found at %s", path)
            return False

        self.df = pd.read_parquet(path)
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        self.df = self.df.dropna(subset=["result"])
        self.df = self.df[self.df["result"].isin(["H", "D", "A"])].copy()

        # Filter to requested seasons
        self.df = self.df[self.df["season"].isin(self.seasons)].copy()
        log.info("Loaded %d matches for seasons %s", len(self.df), self.seasons)

        if self.use_lineup_xg:
            self._load_lineups()

        return len(self.df) > 0

    def _load_lineups(self):
        """Load lineup data and initialize LineupXGPredictor."""
        lineup_path = DATA_DIR / "parsed" / "lineups.parquet"
        if not lineup_path.exists():
            log.warning("lineups.parquet not found — lineup-adjusted xG disabled")
            self.use_lineup_xg = False
            return

        ldf = pd.read_parquet(lineup_path)
        # Index by match_id for fast lookup
        self.lineup_db = ldf.set_index("match_id", drop=False)
        n_matches = ldf["match_id"].nunique()
        log.info("Loaded lineups for %d matches", n_matches)

        # Check overlap with our data
        our_ids = set(self.df["match_id"].unique())
        lineup_ids = set(ldf["match_id"].unique())
        overlap = our_ids & lineup_ids
        log.info("Lineup coverage: %d/%d matches (%.0f%%)",
                 len(overlap), len(our_ids), 100 * len(overlap) / max(len(our_ids), 1))

        if len(overlap) == 0:
            log.warning("No lineup overlap — lineup-adjusted xG disabled")
            self.use_lineup_xg = False
            return

        # Initialize LineupXGPredictor
        try:
            from features.player_xg_model import LineupXGPredictor, PlayerXGDatabase

            player_db = PlayerXGDatabase()
            if not player_db.load():
                log.info("Building player database from match data...")
                player_db.build_from_match_data()
                player_db.build_from_sofascore()
            self.lineup_predictor = LineupXGPredictor(player_db)
            log.info("LineupXGPredictor initialized with %d player profiles",
                     len(player_db.profiles))
        except Exception as e:
            log.warning("Failed to init LineupXGPredictor: %s — disabling lineup xG", e)
            self.use_lineup_xg = False

    def run(self) -> Dict[str, Any]:
        """Run the full multi-market backtest."""
        if self.df is None or self.df.empty:
            return {"error": "No data loaded"}

        # Import 1X2 ensemble predictor
        from scripts.analysis.backtest_unified import predict_ensemble

        # Enable walk-forward mode if requested (uses per-fold models, no leakage)
        if self.walk_forward:
            from scripts.analysis.backtest_unified import enable_walk_forward_backtest
            if enable_walk_forward_backtest():
                log.info("Walk-forward backtest: using per-fold models (leakage-free)")
            else:
                log.warning("Walk-forward requested but no fold models found. "
                            "Run `from ml.ensemble import build_fold_models; build_fold_models()` first. "
                            "Falling back to all-data model (in-sample).")

        # Collect all bets across all thresholds
        all_bets: Dict[float, List[BetRecord]] = {t: [] for t in self.thresholds}

        n_total = len(self.df)
        n_with_lineup = 0
        n_skipped_no_odds = 0

        for idx, (_, row) in enumerate(self.df.iterrows()):
            if idx % 100 == 0:
                log.info("Processing match %d/%d", idx + 1, n_total)

            match_id = row["match_id"]
            home_goals = int(row["home_score"]) if pd.notna(row.get("home_score")) else None
            away_goals = int(row["away_score"]) if pd.notna(row.get("away_score")) else None

            if home_goals is None or away_goals is None:
                continue

            tags = tag_match(row)
            match_meta = {
                "match_id": match_id,
                "season": row["season"],
                "match_date": str(row["match_date"])[:10],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "result_goals": f"{home_goals}-{away_goals}",
                "tags": tags,
            }

            # --- 1. 1X2 via ensemble ---
            self._process_1x2(row, predict_ensemble, home_goals, away_goals,
                              match_meta, all_bets)

            # --- 2. Team-level xG → Poisson matrix ---
            team_xg = get_team_level_xg(row)
            if team_xg:
                home_xg_t, away_xg_t = team_xg
                matrix_t = poisson_goal_matrix(home_xg_t, away_xg_t)
                self._process_ou(row, matrix_t, "team_xg", home_goals, away_goals,
                                 match_meta, all_bets)
                self._process_ah(row, matrix_t, "team_xg", home_goals, away_goals,
                                 match_meta, all_bets)

            # --- 3. Lineup-adjusted xG → Poisson matrix ---
            if self.use_lineup_xg and self.lineup_db is not None:
                lineup_xg = get_lineup_xg(row, self.lineup_db, self.lineup_predictor)
                if lineup_xg:
                    n_with_lineup += 1
                    home_xg_l, away_xg_l = lineup_xg
                    matrix_l = poisson_goal_matrix(home_xg_l, away_xg_l)
                    self._process_ou(row, matrix_l, "lineup_xg", home_goals, away_goals,
                                     match_meta, all_bets)
                    self._process_ah(row, matrix_l, "lineup_xg", home_goals, away_goals,
                                     match_meta, all_bets)

                    # --- 4. Blended xG (production behavior) ---
                    # Mirrors over_under_model.py: blend lineup xG into team xG
                    # 15% weight = predicted lineups, 40% = confirmed lineups
                    if team_xg:
                        for blend_name, blend_w in [("blended_15", 0.15), ("blended_40", 0.40)]:
                            bh = (1 - blend_w) * home_xg_t + blend_w * home_xg_l
                            ba = (1 - blend_w) * away_xg_t + blend_w * away_xg_l
                            matrix_b = poisson_goal_matrix(bh, ba)
                            self._process_ou(row, matrix_b, blend_name, home_goals, away_goals,
                                             match_meta, all_bets)
                            self._process_ah(row, matrix_b, blend_name, home_goals, away_goals,
                                             match_meta, all_bets)

        log.info("Done. Matches with lineup xG: %d/%d", n_with_lineup, n_total)

        # Build results structure
        results = self._build_results(all_bets)
        results["meta"] = {
            "seasons": self.seasons,
            "total_matches": n_total,
            "matches_with_lineup": n_with_lineup,
            "use_lineup_xg": self.use_lineup_xg,
            "walk_forward": self.walk_forward,
            "evaluation_type": "out-of-sample" if self.walk_forward else "in-sample",
            "thresholds": self.thresholds,
            "flat_stake": FLAT_STAKE,
            "run_date": datetime.now().isoformat(),
        }
        if not self.walk_forward:
            results["meta"]["leakage_warning"] = (
                "ML predictions use all-data model (in-sample). "
                "ROI may be overstated by 2-5pp. Use walk_forward=True for honest evaluation."
            )
        return results

    # --- Production simulation ---

    # Situational adjustments (mirrors betting_unified.py BettingConfig)
    _PROD_SITUATIONAL = {
        "derby__1X2__Draw":          -2.0,
        "promoted_away__1X2__Draw":  -1.5,
        "mgr_change__1X2__Draw":     -1.5,
        "derby__O/U__Over":          +3.0,
        "midweek__O/U__Over":        +2.0,
        "midweek__O/U__Under":       +2.0,
    }

    # Promoted teams per season (mirrors betting_unified.py)
    _PROMOTED = {
        "2023-2024": {"Cagliari", "Frosinone", "Genoa"},
        "2024-2025": {"Como", "Parma", "Venezia"},
        "2025-2026": {"Sassuolo", "Pisa", "Cremonese"},
    }

    def _prod_situational_tags(self, row: pd.Series) -> List[str]:
        """Compute situational tags for a match row (mirrors betting engine)."""
        tags = tag_match(row)  # reuse existing tag_match()
        # Add promoted tags if not already present
        season = row.get("season", "")
        promoted = self._PROMOTED.get(season, set())
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        if home in promoted and "promoted_home" not in tags:
            tags.append("promoted_home")
        if away in promoted and "promoted_away" not in tags:
            tags.append("promoted_away")
        return tags

    def _prod_adjust_threshold(self, base_edge: float, market_cat: str,
                                selection: str, tags: List[str]) -> float:
        """Apply situational adjustment to edge threshold."""
        total_adj = 0.0
        sel_norm = selection.split("_")[-1] if "_" in selection else selection
        for tag in tags:
            key = f"{tag}__{market_cat}__{sel_norm}"
            if key in self._PROD_SITUATIONAL:
                total_adj += self._PROD_SITUATIONAL[key]
        return max(1.0, base_edge + total_adj)

    def run_production(
        self,
        staking: str = "proportional",
        bankroll_start: float = 1000.0,
        kelly_fraction: float = 0.10,
        proportional_pct: float = 2.0,
        max_stake_pct: float = 2.5,
        draw_min_edge: float = 0.04,
        ou_min_edge: float = 0.06,
        max_edge: float = 0.08,
        enable_1x2_home: bool = False,
        enable_1x2_away: bool = True,
        home_min_edge: float = 0.07,
        away_min_edge: float = 0.09,
    ) -> Dict[str, Any]:
        """Simulate the full production betting system over historical data.

        Applies ALL production rules simultaneously:
        - 1X2: Draw (+ optionally Home/Away) with per-selection min edge
        - O/U 2.5: Over only, configurable min edge
        - AH: DISABLED
        - xG: blended (15% lineup where available, team_xg otherwise)
        - Max edge: configurable cap (filter model overconfidence)

        Staking modes:
        - "flat": fixed FLAT_STAKE per bet (legacy, for comparison)
        - "kelly": fractional Kelly, sized by edge (kelly_fraction default 0.10)
        - "proportional": fixed % of current bankroll (default 2%)
        """
        if self.df is None or self.df.empty:
            return {"error": "No data loaded"}

        from scripts.analysis.backtest_unified import predict_ensemble
        from scripts.betting.betting_unified import calculate_kelly

        # Enable walk-forward mode if requested
        if self.walk_forward:
            from scripts.analysis.backtest_unified import enable_walk_forward_backtest
            if enable_walk_forward_backtest():
                log.info("Walk-forward production backtest: per-fold models (leakage-free)")
            else:
                log.warning("Walk-forward requested but no fold models. Using all-data model.")

        # Production rules (parameterized, defaults from Phase 7 threshold sweep)
        DRAW_MIN_EDGE = draw_min_edge
        OU_MIN_EDGE = ou_min_edge
        MAX_EDGE = max_edge
        HOME_MIN_EDGE = home_min_edge
        AWAY_MIN_EDGE = away_min_edge

        bets: List[BetRecord] = []
        n_total = len(self.df)
        n_with_lineup = 0
        bankroll = bankroll_start
        peak_bankroll = bankroll
        max_drawdown = 0.0
        bankroll_history: List[float] = [bankroll]

        def _compute_stake(model_prob: float, odds: float) -> float:
            """Compute stake based on staking mode and current bankroll."""
            if staking == "flat":
                return FLAT_STAKE
            elif staking == "kelly":
                k = calculate_kelly(model_prob, odds, fraction=kelly_fraction)
                pct = min(k * 100, max_stake_pct)
                if pct < 0.15:
                    return 0  # Edge too small for Kelly
                pct = max(pct, 0.5)  # Floor at 0.5%
                return round(bankroll * pct / 100, 2)
            else:  # proportional
                pct = min(proportional_pct, max_stake_pct)
                return round(bankroll * pct / 100, 2)

        for idx, (_, row) in enumerate(self.df.iterrows()):
            if idx % 100 == 0:
                log.info("Production sim: match %d/%d (bankroll: %.0f)", idx + 1, n_total, bankroll)

            home_goals = int(row["home_score"]) if pd.notna(row.get("home_score")) else None
            away_goals = int(row["away_score"]) if pd.notna(row.get("away_score")) else None
            if home_goals is None or away_goals is None:
                continue

            tags = self._prod_situational_tags(row)
            meta = {
                "match_id": row["match_id"],
                "season": row["season"],
                "match_date": str(row["match_date"])[:10],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "result_goals": f"{home_goals}-{away_goals}",
                "tags": tags,
            }

            # --- 1X2 (Draw + optionally Home/Away) ---
            psh = row.get(ODDS_MAP["1x2_open"]["H"], np.nan)
            psd = row.get(ODDS_MAP["1x2_open"]["D"], np.nan)
            psa = row.get(ODDS_MAP["1x2_open"]["A"], np.nan)
            if not any(pd.isna(v) or v <= 1.0 for v in [psh, psd, psa]):
                pred = predict_ensemble(row)
                sharp_probs = remove_overround([psh, psd, psa])

                # Build list of 1X2 selections to check
                _1x2_checks = [
                    ("1X2_Draw", "D", pred["prob_D"], sharp_probs[1], psd, DRAW_MIN_EDGE, True),
                ]
                if enable_1x2_home:
                    _1x2_checks.append(
                        ("1X2_Home", "H", pred["prob_H"], sharp_probs[0], psh, HOME_MIN_EDGE, True))
                if enable_1x2_away:
                    _1x2_checks.append(
                        ("1X2_Away", "A", pred["prob_A"], sharp_probs[2], psa, AWAY_MIN_EDGE, True))

                for mkt_label, sel, model_p, sharp_p, odds_open, min_edge_base, enabled in _1x2_checks:
                    if not enabled:
                        continue
                    edge = model_p - sharp_p
                    adj_min = self._prod_adjust_threshold(
                        min_edge_base * 100, "1X2", sel, tags) / 100
                    adj_max = MAX_EDGE

                    if adj_min <= edge <= adj_max:
                        stake = _compute_stake(model_p, odds_open)
                        if stake > 0:
                            close_key = {"H": "H", "D": "D", "A": "A"}[sel]
                            odds_close = row.get(ODDS_MAP["1x2_close"][close_key], np.nan)
                            clv = compute_clv(odds_open, odds_close) if pd.notna(odds_close) and odds_close > 1 else 0.0
                            settlement = settle_1x2(sel, home_goals, away_goals)
                            profit = stake * (odds_open - 1) * settlement - stake * (1 - settlement)
                            bankroll += profit
                            bets.append(BetRecord(
                                **meta, market=mkt_label, selection=sel,
                                xg_source="ensemble", model_prob=model_p,
                                sharp_prob=sharp_p, edge=edge, odds=odds_open,
                                stake=stake, settlement=settlement,
                                profit=profit, clv=clv,
                            ))
                            peak_bankroll = max(peak_bankroll, bankroll)
                            dd = (peak_bankroll - bankroll) / peak_bankroll
                            max_drawdown = max(max_drawdown, dd)
                            bankroll_history.append(bankroll)

            # --- O/U 2.5 Over (blended xG) ---
            odds_over = row.get(ODDS_MAP["ou_open"]["over"], np.nan)
            odds_under = row.get(ODDS_MAP["ou_open"]["under"], np.nan)
            if not any(pd.isna(v) or v <= 1.0 for v in [odds_over, odds_under]):
                team_xg = get_team_level_xg(row)
                lineup_xg_vals = None
                if self.use_lineup_xg and self.lineup_db is not None:
                    lineup_xg_vals = get_lineup_xg(row, self.lineup_db, self.lineup_predictor)

                if team_xg:
                    home_xg_t, away_xg_t = team_xg
                    if lineup_xg_vals:
                        n_with_lineup += 1
                        home_xg_l, away_xg_l = lineup_xg_vals
                        w = 0.15
                        bh = (1 - w) * home_xg_t + w * home_xg_l
                        ba = (1 - w) * away_xg_t + w * away_xg_l
                        xg_src = "blended_15"
                    else:
                        bh, ba = home_xg_t, away_xg_t
                        xg_src = "team_xg"

                    matrix = poisson_goal_matrix(bh, ba)
                    model_probs = ou_probs_from_matrix(matrix, line=2.5)
                    sharp_probs_ou = remove_overround([odds_over, odds_under])

                    model_over = model_probs["over"]
                    edge = model_over - sharp_probs_ou[0]

                    adj_min = self._prod_adjust_threshold(
                        OU_MIN_EDGE * 100, "O/U", "Over", tags) / 100
                    adj_max = MAX_EDGE

                    if adj_min <= edge <= adj_max:
                        stake = _compute_stake(model_over, odds_over)
                        if stake > 0:
                            close_over = row.get(ODDS_MAP["ou_close"]["over"], np.nan)
                            clv = compute_clv(odds_over, close_over) if pd.notna(close_over) and close_over > 1 else 0.0
                            settlement = settle_ou("over", home_goals, away_goals)
                            if settlement == 1.0:
                                profit = stake * (odds_over - 1)
                            elif settlement == 0.0:
                                profit = -stake
                            else:
                                profit = 0.0
                            bankroll += profit
                            bets.append(BetRecord(
                                **meta, market="OU_Over", selection="over",
                                xg_source=xg_src, model_prob=model_over,
                                sharp_prob=sharp_probs_ou[0], edge=edge, odds=odds_over,
                                stake=stake, settlement=settlement,
                                profit=profit, clv=clv,
                            ))
                            peak_bankroll = max(peak_bankroll, bankroll)
                            dd = (peak_bankroll - bankroll) / peak_bankroll
                            max_drawdown = max(max_drawdown, dd)
                            bankroll_history.append(bankroll)

        log.info("Production sim done. %d bets placed (%d with lineup xG)", len(bets), n_with_lineup)

        # Build results
        def stats(bl):
            if not bl:
                return {"bets": 0, "wins": 0, "profit": 0, "roi": 0, "avg_clv": 0, "avg_edge": 0}
            wins = sum(1 for b in bl if b.settlement == 1.0)
            profit = sum(b.profit for b in bl)
            total_staked = sum(b.stake for b in bl)
            return {
                "bets": len(bl),
                "wins": wins,
                "win_rate": round(wins / len(bl) * 100, 1) if bl else 0,
                "profit": round(profit, 2),
                "roi": round(profit / total_staked * 100, 1) if total_staked else 0,
                "avg_edge": round(np.mean([b.edge for b in bl]) * 100, 2),
                "avg_clv": round(np.mean([b.clv for b in bl]) * 100, 3) if bl else 0,
            }

        # Per-market grouping
        markets = sorted(set(b.market for b in bets)) if bets else []
        by_market = {}
        for m in markets:
            by_market[m] = stats([b for b in bets if b.market == m])

        # Per-season split
        seasons = sorted(set(b.season for b in bets)) if bets else []
        per_season = {}
        for s in seasons:
            s_bets = [b for b in bets if b.season == s]
            per_season[s] = {"combined": stats(s_bets)}
            for m in markets:
                per_season[s][m] = stats([b for b in s_bets if b.market == m])

        return {
            "mode": "production",
            "staking": {
                "mode": staking,
                "bankroll_start": bankroll_start,
                "bankroll_final": round(bankroll, 2),
                "bankroll_growth": round((bankroll / bankroll_start - 1) * 100, 1),
                "max_drawdown_pct": round(max_drawdown * 100, 1),
                "kelly_fraction": kelly_fraction if staking == "kelly" else None,
                "proportional_pct": proportional_pct if staking == "proportional" else None,
            },
            "meta": {
                "seasons": self.seasons,
                "total_matches": n_total,
                "matches_with_lineup": n_with_lineup,
                "draw_min_edge": DRAW_MIN_EDGE,
                "ou_min_edge": OU_MIN_EDGE,
                "home_min_edge": HOME_MIN_EDGE if enable_1x2_home else None,
                "away_min_edge": AWAY_MIN_EDGE if enable_1x2_away else None,
                "max_edge": MAX_EDGE,
                "situational_adjustments": self._PROD_SITUATIONAL,
                "run_date": datetime.now().isoformat(),
            },
            "by_market": by_market,
            "combined": stats(bets),
            "per_season": per_season,
            "all_bets": bets,
            "bankroll_history": bankroll_history,
        }

    # --- Market processors ---

    def _process_1x2(
        self,
        row: pd.Series,
        predict_fn,
        home_goals: int,
        away_goals: int,
        meta: Dict,
        all_bets: Dict[float, List[BetRecord]],
    ):
        """Generate 1X2 ensemble predictions and check for value."""
        # Get odds
        psh = row.get(ODDS_MAP["1x2_open"]["H"], np.nan)
        psd = row.get(ODDS_MAP["1x2_open"]["D"], np.nan)
        psa = row.get(ODDS_MAP["1x2_open"]["A"], np.nan)

        if any(pd.isna(v) or v <= 1.0 for v in [psh, psd, psa]):
            return

        # Closing odds for CLV
        psch = row.get(ODDS_MAP["1x2_close"]["H"], np.nan)
        pscd = row.get(ODDS_MAP["1x2_close"]["D"], np.nan)
        psca = row.get(ODDS_MAP["1x2_close"]["A"], np.nan)

        # Model probabilities
        pred = predict_fn(row)
        sharp_probs = remove_overround([psh, psd, psa])

        outcomes = [
            ("1X2_Home", "H", pred["prob_H"], sharp_probs[0], psh, psch),
            ("1X2_Draw", "D", pred["prob_D"], sharp_probs[1], psd, pscd),
            ("1X2_Away", "A", pred["prob_A"], sharp_probs[2], psa, psca),
        ]

        for market_label, selection, model_p, sharp_p, odds_open, odds_close in outcomes:
            edge = model_p - sharp_p
            clv = compute_clv(odds_open, odds_close) if pd.notna(odds_close) and odds_close > 1 else 0.0
            settlement = settle_1x2(selection, home_goals, away_goals)

            for thresh in self.thresholds:
                if edge >= thresh:
                    profit = FLAT_STAKE * (odds_open - 1) * settlement - FLAT_STAKE * (1 - settlement)
                    bet = BetRecord(
                        **meta,
                        market=market_label,
                        selection=selection,
                        xg_source="ensemble",
                        model_prob=model_p,
                        sharp_prob=sharp_p,
                        edge=edge,
                        odds=odds_open,
                        stake=FLAT_STAKE,
                        settlement=settlement,
                        profit=profit,
                        clv=clv,
                    )
                    all_bets[thresh].append(bet)

    def _process_ou(
        self,
        row: pd.Series,
        matrix: np.ndarray,
        xg_source: str,
        home_goals: int,
        away_goals: int,
        meta: Dict,
        all_bets: Dict[float, List[BetRecord]],
    ):
        """Process Over/Under 2.5 market."""
        odds_over = row.get(ODDS_MAP["ou_open"]["over"], np.nan)
        odds_under = row.get(ODDS_MAP["ou_open"]["under"], np.nan)

        if any(pd.isna(v) or v <= 1.0 for v in [odds_over, odds_under]):
            return

        # Closing odds
        close_over = row.get(ODDS_MAP["ou_close"]["over"], np.nan)
        close_under = row.get(ODDS_MAP["ou_close"]["under"], np.nan)

        # Model probabilities from Poisson matrix
        model_probs = ou_probs_from_matrix(matrix, line=2.5)
        sharp_probs = remove_overround([odds_over, odds_under])

        selections = [
            ("OU_Over", "over", model_probs["over"], sharp_probs[0], odds_over, close_over),
            ("OU_Under", "under", model_probs["under"], sharp_probs[1], odds_under, close_under),
        ]

        for market_label, selection, model_p, sharp_p, odds_open, odds_close in selections:
            edge = model_p - sharp_p
            clv = compute_clv(odds_open, odds_close) if pd.notna(odds_close) and odds_close > 1 else 0.0
            settlement = settle_ou(selection, home_goals, away_goals)

            for thresh in self.thresholds:
                if edge >= thresh:
                    if settlement == 0.5:
                        profit = 0.0  # push
                    elif settlement == 1.0:
                        profit = FLAT_STAKE * (odds_open - 1)
                    else:
                        profit = -FLAT_STAKE
                    bet = BetRecord(
                        **meta,
                        market=market_label,
                        selection=selection,
                        xg_source=xg_source,
                        model_prob=model_p,
                        sharp_prob=sharp_p,
                        edge=edge,
                        odds=odds_open,
                        stake=FLAT_STAKE,
                        settlement=settlement,
                        profit=profit,
                        clv=clv,
                    )
                    all_bets[thresh].append(bet)

    def _process_ah(
        self,
        row: pd.Series,
        matrix: np.ndarray,
        xg_source: str,
        home_goals: int,
        away_goals: int,
        meta: Dict,
        all_bets: Dict[float, List[BetRecord]],
    ):
        """Process Asian Handicap market."""
        odds_home = row.get(ODDS_MAP["ah_open"]["home"], np.nan)
        odds_away = row.get(ODDS_MAP["ah_open"]["away"], np.nan)
        ah_line = row.get(ODDS_MAP["ah_line_open"], np.nan)

        if any(pd.isna(v) or v <= 1.0 for v in [odds_home, odds_away]) or pd.isna(ah_line):
            return

        # Closing odds + line
        close_home = row.get(ODDS_MAP["ah_close"]["home"], np.nan)
        close_away = row.get(ODDS_MAP["ah_close"]["away"], np.nan)

        # Model probabilities from Poisson matrix
        model_probs = ah_probs_from_matrix(matrix, ah_line)
        sharp_probs = remove_overround([odds_home, odds_away])

        selections = [
            ("AH_Home", "home", model_probs["home_cover"], sharp_probs[0], odds_home, close_home),
            ("AH_Away", "away", model_probs["away_cover"], sharp_probs[1], odds_away, close_away),
        ]

        for market_label, selection, model_p, sharp_p, odds_open, odds_close in selections:
            edge = model_p - sharp_p
            clv = compute_clv(odds_open, odds_close) if pd.notna(odds_close) and odds_close > 1 else 0.0
            settlement = settle_ah(selection, home_goals, away_goals, ah_line)

            for thresh in self.thresholds:
                if edge >= thresh:
                    if settlement == 0.5:
                        profit = 0.0  # push
                    elif settlement == 1.0:
                        profit = FLAT_STAKE * (odds_open - 1)
                    else:
                        profit = -FLAT_STAKE
                    bet = BetRecord(
                        **meta,
                        market=market_label,
                        selection=selection,
                        xg_source=xg_source,
                        model_prob=model_p,
                        sharp_prob=sharp_p,
                        edge=edge,
                        odds=odds_open,
                        stake=FLAT_STAKE,
                        settlement=settlement,
                        profit=profit,
                        clv=clv,
                    )
                    all_bets[thresh].append(bet)

    # --- Results aggregation ---

    def _build_results(self, all_bets: Dict[float, List[BetRecord]]) -> Dict[str, Any]:
        """Aggregate bets into structured results."""
        results = {
            "by_threshold": {},
            "lineup_comparison": {},
            "situational": {},
        }

        for thresh, bets in all_bets.items():
            thresh_key = f"{thresh:.0%}"
            by_market: Dict[str, Dict[str, List[BetRecord]]] = defaultdict(lambda: defaultdict(list))

            for b in bets:
                by_market[b.market][b.xg_source].append(b)

            market_results = {}
            for market, by_source in by_market.items():
                for source, source_bets in by_source.items():
                    key = f"{market}|{source}"
                    stats = self._compute_stats(source_bets)
                    market_results[key] = stats

            results["by_threshold"][thresh_key] = market_results

        # Lineup comparison (team_xg vs lineup_xg at each threshold)
        for thresh, bets in all_bets.items():
            thresh_key = f"{thresh:.0%}"
            team_bets = [b for b in bets if b.xg_source == "team_xg"]
            lineup_bets = [b for b in bets if b.xg_source == "lineup_xg"]

            if team_bets or lineup_bets:
                comparison = {}
                for market in ["OU_Over", "OU_Under", "AH_Home", "AH_Away"]:
                    tb = [b for b in team_bets if b.market == market]
                    lb = [b for b in lineup_bets if b.market == market]
                    if tb or lb:
                        comparison[market] = {
                            "team_xg": self._compute_stats(tb) if tb else None,
                            "lineup_xg": self._compute_stats(lb) if lb else None,
                        }
                results["lineup_comparison"][thresh_key] = comparison

        # Situational breakdown (use best threshold = 5%)
        best_thresh = 0.05
        if best_thresh in all_bets:
            bets_5 = all_bets[best_thresh]
            sit_results = defaultdict(lambda: defaultdict(list))
            for b in bets_5:
                for tag in b.tags:
                    sit_results[tag][b.market].append(b)

            sit_summary = {}
            for tag, by_market in sit_results.items():
                tag_markets = {}
                for market, market_bets in by_market.items():
                    tag_markets[market] = self._compute_stats(market_bets)
                sit_summary[tag] = tag_markets
            results["situational"] = sit_summary

        # Per-season breakdown at 5%
        if best_thresh in all_bets:
            season_results = defaultdict(lambda: defaultdict(list))
            for b in all_bets[best_thresh]:
                season_results[b.season][f"{b.market}|{b.xg_source}"].append(b)

            results["per_season"] = {}
            for season, by_key in season_results.items():
                results["per_season"][season] = {
                    k: self._compute_stats(v) for k, v in by_key.items()
                }

        return results

    @staticmethod
    def _compute_stats(bets: List[BetRecord]) -> Dict[str, Any]:
        """Compute summary stats from a list of bet records."""
        if not bets:
            return {"bets": 0}

        n = len(bets)
        wins = sum(1 for b in bets if b.settlement >= 1.0)
        pushes = sum(1 for b in bets if 0 < b.settlement < 1.0)
        total_profit = sum(b.profit for b in bets)
        total_staked = n * FLAT_STAKE
        avg_odds = np.mean([b.odds for b in bets])
        avg_edge = np.mean([b.edge for b in bets])
        avg_clv = np.mean([b.clv for b in bets])
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        return {
            "bets": n,
            "wins": wins,
            "pushes": pushes,
            "win_rate": wins / n if n > 0 else 0,
            "profit": round(total_profit, 2),
            "roi": round(roi, 2),
            "avg_odds": round(avg_odds, 3),
            "avg_edge": round(avg_edge, 4),
            "avg_clv": round(avg_clv, 4),
        }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(results: Dict[str, Any]):
    """Print formatted backtest results."""
    meta = results.get("meta", {})
    print("\n" + "=" * 90)
    print("MULTI-MARKET BACKTEST REPORT")
    print("=" * 90)
    print(f"  Seasons: {meta.get('seasons', [])}")
    print(f"  Matches: {meta.get('total_matches', 0)}")
    print(f"  Lineup xG matches: {meta.get('matches_with_lineup', 0)}")
    print(f"  Flat stake: €{meta.get('flat_stake', FLAT_STAKE):.0f}")
    print(f"  Run: {meta.get('run_date', '')[:16]}")

    # --- Table 1: Market comparison by threshold ---
    print("\n" + "-" * 90)
    print("TABLE 1: MARKET COMPARISON BY EDGE THRESHOLD")
    print("-" * 90)
    header = f"{'Market':<20} {'Source':<12} {'Thresh':>6} {'Bets':>5} {'Wins':>5} {'WinRate':>8} {'Profit':>10} {'ROI':>8} {'AvgCLV':>8}"
    print(header)
    print("-" * 90)

    by_thresh = results.get("by_threshold", {})
    for thresh_key in sorted(by_thresh.keys()):
        markets = by_thresh[thresh_key]
        for key in sorted(markets.keys()):
            s = markets[key]
            if s["bets"] == 0:
                continue
            parts = key.split("|")
            market = parts[0]
            source = parts[1] if len(parts) > 1 else "?"
            print(
                f"{market:<20} {source:<12} {thresh_key:>6} {s['bets']:>5} "
                f"{s['wins']:>5} {s['win_rate']:>7.1%} "
                f"€{s['profit']:>8.0f} {s['roi']:>7.1f}% {s['avg_clv']:>+7.4f}"
            )

    # --- Table 2: Lineup comparison ---
    lineup_comp = results.get("lineup_comparison", {})
    if lineup_comp:
        print("\n" + "-" * 90)
        print("TABLE 2: LINEUP-ADJUSTED vs TEAM-LEVEL xG COMPARISON")
        print("-" * 90)
        header2 = (
            f"{'Thresh':<8} {'Market':<12} "
            f"{'Bets_T':>6} {'Profit_T':>9} {'ROI_T':>8} "
            f"{'Bets_L':>6} {'Profit_L':>9} {'ROI_L':>8} "
            f"{'Delta':>8}"
        )
        print(header2)
        print("-" * 90)

        for thresh_key in sorted(lineup_comp.keys()):
            markets = lineup_comp[thresh_key]
            for market in sorted(markets.keys()):
                data = markets[market]
                t = data.get("team_xg") or {"bets": 0, "profit": 0, "roi": 0}
                l = data.get("lineup_xg") or {"bets": 0, "profit": 0, "roi": 0}
                delta = l.get("profit", 0) - t.get("profit", 0)
                print(
                    f"{thresh_key:<8} {market:<12} "
                    f"{t.get('bets', 0):>6} €{t.get('profit', 0):>7.0f} {t.get('roi', 0):>7.1f}% "
                    f"{l.get('bets', 0):>6} €{l.get('profit', 0):>7.0f} {l.get('roi', 0):>7.1f}% "
                    f"€{delta:>+7.0f}"
                )

    # --- Table 3: Situational breakdown ---
    sit = results.get("situational", {})
    if sit:
        print("\n" + "-" * 90)
        print("TABLE 3: SITUATIONAL BREAKDOWN (at 5% edge)")
        print("-" * 90)
        header3 = f"{'Situation':<25} {'Market':<20} {'Bets':>5} {'ROI':>8} {'AvgCLV':>8}"
        print(header3)
        print("-" * 90)

        for tag in sorted(sit.keys()):
            markets = sit[tag]
            for market in sorted(markets.keys()):
                s = markets[market]
                if s["bets"] < 3:
                    continue
                print(
                    f"{tag:<25} {market:<20} "
                    f"{s['bets']:>5} {s['roi']:>7.1f}% {s['avg_clv']:>+7.4f}"
                )

    # --- Per-season breakdown ---
    per_season = results.get("per_season", {})
    if per_season:
        print("\n" + "-" * 90)
        print("TABLE 4: PER-SEASON BREAKDOWN (at 5% edge)")
        print("-" * 90)
        header4 = f"{'Season':<12} {'Market|Source':<25} {'Bets':>5} {'Profit':>10} {'ROI':>8}"
        print(header4)
        print("-" * 90)

        for season in sorted(per_season.keys()):
            markets = per_season[season]
            for key in sorted(markets.keys()):
                s = markets[key]
                if s["bets"] == 0:
                    continue
                print(
                    f"{season:<12} {key:<25} "
                    f"{s['bets']:>5} €{s['profit']:>8.0f} {s['roi']:>7.1f}%"
                )

    # --- Summary: best market at each threshold ---
    print("\n" + "-" * 90)
    print("SUMMARY: BEST MARKET PER THRESHOLD (min 20 bets)")
    print("-" * 90)
    for thresh_key in sorted(by_thresh.keys()):
        markets = by_thresh[thresh_key]
        best_key = None
        best_roi = -999
        for key, s in markets.items():
            if s["bets"] >= 20 and s["roi"] > best_roi:
                best_roi = s["roi"]
                best_key = key
        if best_key:
            s = markets[best_key]
            print(f"  {thresh_key}: {best_key:<30} ROI={s['roi']:>+7.1f}% ({s['bets']} bets, €{s['profit']:.0f})")

    print("\n" + "=" * 90)


def print_production_report(results: Dict):
    """Print a clean production simulation report."""
    meta = results["meta"]
    combined = results["combined"]
    by_market = results.get("by_market", {})
    staking = results.get("staking", {})

    print("\n" + "=" * 80)
    print("PRODUCTION SYSTEM BACKTEST")
    print("=" * 80)
    print(f"Seasons: {', '.join(meta['seasons'])}  |  Matches: {meta['total_matches']}")
    print(f"Lineup xG: {meta['matches_with_lineup']} matches")

    # Show active thresholds
    rules = []
    rules.append(f"Draw ≥{meta['draw_min_edge']*100:.0f}%")
    if meta.get('home_min_edge') is not None:
        rules.append(f"Home ≥{meta['home_min_edge']*100:.0f}%")
    if meta.get('away_min_edge') is not None:
        rules.append(f"Away ≥{meta['away_min_edge']*100:.0f}%")
    rules.append(f"O/U ≥{meta['ou_min_edge']*100:.0f}%")
    rules.append(f"Max {meta['max_edge']*100:.0f}%")
    print(f"Rules: {', '.join(rules)}")
    print(f"Situational adjustments: {len(meta['situational_adjustments'])} rules active")

    # Staking info
    if staking:
        mode = staking.get("mode", "flat")
        bankroll_start = staking.get("bankroll_start", 1000)
        bankroll_final = staking.get("bankroll_final", 0)
        if mode == "kelly":
            print(f"Staking: Kelly {staking.get('kelly_fraction', 0.10):.0%} "
                  f"| Bankroll: €{bankroll_start:.0f} → €{bankroll_final:.0f}")
        elif mode == "proportional":
            print(f"Staking: Proportional {staking.get('proportional_pct', 2.0):.1f}% "
                  f"| Bankroll: €{bankroll_start:.0f} → €{bankroll_final:.0f}")
        else:
            print(f"Staking: Flat €{FLAT_STAKE:.0f}")

    print("\n" + "-" * 80)
    print(f"{'Market':<15} {'Bets':>5} {'Wins':>5} {'WinRate':>8} {'Profit':>10} {'ROI':>8} "
          f"{'AvgEdge':>8} {'AvgCLV':>8}")
    print("-" * 80)
    for mkt_key in sorted(by_market.keys()):
        s = by_market[mkt_key]
        label = mkt_key.replace("_", " ")
        if s["bets"] > 0:
            print(f"{label:<15} {s['bets']:>5} {s['wins']:>5} {s['win_rate']:>7.1f}% "
                  f"€{s['profit']:>9.0f} {s['roi']:>+7.1f}% "
                  f"{s['avg_edge']:>+7.2f}% {s['avg_clv']:>+7.3f}%")
    print("-" * 80)
    print(f"{'COMBINED':<15} {combined['bets']:>5} {combined['wins']:>5} {combined['win_rate']:>7.1f}% "
          f"€{combined['profit']:>9.0f} {combined['roi']:>+7.1f}% "
          f"{combined['avg_edge']:>+7.2f}% {combined['avg_clv']:>+7.3f}%")

    print("\n" + "-" * 80)
    print("PER-SEASON BREAKDOWN")
    print("-" * 80)
    for season, data in sorted(results["per_season"].items()):
        print(f"\n  {season}:")
        for mkt_key in sorted(k for k in data.keys() if k != "combined"):
            s = data[mkt_key]
            label = mkt_key.replace("_", " ")
            if s["bets"] > 0:
                print(f"    {label:<14} {s['bets']:>4} bets  €{s['profit']:>8.0f}  ROI={s['roi']:>+6.1f}%")
        s = data["combined"]
        if s["bets"] > 0:
            print(f"    {'Combined':<14} {s['bets']:>4} bets  €{s['profit']:>8.0f}  ROI={s['roi']:>+6.1f}%")

    # Bankroll summary
    if staking and staking.get("bankroll_final"):
        total_staked = sum(b.stake for b in results.get("all_bets", []))
        n_seasons = len(results["per_season"])
        print(f"\n{'='*80}")
        print(f"BANKROLL: €{staking['bankroll_start']:.0f} → €{staking['bankroll_final']:.0f} "
              f"(+{staking['bankroll_growth']:.1f}%)  |  Max Drawdown: {staking['max_drawdown_pct']:.1f}%")
        if n_seasons > 0:
            annual_growth = staking['bankroll_growth'] / n_seasons
            print(f"Annualized growth: +{annual_growth:.1f}%/season  |  "
                  f"Total staked: €{total_staked:.0f}  |  ROI: {combined['roi']:+.1f}%")
        print(f"{'='*80}\n")
    else:
        total_staked = combined["bets"] * FLAT_STAKE
        n_seasons = len(results["per_season"])
        if n_seasons > 0 and total_staked > 0:
            annual_profit = combined["profit"] / n_seasons
            annual_bets = combined["bets"] / n_seasons
            print(f"\n{'='*80}")
            print(f"ANNUALIZED: ~{annual_bets:.0f} bets/season, ~€{annual_profit:.0f}/season profit, "
                  f"ROI={combined['roi']:+.1f}%")
            print(f"Total staked: €{total_staked:.0f} | Total profit: €{combined['profit']:.0f}")
            print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Multi-market backtest with lineup-adjusted xG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seasons", nargs="+", default=["2023-2024", "2024-2025"],
        help="Seasons to backtest (default: 2023-2024 2024-2025)",
    )
    parser.add_argument(
        "--use-lineup-xg", dest="use_lineup_xg", action="store_true", default=True,
        help="Include lineup-adjusted xG comparison (default: on)",
    )
    parser.add_argument(
        "--no-lineup-xg", dest="use_lineup_xg", action="store_false",
        help="Disable lineup-adjusted xG",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Custom save path for results JSON",
    )
    parser.add_argument(
        "--production", action="store_true",
        help="Run production system simulation (all rules applied)",
    )
    parser.add_argument(
        "--staking", choices=["flat", "kelly", "proportional"], default="proportional",
        help="Staking mode for production sim (default: proportional)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Starting bankroll (default: 1000)",
    )
    parser.add_argument(
        "--kelly-fraction", type=float, default=0.10,
        help="Kelly fraction for kelly staking (default: 0.10)",
    )
    parser.add_argument(
        "--prop-pct", type=float, default=2.0,
        help="Proportional stake %% for proportional staking (default: 2.0)",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Use per-fold models for leakage-free out-of-sample evaluation. "
             "Requires fold models (run: python -c 'from ml.ensemble import build_fold_models; build_fold_models()')",
    )
    parser.add_argument(
        "--build-fold-models", action="store_true",
        help="Build per-fold models before running backtest (takes a few minutes)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Build fold models if requested
    if args.build_fold_models:
        log.info("Building per-fold models for walk-forward backtesting...")
        from ml.ensemble import build_fold_models
        build_fold_models()
        log.info("Fold models built successfully.")

    bt = MultiMarketBacktest(
        seasons=args.seasons,
        use_lineup_xg=args.use_lineup_xg,
        walk_forward=args.walk_forward or args.build_fold_models,
    )

    if not bt.load_data():
        log.error("Failed to load data. Exiting.")
        sys.exit(1)

    if args.production:
        results = bt.run_production(
            staking=args.staking,
            bankroll_start=args.bankroll,
            kelly_fraction=args.kelly_fraction,
            proportional_pct=args.prop_pct,
        )
        print_production_report(results)
    else:
        results = bt.run()
        print_report(results)

    # Save results
    save_path = args.save
    if save_path is None:
        results_dir = DATA_DIR / "optimization"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(results_dir / f"multimarket_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    cleaned = clean_for_json(results)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(cleaned, f, indent=2, default=str)

    log.info("Results saved to %s", save_path)


if __name__ == "__main__":
    main()
