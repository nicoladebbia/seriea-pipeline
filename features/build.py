"""Assemble the final ML-ready feature table.

Plugin-based architecture with intermediate caching.

Orchestrates all feature modules (37 steps) via FeaturePlugin subclasses:
  1-8. Team match log, rolling, home/away, xG trends, strength, rest, momentum, derived
  9-20. Match-level features: H2H, Elo, player, referee, aggregates, GK, shots,
        advanced player, advanced shots, Understat, FBref
  21-26. League position, manager (with H2H), congestion, suspensions, formations, derived
  27-31. League-draw, venue, weather, odds, market data
  32. Injury impact features (positional weighting, chemistry disruption)
  33. PPDA/pressing intensity features (Understat)
  34. Manager H2H features (computed in step 22)
  35. January window + transfer impact features
  36. Interaction features (cross weak x strong signals)
  37. Contextual features (kickoff time, day of week, season phase, travel-rest)
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from features.base import STAT_COLUMNS, build_team_match_log
from features.congestion import add_congestion_features
from features.derived import add_derived_features, add_match_level_derived
from features.gk_quality import add_gk_quality_features
from features.h2h import add_h2h_features
from features.home_away import add_home_away_features
from features.league_position import add_league_position_features
from features.manager import add_manager_features
from features.enhanced_momentum import add_momentum_features
from features.player_impact import add_player_impact_features
from features.referee import add_referee_features
from features.rest_days import add_rest_days, pivot_rest_to_match
from features.rolling import add_rolling_features, add_rolling_variance_features
from features.shot_quality import add_shot_quality_features
from features.strength import add_elo_ratings, add_strength_ratings
from features.team_aggregates import add_team_aggregate_features
from features.xg_trends import add_xg_trends
from features.advanced_player import add_advanced_player_features
from features.advanced_shots import add_advanced_shot_features
from config.leagues import get_derbies, get_matchweeks

try:
    from features.player_depth import add_player_depth_features
    HAS_PLAYER_DEPTH = True
except ImportError:
    HAS_PLAYER_DEPTH = False

try:
    from features.lineup_xg import add_lineup_xg_features
    HAS_LINEUP_XG = True
except ImportError:
    HAS_LINEUP_XG = False

try:
    from features.match_patterns import add_match_pattern_features
    HAS_MATCH_PATTERNS = True
except ImportError:
    HAS_MATCH_PATTERNS = False

try:
    from features.creative_factors import add_creative_factors
    HAS_CREATIVE_FACTORS = True
except ImportError:
    HAS_CREATIVE_FACTORS = False

try:
    from features.captain_features import add_captain_features
    HAS_CAPTAIN_FEATURES = True
except ImportError:
    HAS_CAPTAIN_FEATURES = False

try:
    from features.card_timing import add_card_timing_features
    HAS_CARD_TIMING = True
except ImportError:
    HAS_CARD_TIMING = False

try:
    from features.suspensions import compute_suspension_features
    HAS_SUSPENSIONS = True
except ImportError:
    HAS_SUSPENSIONS = False

try:
    from features.formation_analysis import add_formation_features
    HAS_FORMATIONS = True
except ImportError:
    HAS_FORMATIONS = False

try:
    from features.injury_impact import add_injury_features
    HAS_INJURY_IMPACT = True
except ImportError:
    HAS_INJURY_IMPACT = False

try:
    from features.pressing import add_ppda_features
    HAS_PRESSING = True
except ImportError:
    HAS_PRESSING = False

try:
    from features.transfer_impact_analysis import add_transfer_impact_features
    HAS_TRANSFER_IMPACT = True
except ImportError:
    HAS_TRANSFER_IMPACT = False

try:
    from features.odds_features import add_odds_derived_features
    HAS_ODDS_FEATURES = True
except ImportError:
    HAS_ODDS_FEATURES = False
from features.understat_features import add_understat_features, add_understat_rolling_features
from features.fbref_features import add_fbref_features
try:
    from features.sofascore_features import add_sofascore_features
    HAS_SOFASCORE = True
except ImportError:
    HAS_SOFASCORE = False
try:
    from features.sofascore_indices import compute_sofascore_indices
    HAS_SOFASCORE_INDICES = True
except ImportError:
    HAS_SOFASCORE_INDICES = False
try:
    from features.substitution_features import add_substitution_features
    HAS_SUBSTITUTION = True
except ImportError:
    HAS_SUBSTITUTION = False
from storage.paths import features_path, parsed_path

log = logging.getLogger(__name__)

# Derby/rivalry definitions sourced from the central league registry.
# Default to Serie A; will be parameterized per-league in future phases.
SERIE_A_DERBIES = get_derbies("serie_a")


# ────────────────────────────────────────────────────────────────────────────
#  Plugin Architecture: FeatureState, FeaturePlugin, FeaturePipeline
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class FeatureState:
    """Holds all intermediate data flowing through the feature pipeline.

    Plugins read from and write to this shared state object.
    The pipeline transitions from team-level (team_log) to match-level
    (feature_df) at the pivot step (Step 9).
    """
    matches: pd.DataFrame
    season: Optional[str] = None
    team_log: Optional[pd.DataFrame] = None
    feature_df: Optional[pd.DataFrame] = None

    # Track column counts for logging
    _cols_before: set = field(default_factory=set, repr=False)


class FeaturePlugin(ABC):
    """Base class for all feature engineering plugins.

    Each plugin encapsulates a single feature engineering step.
    Subclasses must set `name`, `version`, and `dependencies`,
    and implement `apply()`.
    """
    name: str = ""
    version: str = "1.0"
    dependencies: List[str] = []

    # If True, errors are caught and logged rather than raised.
    # Critical plugins (e.g. team_match_log, pivot) should set this to False.
    non_critical: bool = False

    @abstractmethod
    def apply(self, state: FeatureState) -> FeatureState:
        """Apply this feature engineering step to the pipeline state."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} v{self.version}>"


class FeaturePipeline:
    """Orchestrates plugin execution with dependency resolution and caching.

    Plugins are registered and then executed in dependency order.
    Each plugin's output can be cached to parquet for fast re-runs.
    """

    def __init__(self, cache_dir: Optional[Path] = None, league: Optional[str] = None):
        from config.settings import DATA_DIR
        self._plugins: List[FeaturePlugin] = []
        self._plugin_map: dict[str, FeaturePlugin] = {}
        self._league = league
        base_cache = cache_dir or (DATA_DIR / "cache" / "features")
        # Per-league cache isolation prevents cross-league contamination
        self._cache_dir = base_cache / league if league else base_cache

    def register(self, plugin: FeaturePlugin) -> None:
        """Register a plugin for execution."""
        if plugin.name in self._plugin_map:
            raise ValueError(f"Duplicate plugin name: {plugin.name!r}")
        self._plugins.append(plugin)
        self._plugin_map[plugin.name] = plugin

    def _resolve_order(self) -> List[FeaturePlugin]:
        """Topological sort of plugins by dependencies.

        Falls back to registration order if no dependency issues.
        Raises ValueError on missing or circular dependencies.
        """
        # Build adjacency
        visited: dict[str, int] = {}  # 0=unvisited, 1=in-progress, 2=done
        order: list[str] = []

        for p in self._plugins:
            visited[p.name] = 0

        def visit(name: str) -> None:
            if visited.get(name, -1) == 2:
                return
            if visited.get(name, -1) == 1:
                raise ValueError(f"Circular dependency detected involving {name!r}")
            if name not in self._plugin_map:
                raise ValueError(f"Missing dependency: {name!r}")
            visited[name] = 1
            plugin = self._plugin_map[name]
            for dep in plugin.dependencies:
                visit(dep)
            visited[name] = 2
            order.append(name)

        for p in self._plugins:
            if visited[p.name] == 0:
                visit(p.name)

        return [self._plugin_map[n] for n in order]

    def _cache_path(self, plugin: FeaturePlugin, suffix: str = "") -> Path:
        """Return the cache file path for a plugin's output."""
        tag = f"{plugin.name}_v{plugin.version}"
        if suffix:
            tag += f"_{suffix}"
        return self._cache_dir / f"{tag}.parquet"

    def _source_fingerprint(self, plugin: FeaturePlugin) -> str:
        """Return a hash of the plugin's apply() source code.

        Used to detect if the plugin logic changed even without a
        version bump.
        """
        try:
            src = inspect.getsource(plugin.apply)
            return hashlib.md5(src.encode()).hexdigest()[:12]
        except (OSError, TypeError):
            return "unknown"

    def _is_cache_valid(self, plugin: FeaturePlugin) -> bool:
        """Check if a cached result exists and is still valid.

        A cache file is valid if:
          1. It exists on disk
          2. The plugin version matches (encoded in filename)
          3. The plugin source code hasn't changed (fingerprint check)
        """
        cache_file = self._cache_path(plugin)
        if not cache_file.exists():
            return False

        # Check source fingerprint — detect code changes without version bumps
        fp_file = cache_file.with_suffix(".fingerprint")
        current_fp = self._source_fingerprint(plugin)
        if fp_file.exists():
            stored_fp = fp_file.read_text().strip()
            if stored_fp != current_fp:
                log.info("Cache invalidated for %s: source code changed (was %s, now %s)",
                         plugin.name, stored_fp[:8], current_fp[:8])
                return False
        return True

    def _save_cache(self, plugin: FeaturePlugin, df: pd.DataFrame,
                    suffix: str = "") -> None:
        """Save a DataFrame and source fingerprint to the plugin's cache file."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(plugin, suffix)
        from config.settings import atomic_write_parquet
        atomic_write_parquet(path, df, index=False)
        # Save source fingerprint alongside cache for invalidation
        fp_file = path.with_suffix(".fingerprint")
        fp_file.write_text(self._source_fingerprint(plugin))
        log.debug("Cached %s → %s (%d rows x %d cols)",
                  plugin.name, path.name, len(df), len(df.columns))

    def _load_cache(self, plugin: FeaturePlugin,
                    suffix: str = "") -> pd.DataFrame:
        """Load a cached DataFrame for a plugin."""
        path = self._cache_path(plugin, suffix)
        df = pd.read_parquet(path)
        log.debug("Loaded cache %s ← %s (%d rows x %d cols)",
                  plugin.name, path.name, len(df), len(df.columns))
        return df

    def build(self, state: FeatureState,
              use_cache: bool = True) -> FeatureState:
        """Execute all registered plugins in dependency order.

        Args:
            state: The initial FeatureState with matches loaded.
            use_cache: If True, skip plugins with valid cache.
                       If False, force rebuild everything.

        Returns:
            The final FeatureState with all features computed.
        """
        plugins = self._resolve_order()
        total = len(plugins)
        completed = 0
        cache_hits = 0
        cache_misses = 0

        log.info("Feature pipeline: %d plugins registered", total)

        for i, plugin in enumerate(plugins, 1):
            step_start = time.perf_counter()

            # Determine which DataFrame this plugin produces
            # Team-level plugins (before pivot) produce team_log
            # Match-level plugins (after pivot) produce feature_df
            is_team_level = plugin.name in {
                "team_match_log", "rolling_stats", "home_away_splits",
                "xg_trends", "strength_ratings", "rest_days",
                "momentum_streaks", "derived_team_features",
            }
            # Pivot and special plugins are never cached
            never_cache = plugin.name in {
                "pivot_to_match_level", "backfill_managers",
                "backfill_referees", "odds", "market_data",
                "manager_h2h_noop",
            }

            # Check cache
            cache_hit = False
            if use_cache and not never_cache and self._is_cache_valid(plugin):
                try:
                    cached_df = self._load_cache(plugin)
                    if is_team_level:
                        state.team_log = cached_df
                    else:
                        state.feature_df = cached_df
                    cache_hit = True
                    cache_hits += 1
                except Exception as e:
                    log.warning("Cache load failed for %s: %s — rebuilding",
                                plugin.name, e)
                    cache_hit = False

            if not cache_hit:
                cache_misses += 1
                plugin_succeeded = True
                # Execute the plugin
                if plugin.non_critical:
                    try:
                        state = plugin.apply(state)
                    except Exception as e:
                        log.warning(
                            "Plugin %s failed (non-critical): %s",
                            plugin.name, e,
                        )
                        plugin_succeeded = False
                else:
                    state = plugin.apply(state)

                # Save to cache ONLY if plugin succeeded (don't cache failure states)
                if not never_cache and plugin_succeeded:
                    try:
                        df_to_cache = (
                            state.team_log if is_team_level
                            else state.feature_df
                        )
                        if df_to_cache is not None:
                            self._save_cache(plugin, df_to_cache)
                    except Exception as e:
                        log.debug("Cache save failed for %s: %s", plugin.name, e)

            elapsed = time.perf_counter() - step_start
            status = "CACHE HIT" if cache_hit else "computed"
            completed += 1
            log.info(
                "Step %d/%d: %-35s [%s] %.2fs",
                i, total, plugin.name, status, elapsed,
            )

        log.info(
            "Feature pipeline complete: %d/%d steps, %d cache hits, %d computed",
            completed, total, cache_hits, cache_misses,
        )
        return state


# ────────────────────────────────────────────────────────────────────────────
#  Plugin Implementations — Steps 1-37
# ────────────────────────────────────────────────────────────────────────────


class BackfillManagersPlugin(FeaturePlugin):
    """Pre-step: Backfill manager names from static lookup."""
    name = "backfill_managers"
    version = "1.0"
    dependencies = []
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        from config.managers import backfill_managers
        state.matches = backfill_managers(state.matches)
        return state


class BackfillRefereesPlugin(FeaturePlugin):
    """Pre-step: Backfill referee names from scraped worldfootball.net data."""
    name = "backfill_referees"
    version = "1.0"
    dependencies = ["backfill_managers"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        from scraper.referee import merge_referee_with_matches
        state.matches = merge_referee_with_matches(state.matches)
        return state


class Step01TeamMatchLog(FeaturePlugin):
    """Step 1: Build team match log from matches."""
    name = "team_match_log"
    version = "1.0"
    dependencies = ["backfill_managers", "backfill_referees"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.team_log = build_team_match_log(state.matches)
        return state


class Step02RollingStats(FeaturePlugin):
    """Step 2: Rolling averages."""
    name = "rolling_stats"
    version = "1.0"
    dependencies = ["team_match_log"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.team_log = add_rolling_features(state.team_log)
        state.team_log = add_rolling_variance_features(state.team_log)
        return state


class Step03HomeAwaySplits(FeaturePlugin):
    """Step 3: Home/away splits."""
    name = "home_away_splits"
    version = "1.0"
    dependencies = ["rolling_stats"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.team_log = add_home_away_features(state.team_log)
        return state


class Step04XgTrends(FeaturePlugin):
    """Step 4: xG trends."""
    name = "xg_trends"
    version = "1.0"
    dependencies = ["home_away_splits"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.team_log = add_xg_trends(state.team_log)
        return state


class Step05StrengthRatings(FeaturePlugin):
    """Step 5: Strength ratings."""
    name = "strength_ratings"
    version = "1.0"
    dependencies = ["xg_trends"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.team_log = add_strength_ratings(state.team_log)
        return state


class Step06RestDays(FeaturePlugin):
    """Step 6: Rest days."""
    name = "rest_days"
    version = "1.0"
    dependencies = ["strength_ratings"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.team_log = add_rest_days(state.team_log)
        return state


class Step07MomentumStreaks(FeaturePlugin):
    """Step 7: Momentum and streaks."""
    name = "momentum_streaks"
    version = "1.0"
    dependencies = ["rest_days"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.team_log = add_momentum_features(state.team_log)
        return state


class Step08DerivedTeamFeatures(FeaturePlugin):
    """Step 8: Derived team-level features (opponent-adjusted, form, GD momentum)."""
    name = "derived_team_features"
    version = "1.0"
    dependencies = ["momentum_streaks"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.team_log = add_derived_features(state.team_log)
        return state


class Step09PivotToMatchLevel(FeaturePlugin):
    """Step 9: Pivot team log back to match level."""
    name = "pivot_to_match_level"
    version = "1.0"
    dependencies = ["derived_team_features"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _pivot_to_match_level(state.matches, state.team_log)
        return state


class Step10H2H(FeaturePlugin):
    """Step 10: Head-to-head features."""
    name = "h2h"
    version = "1.0"
    dependencies = ["pivot_to_match_level"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_h2h_features(state.feature_df)
        return state


class Step11Elo(FeaturePlugin):
    """Step 11: Elo ratings."""
    name = "elo_ratings"
    version = "1.0"
    dependencies = ["h2h"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_elo_ratings(state.feature_df)
        return state


class Step12PlayerImpact(FeaturePlugin):
    """Step 12: Player impact features."""
    name = "player_impact"
    version = "1.0"
    dependencies = ["elo_ratings"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        _cols_before = set(state.feature_df.columns)
        state.feature_df = add_player_impact_features(state.feature_df)
        _new_cols = set(state.feature_df.columns) - _cols_before
        log.info("  -> Player impact: added %d columns", len(_new_cols))
        return state


class Step13Referee(FeaturePlugin):
    """Step 13: Referee tendency features."""
    name = "referee"
    version = "1.0"
    dependencies = ["player_impact"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_referee_features(state.feature_df)
        return state


class Step14TeamAggregates(FeaturePlugin):
    """Step 14: Team-level aggregated player stats."""
    name = "team_aggregates"
    version = "1.0"
    dependencies = ["referee"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        _cols_before = set(state.feature_df.columns)
        state.feature_df = add_team_aggregate_features(state.feature_df)
        _new_cols = set(state.feature_df.columns) - _cols_before
        log.info("  -> Team aggregates: added %d columns", len(_new_cols))
        return state


class Step15GkQuality(FeaturePlugin):
    """Step 15: GK quality features."""
    name = "gk_quality"
    version = "1.0"
    dependencies = ["team_aggregates"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_gk_quality_features(state.feature_df)
        return state


class Step16ShotQuality(FeaturePlugin):
    """Step 16: Shot quality features."""
    name = "shot_quality"
    version = "1.0"
    dependencies = ["gk_quality"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_shot_quality_features(state.feature_df)
        return state


class Step17AdvancedPlayer(FeaturePlugin):
    """Step 17: Advanced player-level ratio & dependency features."""
    name = "advanced_player"
    version = "1.0"
    dependencies = ["shot_quality"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        _cols_before = set(state.feature_df.columns)
        state.feature_df = add_advanced_player_features(state.feature_df)
        _new_cols = set(state.feature_df.columns) - _cols_before
        log.info("  -> Advanced player: added %d columns", len(_new_cols))
        return state


class Step18AdvancedShots(FeaturePlugin):
    """Step 18: Advanced shot creation chain & finishing features."""
    name = "advanced_shots"
    version = "1.0"
    dependencies = ["advanced_player"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_advanced_shot_features(state.feature_df)
        return state


class Step19UnderstatXg(FeaturePlugin):
    """Step 19: Understat xG features (reliable historical xG from 2014+)."""
    name = "understat_xg"
    version = "1.0"
    dependencies = ["advanced_shots"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        df = state.feature_df
        _xg_cols_before = set(c for c in df.columns if "xg" in c.lower())
        df = add_understat_features(df)
        df = add_understat_rolling_features(df)
        _xg_cols_after = set(c for c in df.columns if "xg" in c.lower())
        new_xg_cols = _xg_cols_after - _xg_cols_before
        if new_xg_cols:
            sample_col = next(iter(new_xg_cols))
            df["has_xg_data"] = df[sample_col].notna().astype(int)
            log.info("Added has_xg_data flag (%d rows with xG data)",
                     df["has_xg_data"].sum())
        else:
            df["has_xg_data"] = 0
            log.info("Added has_xg_data flag (no Understat xG columns found)")
        state.feature_df = df
        return state


class Step19bSofascore(FeaturePlugin):
    """Step 19b: Sofascore per-match player features (rolling xG, xA, etc.)."""
    name = "sofascore"
    version = "1.0"
    dependencies = ["understat_xg"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_SOFASCORE:
            log.info("Skipping Sofascore features (module not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = add_sofascore_features(state.feature_df)
        _cols_after = len(state.feature_df.columns)
        log.info("  Sofascore added %d new columns", _cols_after - _cols_before)
        return state


class Step19b2SofascoreIndices(FeaturePlugin):
    """Step 19b2: Sofascore domain composite indices (rank-based aggregation)."""
    name = "sofascore_indices"
    version = "1.0"
    dependencies = ["sofascore"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not (HAS_SOFASCORE and HAS_SOFASCORE_INDICES):
            log.info("Skipping Sofascore indices (dependencies not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = compute_sofascore_indices(state.feature_df)
        log.info("  Sofascore indices added %d new columns",
                 len(state.feature_df.columns) - _cols_before)
        return state


class Step19cPlayerDepth(FeaturePlugin):
    """Step 19c: Tier 6 player depth features (variance, dependency, fatigue, chemistry)."""
    name = "player_depth"
    version = "1.0"
    dependencies = ["sofascore"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not (HAS_PLAYER_DEPTH and HAS_SOFASCORE):
            log.info("Skipping player depth features (dependencies not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = add_player_depth_features(state.feature_df)
        log.info("  Player depth added %d new columns",
                 len(state.feature_df.columns) - _cols_before)
        return state


class Step19c2LineupXg(FeaturePlugin):
    """Step 19c2: Historical lineup xG reconstruction (Sofascore + Understat)."""
    name = "lineup_xg"
    version = "1.0"
    dependencies = ["player_depth"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_LINEUP_XG:
            log.info("Skipping lineup xG features (module not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = add_lineup_xg_features(state.feature_df)
        log.info("  Lineup xG added %d new columns",
                 len(state.feature_df.columns) - _cols_before)
        return state


class Step19dMatchPatterns(FeaturePlugin):
    """Step 19d: Tier 7-9 match pattern features (H2H, comeback, late goals, set pieces)."""
    name = "match_patterns"
    version = "1.0"
    dependencies = ["player_depth"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_MATCH_PATTERNS:
            log.info("Skipping match pattern features (module not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = add_match_pattern_features(state.feature_df)
        log.info("  Match patterns added %d new columns",
                 len(state.feature_df.columns) - _cols_before)
        return state


class Step19eCreativeFactors(FeaturePlugin):
    """Step 19e: Tier 10-11 creative factor features (timing, promoted, motivation, calendar)."""
    name = "creative_factors"
    version = "1.0"
    dependencies = ["match_patterns"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_CREATIVE_FACTORS:
            log.info("Skipping creative factors (module not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = add_creative_factors(state.feature_df)
        log.info("  Creative factors added %d new columns",
                 len(state.feature_df.columns) - _cols_before)
        return state


class Step19fCaptain(FeaturePlugin):
    """Step 19f: Captain features (captain played, consistency, win rate effect)."""
    name = "captain_features"
    version = "1.0"
    dependencies = ["creative_factors"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_CAPTAIN_FEATURES:
            log.info("Skipping captain features (module not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = add_captain_features(state.feature_df)
        log.info("  Captain features added %d new columns",
                 len(state.feature_df.columns) - _cols_before)
        return state


class Step19gCardTiming(FeaturePlugin):
    """Step 19g: Card/goal timing features (actual minute-level patterns)."""
    name = "card_timing"
    version = "1.0"
    dependencies = ["captain_features"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_CARD_TIMING:
            log.info("Skipping card timing features (module not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = add_card_timing_features(state.feature_df)
        log.info("  Card timing features added %d new columns",
                 len(state.feature_df.columns) - _cols_before)
        return state


class Step20Fbref(FeaturePlugin):
    """Step 20: FBref team aggregated stats (shooting, passing, defense, etc.)."""
    name = "fbref"
    version = "1.0"
    dependencies = ["card_timing"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_fbref_features(state.feature_df)
        return state


class Step21LeaguePosition(FeaturePlugin):
    """Step 21: League position & motivation."""
    name = "league_position"
    version = "1.0"
    dependencies = ["fbref"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_league_position_features(state.feature_df)
        return state


class Step21bLeaguePositionContext(FeaturePlugin):
    """Step 21b: League position context (CL zone gap, relegation gap, position zone)."""
    name = "league_position_context"
    version = "1.0"
    dependencies = ["league_position"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_league_position_context(state.feature_df)
        return state


class Step22Manager(FeaturePlugin):
    """Step 22: Manager tenure features."""
    name = "manager"
    version = "1.0"
    dependencies = ["league_position_context"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_manager_features(state.feature_df)
        return state


class Step23Congestion(FeaturePlugin):
    """Step 23: Fixture congestion features."""
    name = "congestion"
    version = "1.0"
    dependencies = ["manager"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_congestion_features(state.feature_df)
        return state


class Step24Suspensions(FeaturePlugin):
    """Step 24: Suspension features (yellow card accumulation, at-risk players)."""
    name = "suspensions"
    version = "1.0"
    dependencies = ["congestion"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_SUSPENSIONS:
            log.info("Skipping suspension features (module not available)")
            return state
        state.feature_df = compute_suspension_features(state.feature_df)
        return state


class Step25Formations(FeaturePlugin):
    """Step 25: Formation features (tactical matchup analysis)."""
    name = "formations"
    version = "1.0"
    dependencies = ["suspensions"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_FORMATIONS:
            log.info("Skipping formation features (module not available)")
            return state
        state.feature_df = add_formation_features(state.feature_df)
        return state


class Step26DerivedMatchLevel(FeaturePlugin):
    """Step 26: Derived match-level features (HT patterns, points pace, relative diffs)."""
    name = "derived_match_level"
    version = "1.0"
    dependencies = ["formations"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = add_match_level_derived(state.feature_df)
        return state


class Step26bDerby(FeaturePlugin):
    """Step 26b: Derby/rivalry features."""
    name = "derby"
    version = "1.0"
    dependencies = ["derived_match_level"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_derby_features(state.feature_df)
        return state


class Step26cGoalFeatures(FeaturePlugin):
    """Step 26c: Goal-specific features for O/U and BTTS prediction."""
    name = "goal_features"
    version = "1.0"
    dependencies = ["derby"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        from features.goal_features import add_goal_features
        state.feature_df = add_goal_features(state.feature_df)
        return state


class Step27LeagueDrawFeatures(FeaturePlugin):
    """Step 27: League-aware and draw-specific features."""
    name = "league_draw"
    version = "1.0"
    dependencies = ["goal_features"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_league_draw_features(state.feature_df)
        return state


class Step28Venue(FeaturePlugin):
    """Step 28: Venue features (travel distance, altitude, capacity)."""
    name = "venue"
    version = "1.0"
    dependencies = ["league_draw"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_venue_features(state.feature_df)
        return state


class Step29Weather(FeaturePlugin):
    """Step 29: Weather data (external or estimated)."""
    name = "weather"
    version = "1.0"
    dependencies = ["venue"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_weather(state.feature_df)
        return state


class Step30Odds(FeaturePlugin):
    """Step 30: Betting odds (external)."""
    name = "odds"
    version = "1.0"
    dependencies = ["weather"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_odds(state.feature_df, state.season)
        return state


class Step30bDerivedOdds(FeaturePlugin):
    """Step 30b: Derived odds features (sharp-soft divergence, AH, goal totals)."""
    name = "derived_odds"
    version = "1.0"
    dependencies = ["odds"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_ODDS_FEATURES:
            log.info("Skipping derived odds features (module not available)")
            return state
        _cols_before = len(state.feature_df.columns)
        state.feature_df = add_odds_derived_features(state.feature_df)
        log.info("  -> Derived odds: added %d columns",
                 len(state.feature_df.columns) - _cols_before)
        return state


class Step31MarketData(FeaturePlugin):
    """Step 31: Squad market values & transfers (external)."""
    name = "market_data"
    version = "1.0"
    dependencies = ["derived_odds"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_market_data(state.feature_df, state.season)
        return state


class Step32InjuryImpact(FeaturePlugin):
    """Step 32: Injury impact features."""
    name = "injury_impact"
    version = "1.0"
    dependencies = ["market_data"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_INJURY_IMPACT:
            log.info("Skipping injury impact features (module not available)")
            return state
        state.feature_df = add_injury_features(state.feature_df)
        return state


class Step33PPDA(FeaturePlugin):
    """Step 33: PPDA / pressing features."""
    name = "ppda"
    version = "1.0"
    dependencies = ["injury_impact"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_PRESSING:
            log.info("Skipping PPDA features (module not available)")
            return state
        state.feature_df = add_ppda_features(state.feature_df)
        return state


class Step34ManagerH2HNoop(FeaturePlugin):
    """Step 34: Manager H2H features (already computed in Step 22)."""
    name = "manager_h2h_noop"
    version = "1.0"
    dependencies = ["ppda"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        log.info("Manager H2H features (computed in Step 22)")
        return state


class Step35TransferImpact(FeaturePlugin):
    """Step 35: January window + transfer impact features."""
    name = "transfer_impact"
    version = "1.0"
    dependencies = ["manager_h2h_noop"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_TRANSFER_IMPACT:
            log.info("Skipping transfer impact features (module not available)")
            return state
        state.feature_df = add_transfer_impact_features(state.feature_df)
        return state


class Step36Interactions(FeaturePlugin):
    """Step 36: Interaction features -- amplify weak signals via strong ones."""
    name = "interactions"
    version = "1.0"
    dependencies = ["transfer_impact"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_interaction_features(state.feature_df)
        return state


class Step37Contextual(FeaturePlugin):
    """Step 37: Contextual features (kickoff time, day of week, season phase, etc.)."""
    name = "contextual"
    version = "1.0"
    dependencies = ["interactions"]
    non_critical = False

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = _add_contextual_features(state.feature_df)
        return state


class Step38SubstitutionPatterns(FeaturePlugin):
    """Step 38: Substitution pattern features (avg subs/game, avg sub minute, first sub minute).

    Coverage is 0% for historical rows and will increase as live matchday
    data accumulates in data/lineup_history/substitutions.json.
    non_critical=True so a missing ledger never breaks the pipeline.
    """
    name = "substitution_patterns"
    version = "1.0"
    dependencies = ["contextual"]
    non_critical = True

    def apply(self, state: FeatureState) -> FeatureState:
        if not HAS_SUBSTITUTION:
            log.info("Skipping substitution pattern features (module not available)")
            return state
        state.feature_df = add_substitution_features(state.feature_df)
        return state


# ────────────────────────────────────────────────────────────────────────────
#  Pipeline Assembly & Public API
# ────────────────────────────────────────────────────────────────────────────


def _create_pipeline(league: Optional[str] = None) -> FeaturePipeline:
    """Create the feature pipeline and register all 38 step plugins."""
    pipeline = FeaturePipeline(league=league)

    # Pre-steps
    pipeline.register(BackfillManagersPlugin())
    pipeline.register(BackfillRefereesPlugin())

    # Steps 1-8: team-level
    pipeline.register(Step01TeamMatchLog())
    pipeline.register(Step02RollingStats())
    pipeline.register(Step03HomeAwaySplits())
    pipeline.register(Step04XgTrends())
    pipeline.register(Step05StrengthRatings())
    pipeline.register(Step06RestDays())
    pipeline.register(Step07MomentumStreaks())
    pipeline.register(Step08DerivedTeamFeatures())

    # Step 9: pivot
    pipeline.register(Step09PivotToMatchLevel())

    # Steps 10-20: match-level core features
    pipeline.register(Step10H2H())
    pipeline.register(Step11Elo())
    pipeline.register(Step12PlayerImpact())
    pipeline.register(Step13Referee())
    pipeline.register(Step14TeamAggregates())
    pipeline.register(Step15GkQuality())
    pipeline.register(Step16ShotQuality())
    pipeline.register(Step17AdvancedPlayer())
    pipeline.register(Step18AdvancedShots())
    pipeline.register(Step19UnderstatXg())
    pipeline.register(Step19bSofascore())
    pipeline.register(Step19b2SofascoreIndices())
    pipeline.register(Step19cPlayerDepth())
    pipeline.register(Step19c2LineupXg())
    pipeline.register(Step19dMatchPatterns())
    pipeline.register(Step19eCreativeFactors())
    pipeline.register(Step19fCaptain())
    pipeline.register(Step19gCardTiming())
    pipeline.register(Step20Fbref())

    # Steps 21-26: league context + manager + formations + derived
    pipeline.register(Step21LeaguePosition())
    pipeline.register(Step21bLeaguePositionContext())
    pipeline.register(Step22Manager())
    pipeline.register(Step23Congestion())
    pipeline.register(Step24Suspensions())
    pipeline.register(Step25Formations())
    pipeline.register(Step26DerivedMatchLevel())
    pipeline.register(Step26bDerby())
    pipeline.register(Step26cGoalFeatures())

    # Steps 27-31: external data
    pipeline.register(Step27LeagueDrawFeatures())
    pipeline.register(Step28Venue())
    pipeline.register(Step29Weather())
    pipeline.register(Step30Odds())
    pipeline.register(Step30bDerivedOdds())
    pipeline.register(Step31MarketData())

    # Steps 32-38: advanced + interaction + contextual + substitution patterns
    pipeline.register(Step32InjuryImpact())
    pipeline.register(Step33PPDA())
    pipeline.register(Step34ManagerH2HNoop())
    pipeline.register(Step35TransferImpact())
    pipeline.register(Step36Interactions())
    pipeline.register(Step37Contextual())
    pipeline.register(Step38SubstitutionPatterns())

    return pipeline


def _build_features_for_matches(matches: pd.DataFrame,
                                season: str | None = None,
                                use_cache: bool = True,
                                league: str | None = None) -> pd.DataFrame:
    """Build features for a single league's matches.

    This is the core feature-building pipeline extracted from build_features()
    so it can be called per-league for multi-league data.
    """
    log.info("Building features for %d matches (league=%s)", len(matches), league or "all")

    # Create pipeline and state
    pipeline = _create_pipeline(league=league)
    state = FeatureState(matches=matches, season=season)

    # Execute the full pipeline
    state = pipeline.build(state, use_cache=use_cache)

    feature_df = state.feature_df

    # ── Feature validation: check domain constraints ──────────────
    feature_df = _validate_features(feature_df)

    # ── Column cleanup: drop garbage before returning ──────────────
    pre_clean = len(feature_df.columns)

    # 1. Drop Unnamed / level_0 artifacts from HTML header parsing
    garbage_cols = [c for c in feature_df.columns
                    if "Unnamed" in c or "level_0" in c]
    if garbage_cols:
        feature_df = feature_df.drop(columns=garbage_cols)
        log.info("Dropped %d Unnamed/level_0 artifact columns:", len(garbage_cols))
        for col in garbage_cols[:50]:
            log.info("  [artifact] %s", col)

    # 2. Drop constant (zero-variance) columns — useless for ML
    nunique = feature_df.nunique(dropna=True)
    constant_cols = [c for c in nunique.index
                     if nunique[c] <= 1 and c not in {"result", "league"}]
    if constant_cols:
        feature_df = feature_df.drop(columns=constant_cols)
        log.info("Dropped %d constant/zero-variance columns:", len(constant_cols))
        for col in constant_cols[:50]:
            log.info("  [constant] %s", col)

    # 3. Drop columns with >95% null — no signal, just noise
    null_pct = feature_df.isnull().mean()
    high_null_cols = [c for c in null_pct.index
                      if null_pct[c] > 0.95 and c in feature_df.columns]
    if high_null_cols:
        feature_df = feature_df.drop(columns=high_null_cols)
        log.info("Dropped %d columns with >95%% null values:", len(high_null_cols))
        for col in high_null_cols[:50]:
            log.info("  [high-null %.1f%%] %s", null_pct[col] * 100, col)

    # 4. Drop exact-duplicate columns (|r| >= 0.999)
    numeric_cols = [c for c in feature_df.columns
                    if feature_df[c].dtype in ("float64", "float32", "int64", "int32")]
    if len(numeric_cols) > 1:
        dup_to_drop = set()
        # Use dropna instead of fillna(0) — filling NaN with 0 inflates
        # correlations between columns with shared missingness patterns
        sample = feature_df[numeric_cols].dropna(axis=0, how="any")
        if len(sample) < 200:
            # Not enough complete rows; fall back to pairwise correlation
            sample = feature_df[numeric_cols]
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        corr = sample.corr().abs()
        for i in range(len(numeric_cols)):
            if numeric_cols[i] in dup_to_drop:
                continue
            for j in range(i + 1, len(numeric_cols)):
                if numeric_cols[j] in dup_to_drop:
                    continue
                if corr.iloc[i, j] >= 0.999:
                    dup_to_drop.add(numeric_cols[j])
        if dup_to_drop:
            feature_df = feature_df.drop(columns=list(dup_to_drop))
            log.info("Dropped %d exact-duplicate columns (|r| >= 0.999):", len(dup_to_drop))
            for col in sorted(dup_to_drop)[:30]:
                log.info("  [exact-dup] %s", col)

    # 5. Sanitize column names — XGBoost rejects <, >, [, ] in feature names
    bad_chars = {"<": "_lt_", ">": "_gt_", "[": "_lb_", "]": "_rb_"}
    rename_map = {}
    for col in feature_df.columns:
        new_col = col
        for ch, repl in bad_chars.items():
            new_col = new_col.replace(ch, repl)
        if new_col != col:
            rename_map[col] = new_col
    if rename_map:
        feature_df = feature_df.rename(columns=rename_map)
        log.info("Sanitized %d column names (removed <, >, [, ] chars):", len(rename_map))
        for old, new in sorted(rename_map.items())[:20]:
            log.info("  %s -> %s", old, new)

    log.info("Column cleanup: %d -> %d columns (dropped %d total)",
             pre_clean, len(feature_df.columns), pre_clean - len(feature_df.columns))

    # Add target columns
    feature_df["result"] = feature_df.apply(_match_result, axis=1)

    return feature_df


def build_features(season: str | None = None,
                   use_cache: bool = True) -> pd.DataFrame:
    """Build the complete ML feature table.

    Returns a DataFrame with one row per match and ~400+ feature columns.
    If multi-league data is present, builds features per-league to keep
    strength ratings, league position, and Elo scoped correctly.

    Args:
        season: Optional season filter (e.g. "2024-2025").
        use_cache: If True (default), use intermediate caching.
                   Set to False to force full rebuild.
    """
    # Load matches
    matches_path = parsed_path("matches")
    if not matches_path.exists():
        raise FileNotFoundError(
            f"matches.parquet not found at {matches_path}. Run 'parse' first."
        )

    matches = pd.read_parquet(matches_path)
    # Deduplicate matches to prevent double-counting (multiple data sources can create dupes)
    before = len(matches)
    matches = matches.drop_duplicates(subset=["home_team", "away_team", "match_date", "season"], keep="first")
    if len(matches) < before:
        log.warning("Dropped %d duplicate matches from matches.parquet", before - len(matches))
    if season:
        matches = matches[matches["season"] == season].copy()

    if matches.empty:
        log.warning("No matches found%s", f" for season {season}" if season else "")
        return pd.DataFrame()

    # Ensure league column exists
    if "league" not in matches.columns:
        matches["league"] = "serie_a"
    if "league_name" not in matches.columns:
        matches["league_name"] = "Serie A"

    # If multi-league data, build features per-league then combine
    leagues = matches["league"].unique()
    if len(leagues) > 1:
        log.info("Multi-league data detected: %s. Building features per-league.",
                 list(leagues))
        all_features = []
        for league in leagues:
            league_matches = matches[matches["league"] == league].copy()
            log.info("--- Building features for %s (%d matches) ---",
                     league, len(league_matches))
            features = _build_features_for_matches(
                league_matches, season, use_cache=use_cache, league=league
            )
            if not features.empty:
                # Save per-league parquet (source of truth for per-league training)
                league_out = features_path(league=league)
                features.to_parquet(league_out, index=False)
                log.info("Per-league features saved: %s (%d rows x %d cols)",
                         league_out, len(features), len(features.columns))
                all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        feature_df = pd.concat(all_features, ignore_index=True)

        # Sort by date across all leagues
        feature_df["_sort_date"] = pd.to_datetime(
            feature_df["match_date"], errors="coerce"
        )
        feature_df = feature_df.sort_values("_sort_date").reset_index(drop=True)
        feature_df.drop(columns=["_sort_date"], inplace=True)
    else:
        feature_df = _build_features_for_matches(
            matches, season, use_cache=use_cache
        )

    # Tag columns: mark which are safe for ML input vs metadata/leakage
    ml_cols = get_ml_feature_columns(feature_df)
    log.info("ML-safe feature columns: %d / %d total",
             len(ml_cols), len(feature_df.columns))

    # Save
    out_path = features_path()
    feature_df.to_parquet(out_path, index=False)
    log.info(
        "Feature table saved: %d rows x %d columns at %s",
        len(feature_df),
        len(feature_df.columns),
        out_path,
    )

    return feature_df


# ────────────────────────────────────────────────────────────────────────────
#  Helper Functions (unchanged from original)
# ────────────────────────────────────────────────────────────────────────────


def get_ml_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are safe for ML model input.

    Excludes:
      - Identifiers: match_id, season, match_date, kickoff_time, data_source
      - Team names: home_team, away_team
      - Venue/metadata: venue, attendance, referee, *_manager, *_captain, *_formation
      - Raw match-day stats (leakage): home_score, away_score, *_xg, *_ht_score,
        and all home_*/away_* raw stat columns from the matches table
      - Target column: result
    """
    EXCLUDE_EXACT = {
        "match_id", "season", "match_date", "kickoff_time", "matchweek",
        "home_team", "away_team",
        "venue", "attendance", "referee", "data_source",
        "league", "league_name",
        "home_manager", "away_manager", "home_captain", "away_captain",
        "home_formation", "away_formation", "formation_confidence",
        "home_score", "away_score",
        "home_xg", "away_xg",
        "home_ht_score", "away_ht_score",
        "result",
        # Same-match shot totals (leakage)
        "home_shots_total", "away_shots_total",
        # Half-time stats (leakage — known after the match)
        "home_ht_goals", "away_ht_goals", "ht_result",
        # Post-match lineup ratings (leakage — computed from match performance)
        "home_lineup_rating_mean", "away_lineup_rating_mean",
        "lineup_rating_mean_diff",
        # Noise features — near-zero predictive signal, dilute gradient splits
        "kickoff_hour", "is_night_match", "is_evening_kickoff", "is_primetime",
        "is_early_kickoff", "is_monday_night",
        "day_of_week", "is_weekend", "is_midweek",
        "is_august", "is_december", "is_january", "is_may",
        "is_late_season", "season_phase",
        "nothing_to_play_for", "safe_from_relegation", "out_of_europe",
        # Zero-importance features (confirmed 0.0 in feature importance analysis)
        "home_xg_attack_strength", "away_xg_attack_strength",
        "home_xg_defense_strength", "away_xg_defense_strength",
        "xg_attack_strength_diff", "xg_defense_strength_diff",
        "home_short_rest", "away_short_rest",
        "home_very_short_rest", "away_very_short_rest",
        # Features unavailable at prediction time (100% NaN for upcoming matches)
        # Referee features: assigned close to kickoff, not available when predictions run
        "ref_avg_reds", "ref_big_match_cards", "ref_last_match_reds",
        "ref_regression_signal", "ref_strictness_trend", "ref_vs_away_team_bias",
        "ref_avg_yellows", "ref_home_bias", "ref_home_cards_bias",
        "ref_vs_home_team_bias", "ref_avg_total_goals", "ref_avg_xg",
        "ref_xg_vs_league", "ref_matches",
        # Weather: not reliably available at prediction time
        "weather_relative_humidity_2m_mean",
    }

    # Raw match-day stat columns that leak current-match info
    EXCLUDE_PREFIXED_STATS = {
        f"{side}_{stat}"
        for side in ("home", "away")
        for stat in STAT_COLUMNS
    }
    # Also exclude the _count/_total variants
    EXCLUDE_PREFIXED_STATS.update(
        f"{side}_{stat}_{suffix}"
        for side in ("home", "away")
        for stat in ("shots_on_target", "saves", "passing_accuracy")
        for suffix in ("count", "total")
    )
    # Exclude home_cards / away_cards if present
    EXCLUDE_PREFIXED_STATS.update({"home_cards", "away_cards"})

    exclude = EXCLUDE_EXACT | EXCLUDE_PREFIXED_STATS

    # Detect merge-artifact columns: any column ending in _x or _y from
    # pandas merge collisions.  No legitimate engineered feature uses these
    # suffixes — they only appear when pd.merge creates duplicates from
    # overlapping column names.  Drop all of them unconditionally.
    _merge_artifacts = {c for c in df.columns if c.endswith("_x") or c.endswith("_y")}

    ml_cols = [
        c for c in df.columns
        if c not in exclude
        and c not in _merge_artifacts
        and "Unnamed" not in c
        and "level_0" not in c
    ]

    # If Sofascore domain indices exist, exclude raw ss_roll_ features from ML.
    # The 30 composite indices (home_ss_idx_*, away_ss_idx_*, ss_idx_diff_*)
    # replace ~271 raw features that individually get zero importance.
    ss_idx_cols = [c for c in ml_cols if c.startswith(("home_ss_idx_", "away_ss_idx_", "ss_idx_diff_"))]
    if ss_idx_cols:
        _raw_ss_prefixes = ("home_ss_roll_", "away_ss_roll_", "ss_diff_ss_roll_")
        _raw_ss_extra = {
            "ss_coverage", "ss_xg_coverage", "top2_xg_share",
            "home_top2_xg_share", "away_top2_xg_share", "ss_diff_top2_xg_share",
        }
        ml_cols = [
            c for c in ml_cols
            if not c.startswith(_raw_ss_prefixes) and c not in _raw_ss_extra
        ]

    return ml_cols


def _add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contextual timing and situation features.

    Includes:
      - Kickoff time features (hour, early/evening/primetime flags)
      - Day of week features (weekend, Friday/Monday night)
      - Season phase features (early/mid/late season, run-in)
      - Month features (August rust, December congestion, January transfers, May finale)
      - International break detection (rest > 14 days)
      - Travel fatigue (distance / rest interaction)
      - Derby defensive boost
      - Promoted team flags
      - Nothing-to-play-for indicator (mid-table, late season)
    """
    added = 0

    # 1. Kickoff time features
    if "kickoff_time" in df.columns:
        df["kickoff_hour"] = pd.to_datetime(df["kickoff_time"], errors="coerce").dt.hour.fillna(15)
    else:
        df["kickoff_hour"] = 15  # default afternoon

    df["is_early_kickoff"] = (df["kickoff_hour"] <= 13).astype(int)
    df["is_evening_kickoff"] = (df["kickoff_hour"] >= 19).astype(int)
    df["is_primetime"] = (df["kickoff_hour"] >= 20).astype(int)
    added += 4

    # 2. Day of week features
    if "match_date" in df.columns:
        _mdate = pd.to_datetime(df["match_date"], errors="coerce")
        df["day_of_week"] = _mdate.dt.dayofweek.fillna(5)  # 0=Mon, 6=Sun
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_friday_night"] = (df["day_of_week"] == 4).astype(int)
        df["is_monday_night"] = (df["day_of_week"] == 0).astype(int)
        added += 4

    # 3. Season phase features
    if "matchweek" in df.columns:
        mw = pd.to_numeric(df["matchweek"], errors="coerce").fillna(19)
        _league = df["league"].iloc[0] if "league" in df.columns else None
        _max_mw = get_matchweeks(_league)
        # Dynamic thresholds scaled to the league's matchweek count
        _early = round(_max_mw * 0.13)    # ~5 for 38-mw leagues
        _mid_lo = round(_max_mw * 0.39)   # ~15
        _mid_hi = round(_max_mw * 0.66)   # ~25
        _late = round(_max_mw * 0.87)     # ~33
        _run_in = round(_max_mw * 0.79)   # ~30
        df["is_early_season"] = (mw <= _early).astype(int)
        df["is_mid_season"] = ((mw >= _mid_lo) & (mw <= _mid_hi)).astype(int)
        df["is_late_season"] = (mw >= _late).astype(int)
        df["is_run_in"] = (mw >= _run_in).astype(int)
        df["season_phase"] = pd.cut(
            mw, bins=[0, _early, _mid_lo, _mid_hi, _late, _max_mw],
            labels=[1, 2, 3, 4, 5],
        ).astype(float)
        added += 5

    # 4. Month features
    if "match_date" in df.columns:
        _month = pd.to_datetime(df["match_date"], errors="coerce").dt.month.fillna(1)
        df["is_august"] = (_month == 8).astype(int)  # Season opener rust
        df["is_december"] = (_month == 12).astype(int)  # Congestion
        df["is_january"] = (_month == 1).astype(int)  # Transfer window
        df["is_may"] = (_month == 5).astype(int)  # Season finale
        added += 4

    # 5. International break detection (rest > 14 days suggests FIFA break)
    for prefix in ["home", "away"]:
        rest_col = f"{prefix}_rest_days"
        if rest_col in df.columns:
            df[f"{prefix}_post_intl_break"] = (df[rest_col] > 14).astype(int)
            added += 1

    # 6. Travel x rest interaction (travel fatigue)
    if "travel_distance_km" in df.columns:
        for prefix in ["home", "away"]:
            rest_col = f"{prefix}_rest_days"
            if rest_col in df.columns:
                df[f"{prefix}_travel_fatigue"] = (
                    df["travel_distance_km"] / df[rest_col].clip(lower=1)
                ).fillna(0)
                added += 1

    # 7. Derby xG adjustment (derbies tend to be lower-scoring: 1.8 vs 2.6 avg total goals)
    if "is_derby" in df.columns:
        df["derby_defensive_boost"] = df["is_derby"].astype(int)
        added += 1

    # 8. Promoted team flag — handled by Step19e creative_factors (authoritative list)
    # Do NOT overwrite here; the proxy heuristic (matches_played < 38) fails for
    # returning teams like Sassuolo that have extensive historical data.

    # 9. Nothing-to-play-for indicator (mid-table 8-14, late season >28)
    if "home_league_position" in df.columns and "matchweek" in df.columns:
        mw = pd.to_numeric(df["matchweek"], errors="coerce").fillna(19)
        for prefix in ["home", "away"]:
            pos_col = f"{prefix}_league_position"
            if pos_col in df.columns:
                pos = pd.to_numeric(df[pos_col], errors="coerce").fillna(10)
                # Mid-table (8-14), late season (>28), not fighting for anything
                df[f"{prefix}_nothing_to_play_for"] = (
                    (pos >= 8) & (pos <= 14) & (mw >= 28)
                ).astype(int)
                added += 1

    log.info("Added %d contextual timing/situation features", added)
    return df


def _validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate domain constraints on features and clip impossible values.

    Checks and clips:
    - Raw xG columns >= 0 (excludes diffs/trends/overperformance which can be negative)
    - Probability columns in [0, 1]
    - Absolute Elo ratings in [800, 2200] (excludes elo_diff and interaction columns)
    - Rolling averages >= 0
    Logs warnings for violations found.
    """
    violations = {}

    # Signed column patterns — these are differences/interactions that legitimately go negative
    _SIGNED_PATTERNS = ("diff", "trend", "overperformance", "mismatch", "signal",
                        "advantage", "disruption", "momentum", "boost", "_x_")

    def _is_signed(col: str) -> bool:
        cl = col.lower()
        return any(p in cl for p in _SIGNED_PATTERNS)

    # xG columns: must be >= 0 (only raw/rolling xG, not diffs or derived)
    xg_cols = [c for c in df.columns
               if "xg" in c.lower()
               and df[c].dtype in ("float64", "float32", "int64")
               and not _is_signed(c)]
    for col in xg_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            violations[col] = f"{neg_count} negative values"
            df[col] = df[col].clip(lower=0)

    # Probability columns: must be in [0, 1]
    prob_cols = [c for c in df.columns
                 if ("_rate" in c or "probability" in c or "win_rate" in c or "_prob" in c)
                 and df[c].dtype in ("float64", "float32")
                 and not _is_signed(c)]
    for col in prob_cols:
        out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
        if out_of_range > 0:
            violations[col] = f"{out_of_range} values outside [0, 1]"
            df[col] = df[col].clip(lower=0, upper=1)

    # Elo columns: only absolute ratings (home_elo, away_elo), NOT diffs or interactions
    elo_abs_cols = [c for c in df.columns
                    if c in ("home_elo", "away_elo")
                    and df[c].dtype in ("float64", "float32", "int64")]
    for col in elo_abs_cols:
        out_of_range = ((df[col] < 800) | (df[col] > 2200)).sum()
        if out_of_range > 0:
            violations[col] = f"{out_of_range} values outside [800, 2200]"
            df[col] = df[col].clip(lower=800, upper=2200)

    # Rolling averages: must be >= 0
    roll_cols = [c for c in df.columns if c.startswith(("home_roll_", "away_roll_")) and df[c].dtype in ("float64", "float32")]
    for col in roll_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            violations[col] = f"{neg_count} negative values"
            df[col] = df[col].clip(lower=0)

    if violations:
        log.warning("Feature validation found %d columns with domain violations (clipped):", len(violations))
        for col, desc in list(violations.items())[:20]:
            log.warning("  %s: %s", col, desc)
    else:
        log.info("Feature validation: all domain constraints satisfied")

    return df


def _add_derby_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derby/rivalry features based on team matchup.

    League-aware: when a ``league`` column is present, derby definitions are
    looked up from the correct league's registry. Without the column, falls
    back to Serie A derbies (backward compatible).

    Columns added:
      is_derby         - 1 if the match is a derby/rivalry, 0 otherwise
      derby_intensity  - 0-3 scale (0=none, 1=historical, 2=regional, 3=city derby)
      derby_home_advantage_boost - city derbies reduce home advantage (negative value)
    """
    has_league = "league" in df.columns

    # Pre-load derby dicts per league to avoid repeated lookups
    _derby_cache: dict[str, dict[frozenset[str], int]] = {}

    def _get_derbies_for_league(league_key: str) -> dict[frozenset[str], int]:
        if league_key not in _derby_cache:
            _derby_cache[league_key] = get_derbies(league_key)
        return _derby_cache[league_key]

    is_derby = []
    intensity = []
    home_boost = []

    for _, row in df.iterrows():
        pair = frozenset({row.get("home_team", ""), row.get("away_team", "")})
        if has_league and pd.notna(row.get("league")):
            derbies = _get_derbies_for_league(row["league"])
        else:
            derbies = SERIE_A_DERBIES
        level = derbies.get(pair, 0)
        is_derby.append(1 if level > 0 else 0)
        intensity.append(level)
        # City derbies (level 3) reduce home advantage; fans split evenly
        home_boost.append(-0.05 if level == 3 else (-0.02 if level == 2 else 0.0))

    df["is_derby"] = is_derby
    df["derby_intensity"] = intensity
    df["derby_home_advantage_boost"] = home_boost

    n_derbies = sum(is_derby)
    log.info("Added derby features (%d derbies out of %d matches)", n_derbies, len(df))
    return df


def _add_league_position_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add league position context features derived from standings data.

    Columns added:
      points_to_cl_zone    - Points gap to 4th place (positive = above, negative = below)
      points_to_relegation - Points gap to 18th place (positive = safe margin)
      home_position_zone   - 1=title, 2=CL, 3=EL, 4=mid, 5=relegation
      away_position_zone   - same for away team
    """
    for prefix in ("home", "away"):
        pos_col = f"{prefix}_league_pos"
        pts_col = f"{prefix}_league_points"

        if pos_col not in df.columns or pts_col not in df.columns:
            continue

        pos = pd.to_numeric(df[pos_col], errors="coerce")
        pts = pd.to_numeric(df[pts_col], errors="coerce")

        # Position zone classification
        zone = pd.Series(4, index=df.index)  # default: midtable
        zone = zone.where(pos.isna() | (pos > 2), 1)     # title race: pos 1-2
        zone = zone.where(pos.isna() | ~((pos > 2) & (pos <= 4)), 2)   # CL zone: 3-4
        zone = zone.where(pos.isna() | ~((pos > 4) & (pos <= 6)), 3)   # EL zone: 5-6
        zone = zone.where(pos.isna() | (pos < 18), 5)     # relegation: 18-20
        df[f"{prefix}_position_zone"] = zone.where(pos.notna())

    # Points gaps: compute actual point difference to 4th and 18th place
    if "home_league_points" in df.columns and "home_league_pos" in df.columns:
        home_pos = pd.to_numeric(df["home_league_pos"], errors="coerce")
        away_pos = pd.to_numeric(df["away_league_pos"], errors="coerce")
        home_pts = pd.to_numeric(df["home_league_points"], errors="coerce")
        away_pts = pd.to_numeric(df["away_league_points"], errors="coerce")

        pts_at_4 = pd.Series(np.nan, index=df.index)
        pts_at_18 = pd.Series(np.nan, index=df.index)

        for target_pos, pts_series in [(4, pts_at_4), (18, pts_at_18)]:
            exact_home = home_pos == target_pos
            pts_series[exact_home] = home_pts[exact_home]
            exact_away = (~exact_home) & (away_pos == target_pos)
            pts_series[exact_away] = away_pts[exact_away]
            remaining = pts_series.isna() & home_pos.notna() & away_pos.notna()
            if remaining.any():
                pos_range = (away_pos[remaining] - home_pos[remaining]).replace(0, np.nan)
                pts_per_pos = (away_pts[remaining] - home_pts[remaining]) / pos_range
                pts_series[remaining] = home_pts[remaining] + pts_per_pos * (target_pos - home_pos[remaining])

        for prefix in ("home", "away"):
            pts = pd.to_numeric(df[f"{prefix}_league_points"], errors="coerce")
            df[f"{prefix}_points_to_cl_zone"] = (pts - pts_at_4).round(1)
            df[f"{prefix}_points_to_relegation"] = (pts - pts_at_18).round(1)

    added = sum(1 for c in ["home_position_zone", "away_position_zone",
                             "home_points_to_cl_zone", "away_points_to_cl_zone",
                             "home_points_to_relegation", "away_points_to_relegation"]
                if c in df.columns)
    log.info("Added %d league position context features", added)
    return df


def _pivot_to_match_level(matches: pd.DataFrame, team_log: pd.DataFrame) -> pd.DataFrame:
    """Pivot team-level features back to match-level with home_*/away_* prefixes."""
    # Identify feature columns (everything added beyond the base columns)
    base_cols = {
        "match_id", "match_date", "season", "matchweek", "team", "opponent",
        "is_home", "goals_scored", "goals_conceded", "result", "points",
        "clean_sheet", "xg_for", "xg_against", "formation", "league",
    }
    # Exclude raw team stats that are already in matches as home_*/away_*
    base_cols.update(STAT_COLUMNS)
    feature_cols = [c for c in team_log.columns if c not in base_cols]

    # Split into home and away
    home_feats = team_log[team_log["is_home"] == True][["match_id"] + feature_cols].copy()
    away_feats = team_log[team_log["is_home"] == False][["match_id"] + feature_cols].copy()

    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feature_cols})
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feature_cols})

    # Start from the base matches table
    result = matches.copy()
    result = result.merge(home_feats, on="match_id", how="left")
    result = result.merge(away_feats, on="match_id", how="left")

    return result


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features that combine strong predictors.

    Pruned: removed 9 interactions that combined zero-importance features
    (injury, PPDA, manager tenure, squad disruption) — all had 0.0 importance.

    Kept: proven interactions + new strong×strong combinations.
    """
    added = 0

    # --- Proven interactions (kept from original) ---

    # 1. Elo x form: strong team in good form vs weak team in bad form
    if "elo_diff" in df.columns and "home_roll_5_points" in df.columns:
        form_diff = df["home_roll_5_points"].fillna(0) - df.get("away_roll_5_points", pd.Series(0, index=df.index)).fillna(0)
        df["elo_x_form"] = (df["elo_diff"].fillna(0) * form_diff / 10).round(3)
        added += 1

    # 2. Attack strength x defense weakness: mismatch exploitation
    if "home_attack_strength" in df.columns and "away_defense_strength" in df.columns:
        home_mismatch = df["home_attack_strength"].fillna(1) / df["away_defense_strength"].fillna(1).clip(lower=0.1)
        away_mismatch = df["away_attack_strength"].fillna(1) / df["home_defense_strength"].fillna(1).clip(lower=0.1)
        df["attack_defense_mismatch"] = (home_mismatch - away_mismatch).round(3)
        added += 1

    # 3. League position differential
    if "home_league_pos" in df.columns and "away_league_pos" in df.columns:
        df["league_pos_diff"] = (
            df["away_league_pos"].fillna(10) - df["home_league_pos"].fillna(10)
        ).round(0)  # positive = home is higher in table
        added += 1

    # 4. Rest days advantage
    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        rest_diff = df["home_rest_days"].fillna(7) - df["away_rest_days"].fillna(7)
        df["rest_advantage"] = rest_diff.clip(-5, 5).round(0)
        if "elo_diff" in df.columns:
            elo_close = (df["elo_diff"].fillna(0).abs() < 50).astype(float)
            df["rest_x_close_elo"] = (rest_diff * elo_close).round(1)
            added += 1
        added += 1

    # 5. xG overperformance: goals vs xG indicates luck (regression candidate)
    if "home_roll_5_goals_scored" in df.columns and "home_roll_5_xg_for" in df.columns:
        home_ovp = df["home_roll_5_goals_scored"].fillna(0) - df["home_roll_5_xg_for"].fillna(0)
        away_ovp = df.get("away_roll_5_goals_scored", pd.Series(0, index=df.index)).fillna(0) - \
                   df.get("away_roll_5_xg_for", pd.Series(0, index=df.index)).fillna(0)
        df["xg_overperformance_diff"] = (home_ovp - away_ovp).round(3)
        added += 1

    # 6. Defensive solidity differential
    if "home_roll_5_goals_conceded" in df.columns:
        home_def = df["home_roll_5_goals_conceded"].fillna(1.5)
        away_def = df.get("away_roll_5_goals_conceded", pd.Series(1.5, index=df.index)).fillna(1.5)
        df["defensive_form_diff"] = (away_def - home_def).round(3)
        added += 1

    # --- NEW: Strong × Strong interactions ---

    # 7. Elo × xG signal: quality gap amplified by recent scoring efficiency
    #    elo_diff (10.03 importance) × us_xg_diff (1.59 importance)
    if "elo_diff" in df.columns and "us_xg_diff" in df.columns:
        df["elo_xg_signal"] = (
            (df["elo_diff"].fillna(0) / 100) * df["us_xg_diff"].fillna(0)
        ).round(3)
        added += 1

    # 8. Attack strength × form alignment: talent aligned with current form
    #    attack_strength_diff (1.64) × form_points differential
    if "attack_strength_diff" in df.columns and "home_form_points_5" in df.columns:
        form_diff = (
            df["home_form_points_5"].fillna(0) -
            df.get("away_form_points_5", pd.Series(0, index=df.index)).fillna(0)
        )
        df["attack_form_alignment"] = (
            df["attack_strength_diff"].fillna(0) * form_diff / 10
        ).round(3)
        added += 1

    # 9. H2H × competitiveness: history matters more in close games
    #    h2h_home_win_rate (0.84) × matchup_competitiveness (3.50)
    if "h2h_home_win_rate" in df.columns and "matchup_competitiveness" in df.columns:
        df["h2h_competitiveness_signal"] = (
            (df["h2h_home_win_rate"].fillna(0.5) - 0.5) *
            df["matchup_competitiveness"].fillna(0)
        ).round(3)
        added += 1

    # 10. Form × Elo signal: form momentum weighted by quality gap
    #     form_points_5 (0.63) × elo_diff (10.03)
    if "home_form_points_5" in df.columns and "elo_diff" in df.columns:
        form_diff = (
            df["home_form_points_5"].fillna(0) -
            df.get("away_form_points_5", pd.Series(0, index=df.index)).fillna(0)
        )
        df["form_elo_signal"] = (
            form_diff * df["elo_diff"].fillna(0) / 100
        ).round(3)
        added += 1

    # --- Odds-model interaction features (kept) ---

    # 11. Sharp-soft divergence x Elo
    if "sharp_soft_home_div" in df.columns and "elo_diff" in df.columns:
        df["sharp_soft_x_elo"] = (
            df["sharp_soft_home_div"].fillna(0) * df["elo_diff"].fillna(0)
        ).round(4)
        added += 1

    # 12. Market prob vs Elo-expected prob: disagreement signal
    if "home_prob_best" in df.columns and "home_elo" in df.columns and "away_elo" in df.columns:
        elo_diff = df["home_elo"].fillna(1500) - df["away_elo"].fillna(1500)
        elo_expected = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
        df["market_elo_disagreement"] = (df["home_prob_best"].fillna(elo_expected) - elo_expected).round(4)
        added += 1

    # 13. Market implied goals vs rolling xG
    if "market_goal_total" in df.columns and "home_roll_5_xg_for" in df.columns:
        rolling_xg_total = (
            df["home_roll_5_xg_for"].fillna(0) + df.get("away_roll_5_xg_for", pd.Series(0, index=df.index)).fillna(0)
        )
        df["goal_total_vs_xg"] = (df["market_goal_total"].fillna(0) - rolling_xg_total).round(3)
        added += 1

    # 14. Asian handicap line x form differential
    if "ah_line_normalized" in df.columns and "home_roll_5_points" in df.columns:
        form_diff = (
            df["home_roll_5_points"].fillna(0) - df.get("away_roll_5_points", pd.Series(0, index=df.index)).fillna(0)
        )
        df["ah_x_form"] = (df["ah_line_normalized"].fillna(0) * form_diff / 10).round(4)
        added += 1

    # 15. Draw convergence x competitiveness
    if "combined_draw_tendency" in df.columns and "matchup_competitiveness" in df.columns:
        df["draw_convergence_x_competitiveness"] = (
            df["combined_draw_tendency"].fillna(0) * df["matchup_competitiveness"].fillna(0)
        ).round(4)
        added += 1

    log.info("Added %d interaction features", added)
    return df


def _match_result(row) -> str:
    """Determine match result: H (home win), D (draw), A (away win)."""
    hs = row.get("home_score")
    as_ = row.get("away_score")
    if pd.isna(hs) or pd.isna(as_):
        return "U"
    hs, as_ = int(hs), int(as_)
    if hs > as_:
        return "H"
    if hs < as_:
        return "A"
    return "D"


def _add_league_draw_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add league-level historical rates and draw-specific features.

    Features added:
      - league_home_win_rate: expanding home win % in this league
      - league_draw_rate: expanding draw % in this league
      - league_avg_goals: expanding avg total goals per match in this league
      - home_draw_tendency_10/5: rolling draw frequency for home team (last 10/5)
      - away_draw_tendency_10/5: rolling draw frequency for away team (last 10/5)
      - matchup_competitiveness: 1 / (1 + |strength_diff|) -- closer teams draw more
      - both_defenses_strong: min(home_def, away_def) -- mutual impermeability
      - both_attacks_weak: 1 - max(home_atk, away_atk) -- mutual inability to score
      - combined_draw_tendency: avg of both teams' draw tendency_10
      - defense_similarity: 1 / (1 + |def_diff|) -- comparable defensive quality
      - low_scoring_signal: avg recent goals conceded (lower -> draw-conducive)
    """
    import numpy as np

    df = feature_df.copy()
    df["_sort_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("_sort_date").reset_index(drop=True)

    # Determine result for historical computation (use existing columns)
    hs = pd.to_numeric(df.get("home_score"), errors="coerce")
    as_ = pd.to_numeric(df.get("away_score"), errors="coerce")
    df["_is_home_win"] = (hs > as_).astype(float)
    df["_is_draw"] = (hs == as_).astype(float)
    df["_total_goals"] = hs + as_

    # --- League-level expanding rates (shift to avoid leakage) ---
    league_col = "league" if "league" in df.columns else None

    if league_col:
        grp = df.groupby(league_col)
        df["league_home_win_rate"] = grp["_is_home_win"].apply(
            lambda s: s.expanding().mean().shift(1)
        ).reset_index(level=0, drop=True)
        df["league_draw_rate"] = grp["_is_draw"].apply(
            lambda s: s.expanding().mean().shift(1)
        ).reset_index(level=0, drop=True)
        df["league_avg_goals"] = grp["_total_goals"].apply(
            lambda s: s.expanding().mean().shift(1)
        ).reset_index(level=0, drop=True)
    else:
        df["league_home_win_rate"] = df["_is_home_win"].expanding().mean().shift(1)
        df["league_draw_rate"] = df["_is_draw"].expanding().mean().shift(1)
        df["league_avg_goals"] = df["_total_goals"].expanding().mean().shift(1)

    # --- Per-team draw tendency (rolling 10 and 5 matches) ---
    for side in ("home", "away"):
        team_col = f"{side}_team"
        if team_col not in df.columns:
            continue
        # Build per-team draw history
        draw_maps = {}
        tendency_10 = np.full(len(df), np.nan)
        tendency_5 = np.full(len(df), np.nan)
        for i, row in df.iterrows():
            team = row[team_col]
            if pd.isna(team):
                continue
            hist = draw_maps.get(team, [])
            if len(hist) > 0:
                recent_10 = hist[-10:]  # last 10 matches
                tendency_10[i] = sum(recent_10) / len(recent_10)
                recent_5 = hist[-5:]  # last 5 matches (more reactive)
                tendency_5[i] = sum(recent_5) / len(recent_5)
            draw_maps.setdefault(team, []).append(row["_is_draw"])
        df[f"{side}_draw_tendency_10"] = tendency_10
        df[f"{side}_draw_tendency_5"] = tendency_5

    # --- Matchup competitiveness ---
    # Use strength rating diff if available, else Elo diff
    if "home_strength_rating" in df.columns and "away_strength_rating" in df.columns:
        diff = (df["home_strength_rating"] - df["away_strength_rating"]).abs()
        df["matchup_competitiveness"] = 1.0 / (1.0 + diff)
    elif "home_elo" in df.columns and "away_elo" in df.columns:
        diff = (df["home_elo"] - df["away_elo"]).abs()
        df["matchup_competitiveness"] = 1.0 / (1.0 + diff / 400.0)

    # --- Draw-convergence composite features ---
    # These capture team SIMILARITY/CONVERGENCE signals rather than DIFFERENCES.
    # Draws happen when teams are similar; diff-based features predict H vs A well
    # but provide little draw signal.

    # both_defenses_strong: higher = both teams concede few goals -> stalemate
    if "home_defense_strength" in df.columns and "away_defense_strength" in df.columns:
        df["both_defenses_strong"] = np.minimum(
            df["home_defense_strength"], df["away_defense_strength"]
        )

    # both_attacks_weak: higher = neither team has a potent attack -> few goals -> draw
    if "home_attack_strength" in df.columns and "away_attack_strength" in df.columns:
        df["both_attacks_weak"] = (
            1.0 - np.maximum(df["home_attack_strength"], df["away_attack_strength"])
        ).clip(0, 1)

    # combined_draw_tendency: average of both teams' recent draw frequency
    if "home_draw_tendency_10" in df.columns and "away_draw_tendency_10" in df.columns:
        df["combined_draw_tendency"] = (
            df["home_draw_tendency_10"].fillna(0) + df["away_draw_tendency_10"].fillna(0)
        ) / 2.0

    # defense_similarity: how comparable are the two defenses (1 = identical, <1 = different)
    if "home_defense_strength" in df.columns and "away_defense_strength" in df.columns:
        df["defense_similarity"] = 1.0 / (
            1.0 + (df["home_defense_strength"] - df["away_defense_strength"]).abs()
        )

    # low_scoring_signal: average goals conceded recently (lower = draw-conducive)
    gc_home = df.get("home_roll_5_goals_conceded")
    gc_away = df.get("away_roll_5_goals_conceded")
    if gc_home is None:
        gc_home = df.get("home_roll_10_goals_conceded")
    if gc_away is None:
        gc_away = df.get("away_roll_10_goals_conceded")
    if gc_home is not None and gc_away is not None:
        df["low_scoring_signal"] = (gc_home.fillna(1.5) + gc_away.fillna(1.5)) / 2.0

    # Cleanup temp columns
    df.drop(columns=["_sort_date", "_is_home_win", "_is_draw", "_total_goals"],
            inplace=True, errors="ignore")

    draw_feature_cols = [
        "league_home_win_rate", "league_draw_rate", "league_avg_goals",
        "home_draw_tendency_10", "away_draw_tendency_10", "matchup_competitiveness",
        "home_draw_tendency_5", "away_draw_tendency_5",
        "both_defenses_strong", "both_attacks_weak", "combined_draw_tendency",
        "defense_similarity", "low_scoring_signal",
    ]
    n_new = sum(1 for c in draw_feature_cols if c in df.columns)
    log.info("Added %d league-aware/draw features", n_new)

    return df


def _add_weather(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add weather features to match data.

    First tries to load actual weather data from cached file.
    Falls back to estimated weather based on date/location.
    """
    from config.settings import DATA_DIR

    # Try to load cached weather data first
    weather_path = DATA_DIR / "external" / "weather.parquet"
    if weather_path.exists():
        log.info("Loading cached weather data")
        weather = pd.read_parquet(weather_path)
        feature_df = feature_df.merge(weather, on="match_id", how="left")
        return feature_df

    # Fall back to estimated weather from our weather module
    try:
        from features.enhanced_weather import add_weather_features
        log.info("Using estimated weather from date/location")
        return add_weather_features(feature_df)
    except ImportError:
        log.warning("Weather module not available; skipping weather features")
        return feature_df


def _add_venue_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add venue-related features (travel distance, altitude, capacity)."""
    try:
        from features.venue import add_venue_features
        return add_venue_features(feature_df)
    except ImportError:
        log.warning("Venue module not available; skipping venue features")
        return feature_df


def _add_odds(feature_df: pd.DataFrame, season: str | None) -> pd.DataFrame:
    """Merge cached betting odds if available.

    Handles multi-league data by downloading odds for the appropriate league.
    """
    from scraper.odds import compute_implied_probabilities, download_all_odds, merge_odds_with_matches
    from config.settings import DATA_DIR, LEAGUES, SEASONS

    odds_dir = DATA_DIR / "external" / "odds"

    seasons = [season] if season else SEASONS

    # Determine which league this data belongs to
    league = "serie_a"
    if "league" in feature_df.columns:
        league_vals = feature_df["league"].dropna().unique()
        if len(league_vals) == 1:
            league = league_vals[0]

    # Try to download odds for this league
    odds = download_all_odds(seasons, league=league)
    if odds.empty:
        # Check if we have cached data
        if not odds_dir.exists() or not any(odds_dir.iterdir()):
            log.info("No odds data found for %s; skipping", league)
            return feature_df
        return feature_df

    feature_df = merge_odds_with_matches(feature_df, odds)

    # Clean up merge artifacts: when matches.parquet already has odds columns
    # and merge_odds_with_matches adds overlapping ones, pandas creates _x/_y
    # suffixes. Keep _x (from matches.parquet, higher coverage) and drop _y.
    x_cols = [c for c in feature_df.columns if c.endswith('_x')]
    y_cols = [c for c in feature_df.columns if c.endswith('_y')]
    rename_map = {}
    drop_cols = []
    for xc in x_cols:
        base = xc[:-2]  # strip _x
        yc = base + '_y'
        if yc in feature_df.columns:
            # Merge: fill _x NaN from _y, then keep as clean name
            feature_df[xc] = feature_df[xc].fillna(feature_df[yc])
            drop_cols.append(yc)
        rename_map[xc] = base
    if drop_cols:
        feature_df = feature_df.drop(columns=drop_cols, errors='ignore')
    if rename_map:
        feature_df = feature_df.rename(columns=rename_map)
        log.info("Cleaned %d odds merge artifacts (_x/_y -> clean names)", len(rename_map))

    feature_df = compute_implied_probabilities(feature_df)
    return feature_df


def _add_market_data(feature_df: pd.DataFrame, season: str | None) -> pd.DataFrame:
    """Merge cached Transfermarkt data if available.

    IMPORTANT: Market values and transfers are season-specific. We only apply
    them to matches from the corresponding season to prevent cross-season leakage.

    Supports multi-league: looks for league-prefixed files (e.g.
    premier_league_market_values_2024_2025.parquet) and falls back to
    unprefixed files for Serie A (backward compatible).
    """
    from config.settings import DATA_DIR, SEASONS
    tm_dir = DATA_DIR / "external" / "transfermarkt"
    if not tm_dir.exists():
        log.info("No Transfermarkt data found (run 'fetch-transfers' first); skipping")
        return feature_df

    from scraper.transfermarkt import compute_squad_value_features, compute_transfer_features

    # Determine league prefix for file lookup
    league_key = ""
    if "league" in feature_df.columns:
        leagues = feature_df["league"].dropna().unique()
        if len(leagues) == 1:
            league_key = str(leagues[0])

    # Serie A uses no prefix (backward compatible), others use league key prefix
    file_prefix = "" if league_key in ("serie_a", "") else f"{league_key}_"

    # Apply market data per-season to avoid cross-season contamination
    seasons_to_check = [season] if season else SEASONS
    for s in seasons_to_check:
        mv_path = tm_dir / f"{file_prefix}market_values_{s.replace('-', '_')}.parquet"
        tr_path = tm_dir / f"{file_prefix}transfers_{s.replace('-', '_')}.parquet"

        season_mask = feature_df["season"] == s
        if not season_mask.any():
            continue

        season_rows = feature_df[season_mask].copy()
        other_rows = feature_df[~season_mask]

        # Drop any market/transfer columns from prior iterations to avoid merge conflicts
        mv_cols = [c for c in season_rows.columns if any(
            c.endswith(s) for s in (
                "_squad_total_value", "_avg_player_value", "_median_player_value",
                "_max_player_value", "_squad_size", "_transfer_spend",
                "_transfer_income", "_transfers_in", "_transfers_out",
                "_net_spend",
            )
        ) or c == "squad_value_ratio"]
        if mv_cols:
            season_rows = season_rows.drop(columns=mv_cols, errors="ignore")

        if mv_path.exists():
            mv = pd.read_parquet(mv_path)
            season_rows = compute_squad_value_features(season_rows, mv)
        else:
            log.info("No market value data for %s (looked for %s)", s, mv_path.name)

        if tr_path.exists():
            tr = pd.read_parquet(tr_path)
            season_rows = compute_transfer_features(season_rows, tr)
        else:
            log.info("No transfer data for %s (looked for %s)", s, tr_path.name)

        # Reassemble: put enriched season rows back
        feature_df = pd.concat([other_rows, season_rows], ignore_index=True)
        feature_df["_sort_date"] = pd.to_datetime(feature_df["match_date"], errors="coerce")
        feature_df = feature_df.sort_values("_sort_date").reset_index(drop=True)
        feature_df.drop(columns=["_sort_date"], inplace=True)

    return feature_df
