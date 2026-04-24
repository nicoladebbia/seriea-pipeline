"""BacktestReport — the canonical output schema for all backtest runs.

Schema is nested:

    report.markets[market][edge_threshold_pct][stake_policy][odds_source] -> ROIStats

JSON-serializable. Stable: future predictors (simulator, new CatBoost variants,
meta-learners) write into this same schema so a simple diff tells you which
predictor wins per market.

See docs/backtest_report_schema.md for the full spec.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .roi_bootstrap import ROIStats


@dataclass
class BacktestRunMetadata:
    predictor_name: str
    predictor_version: str
    seasons: list[str]
    markets: list[str]
    edge_thresholds_pct: list[float]
    stake_policies: list[str]
    odds_sources_observed: list[str]
    n_matches_scored: int
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed_salt: int = 0
    walk_forward_mode: str = "season_boundary"


@dataclass
class BacktestReport:
    metadata: BacktestRunMetadata
    # markets[market_label] -> per_threshold
    # per_threshold[thresh_key] -> per_stake
    # per_stake[stake_label] -> per_source
    # per_source[source_label] -> ROIStats
    markets: dict[str, dict[str, dict[str, dict[str, ROIStats]]]] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"metadata": asdict(self.metadata), "markets": {}}
        for mk, thresh_dict in self.markets.items():
            out["markets"][mk] = {}
            for tk, stake_dict in thresh_dict.items():
                out["markets"][mk][tk] = {}
                for sp, src_dict in stake_dict.items():
                    out["markets"][mk][tk][sp] = {}
                    for src, stats in src_dict.items():
                        out["markets"][mk][tk][sp][src] = asdict(stats)
        return out

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2, default=str))

    def summary_rows(self) -> list[dict[str, Any]]:
        """Flat list of rows for CLI printing / CSV export."""
        rows: list[dict[str, Any]] = []
        for mk, thresh_dict in self.markets.items():
            for tk, stake_dict in thresh_dict.items():
                for sp, src_dict in stake_dict.items():
                    for src, stats in src_dict.items():
                        rows.append({
                            "market": mk,
                            "threshold": tk,
                            "stake_policy": sp,
                            "odds_source": src,
                            "n_bets": stats.n_bets,
                            "roi_pct": stats.roi_pct_point,
                            "ci_lower": stats.roi_pct_ci_lower,
                            "ci_upper": stats.roi_pct_ci_upper,
                            "max_dd_eur": stats.max_drawdown_eur,
                            "sharpe": stats.sharpe,
                            "long_lose_streak": stats.longest_losing_streak,
                        })
        return rows
