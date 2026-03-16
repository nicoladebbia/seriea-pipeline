#!/usr/bin/env python3
"""Performance Dashboard - Phase 6.

Provides:
- Real-time accuracy tracking
- Betting P&L visualization
- Feature importance trends
- Model version comparison
- Alert management
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and store prediction metrics for dashboard."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DATA_DIR / "monitoring" / "metrics"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_file = self.storage_dir / "predictions.json"
        self.bets_file = self.storage_dir / "betting_results.json"
        self.daily_stats_file = self.storage_dir / "daily_stats.json"

        self.predictions: List[Dict] = self._load_json(self.predictions_file)
        self.bets: List[Dict] = self._load_json(self.bets_file)
        self.daily_stats: Dict[str, Dict] = self._load_json(self.daily_stats_file, default={})

    def _load_json(self, path: Path, default=None) -> any:
        """Load JSON file or return default."""
        if default is None:
            default = []
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default

    def _save_json(self, path: Path, data):
        """Save data to JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def record_prediction(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        predicted_class: int,
        predicted_probs: List[float],
        actual_class: int = None,
        confidence: str = None,
        model_version: str = None,
        odds: Dict = None,
    ):
        """Record a prediction result.

        Args:
            match_id: Unique match identifier
            home_team: Home team name
            away_team: Away team name
            predicted_class: Predicted outcome (0=H, 1=D, 2=A)
            predicted_probs: Probability distribution [H, D, A]
            actual_class: Actual outcome (None if not yet known)
            confidence: Confidence level string
            model_version: Model version used
            odds: Betting odds if available
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "predicted_class": predicted_class,
            "predicted_probs": predicted_probs,
            "actual_class": actual_class,
            "correct": predicted_class == actual_class if actual_class is not None else None,
            "confidence": confidence,
            "confidence_score": max(predicted_probs),
            "model_version": model_version,
            "odds": odds,
        }

        self.predictions.append(record)

        # Update daily stats
        date_key = datetime.now().strftime("%Y-%m-%d")
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                "predictions": 0,
                "correct": 0,
                "high_conf_predictions": 0,
                "high_conf_correct": 0,
            }

        self.daily_stats[date_key]["predictions"] += 1
        if actual_class is not None and predicted_class == actual_class:
            self.daily_stats[date_key]["correct"] += 1

        if max(predicted_probs) >= 0.5:
            self.daily_stats[date_key]["high_conf_predictions"] += 1
            if actual_class is not None and predicted_class == actual_class:
                self.daily_stats[date_key]["high_conf_correct"] += 1

        # Periodic save
        if len(self.predictions) % 10 == 0:
            self.save_all()

    def record_bet(
        self,
        match_id: str,
        bet_type: str,
        stake: float,
        odds: float,
        predicted_prob: float,
        result: str = None,
        profit: float = None,
    ):
        """Record a betting result.

        Args:
            match_id: Match identifier
            bet_type: Type of bet (e.g., "home_win", "over_2.5")
            stake: Amount staked
            odds: Decimal odds
            predicted_prob: Model's probability for this outcome
            result: "win", "loss", or "push"
            profit: Profit/loss amount
        """
        # Calculate implied probability and edge
        implied_prob = 1 / odds
        edge = predicted_prob - implied_prob

        record = {
            "timestamp": datetime.now().isoformat(),
            "match_id": match_id,
            "bet_type": bet_type,
            "stake": stake,
            "odds": odds,
            "predicted_prob": predicted_prob,
            "implied_prob": implied_prob,
            "edge": edge,
            "result": result,
            "profit": profit,
        }

        self.bets.append(record)
        self.save_all()

    def update_prediction_result(self, match_id: str, actual_class: int):
        """Update prediction with actual result."""
        for pred in self.predictions:
            if pred["match_id"] == match_id and pred["actual_class"] is None:
                pred["actual_class"] = actual_class
                pred["correct"] = pred["predicted_class"] == actual_class

                # Update daily stats
                date_key = pred["timestamp"][:10]
                if date_key in self.daily_stats and pred["correct"]:
                    self.daily_stats[date_key]["correct"] += 1
                    if pred["confidence_score"] >= 0.5:
                        self.daily_stats[date_key]["high_conf_correct"] += 1
                break

        self.save_all()

    def save_all(self):
        """Save all data to files."""
        self._save_json(self.predictions_file, self.predictions[-1000:])  # Keep last 1000
        self._save_json(self.bets_file, self.bets)
        self._save_json(self.daily_stats_file, self.daily_stats)


class PerformanceDashboard:
    """Performance monitoring dashboard."""

    def __init__(self, metrics_collector: MetricsCollector = None):
        self.collector = metrics_collector or MetricsCollector()
        self.alerts: List[Dict] = []

    def get_accuracy_summary(self, days: int = 30) -> Dict:
        """Get accuracy summary for recent predictions.

        Args:
            days: Number of days to include

        Returns:
            Dict with accuracy metrics
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent = [p for p in self.collector.predictions if p["timestamp"] >= cutoff and p["actual_class"] is not None]

        if not recent:
            return {"message": "No completed predictions in period"}

        correct = sum(1 for p in recent if p["correct"])
        total = len(recent)

        # By confidence
        high_conf = [p for p in recent if p["confidence_score"] >= 0.5]
        high_conf_correct = sum(1 for p in high_conf if p["correct"])

        # By outcome
        by_outcome = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}, 2: {"correct": 0, "total": 0}}
        for p in recent:
            pred = p["predicted_class"]
            by_outcome[pred]["total"] += 1
            if p["correct"]:
                by_outcome[pred]["correct"] += 1

        return {
            "period_days": days,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": correct / total if total > 0 else 0,
            "high_confidence": {
                "total": len(high_conf),
                "correct": high_conf_correct,
                "accuracy": high_conf_correct / len(high_conf) if high_conf else 0,
            },
            "by_outcome": {
                "home_win": {
                    "accuracy": by_outcome[0]["correct"] / by_outcome[0]["total"] if by_outcome[0]["total"] > 0 else 0,
                    "count": by_outcome[0]["total"],
                },
                "draw": {
                    "accuracy": by_outcome[1]["correct"] / by_outcome[1]["total"] if by_outcome[1]["total"] > 0 else 0,
                    "count": by_outcome[1]["total"],
                },
                "away_win": {
                    "accuracy": by_outcome[2]["correct"] / by_outcome[2]["total"] if by_outcome[2]["total"] > 0 else 0,
                    "count": by_outcome[2]["total"],
                },
            },
        }

    def get_betting_pnl(self, days: int = 30) -> Dict:
        """Get betting profit/loss summary.

        Args:
            days: Number of days to include

        Returns:
            Dict with P&L metrics
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent = [b for b in self.collector.bets if b["timestamp"] >= cutoff and b["result"] is not None]

        if not recent:
            return {"message": "No completed bets in period"}

        total_stake = sum(b["stake"] for b in recent)
        total_profit = sum(b.get("profit", 0) for b in recent)
        wins = sum(1 for b in recent if b["result"] == "win")
        total_bets = len(recent)

        # ROI calculation
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

        # By edge bucket
        edge_buckets = {
            "negative": {"bets": 0, "profit": 0},
            "small": {"bets": 0, "profit": 0},  # 0-5%
            "medium": {"bets": 0, "profit": 0},  # 5-10%
            "large": {"bets": 0, "profit": 0},  # 10%+
        }

        for b in recent:
            edge = b.get("edge", 0)
            profit = b.get("profit", 0)
            if edge < 0:
                edge_buckets["negative"]["bets"] += 1
                edge_buckets["negative"]["profit"] += profit
            elif edge < 0.05:
                edge_buckets["small"]["bets"] += 1
                edge_buckets["small"]["profit"] += profit
            elif edge < 0.10:
                edge_buckets["medium"]["bets"] += 1
                edge_buckets["medium"]["profit"] += profit
            else:
                edge_buckets["large"]["bets"] += 1
                edge_buckets["large"]["profit"] += profit

        return {
            "period_days": days,
            "total_bets": total_bets,
            "wins": wins,
            "win_rate": wins / total_bets if total_bets > 0 else 0,
            "total_stake": total_stake,
            "total_profit": total_profit,
            "roi_percent": roi,
            "by_edge": edge_buckets,
        }

    def get_daily_trend(self, days: int = 14) -> List[Dict]:
        """Get daily accuracy trend.

        Args:
            days: Number of days to include

        Returns:
            List of daily stats
        """
        trend = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            stats = self.collector.daily_stats.get(date, {})

            if stats.get("predictions", 0) > 0:
                trend.append({
                    "date": date,
                    "predictions": stats["predictions"],
                    "accuracy": stats["correct"] / stats["predictions"],
                    "high_conf_accuracy": stats["high_conf_correct"] / stats["high_conf_predictions"]
                        if stats.get("high_conf_predictions", 0) > 0 else None,
                })

        return list(reversed(trend))

    def get_model_comparison(self) -> Dict:
        """Compare performance across model versions."""
        by_version = {}

        for p in self.collector.predictions:
            if p["actual_class"] is None:
                continue

            version = p.get("model_version", "unknown")
            if version not in by_version:
                by_version[version] = {"correct": 0, "total": 0}

            by_version[version]["total"] += 1
            if p["correct"]:
                by_version[version]["correct"] += 1

        return {
            version: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "predictions": stats["total"],
            }
            for version, stats in by_version.items()
        }

    def check_alerts(self) -> List[Dict]:
        """Check for alert conditions.

        Returns:
            List of active alerts
        """
        self.alerts = []

        # Check recent accuracy
        summary = self.get_accuracy_summary(days=7)
        if summary.get("accuracy", 1) < 0.40:
            self.alerts.append({
                "severity": "high",
                "type": "low_accuracy",
                "message": f"7-day accuracy is {summary['accuracy']:.1%}, below 40% threshold",
                "timestamp": datetime.now().isoformat(),
            })
        elif summary.get("accuracy", 1) < 0.45:
            self.alerts.append({
                "severity": "medium",
                "type": "declining_accuracy",
                "message": f"7-day accuracy is {summary['accuracy']:.1%}, below 45% warning threshold",
                "timestamp": datetime.now().isoformat(),
            })

        # Check high confidence accuracy
        if summary.get("high_confidence", {}).get("accuracy", 1) < 0.50:
            self.alerts.append({
                "severity": "high",
                "type": "low_high_conf_accuracy",
                "message": f"High confidence accuracy is {summary['high_confidence']['accuracy']:.1%}, below 50%",
                "timestamp": datetime.now().isoformat(),
            })

        # Check betting ROI
        pnl = self.get_betting_pnl(days=7)
        if pnl.get("roi_percent", 0) < -10:
            self.alerts.append({
                "severity": "high",
                "type": "negative_roi",
                "message": f"7-day ROI is {pnl['roi_percent']:.1f}%, significant loss",
                "timestamp": datetime.now().isoformat(),
            })

        return self.alerts

    def get_full_report(self) -> Dict:
        """Generate comprehensive performance report.

        Returns:
            Dict with all dashboard metrics
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "accuracy_7d": self.get_accuracy_summary(days=7),
            "accuracy_30d": self.get_accuracy_summary(days=30),
            "betting_pnl_7d": self.get_betting_pnl(days=7),
            "betting_pnl_30d": self.get_betting_pnl(days=30),
            "daily_trend": self.get_daily_trend(days=14),
            "model_comparison": self.get_model_comparison(),
            "alerts": self.check_alerts(),
        }

    def print_summary(self):
        """Print human-readable summary to console."""
        report = self.get_full_report()

        print("\n" + "=" * 60)
        print("PERFORMANCE DASHBOARD")
        print("=" * 60)

        # Accuracy
        acc_7d = report["accuracy_7d"]
        if "accuracy" in acc_7d:
            print(f"\n7-Day Accuracy: {acc_7d['accuracy']:.1%} ({acc_7d['correct_predictions']}/{acc_7d['total_predictions']})")
            hc = acc_7d.get("high_confidence", {})
            if hc.get("total", 0) > 0:
                print(f"  High Conf: {hc['accuracy']:.1%} ({hc['correct']}/{hc['total']})")

        acc_30d = report["accuracy_30d"]
        if "accuracy" in acc_30d:
            print(f"\n30-Day Accuracy: {acc_30d['accuracy']:.1%} ({acc_30d['correct_predictions']}/{acc_30d['total_predictions']})")

        # Betting
        pnl = report["betting_pnl_7d"]
        if "roi_percent" in pnl:
            print(f"\n7-Day Betting: ROI {pnl['roi_percent']:+.1f}% (Profit: {pnl['total_profit']:+.2f})")

        # Alerts
        alerts = report["alerts"]
        if alerts:
            print(f"\nALERTS ({len(alerts)}):")
            for alert in alerts:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")
        else:
            print("\nNo alerts - all systems nominal")

        print("\n" + "=" * 60)


class AlertManager:
    """Manage and dispatch alerts."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DATA_DIR / "monitoring" / "alerts"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.alerts_file = self.storage_dir / "alerts.json"
        self.alerts: List[Dict] = self._load_alerts()
        self.handlers: List[callable] = []

    def _load_alerts(self) -> List[Dict]:
        """Load alert history."""
        if self.alerts_file.exists():
            with open(self.alerts_file) as f:
                return json.load(f)
        return []

    def _save_alerts(self):
        """Save alert history."""
        with open(self.alerts_file, "w") as f:
            json.dump(self.alerts[-500:], f, indent=2)

    def add_handler(self, handler: callable):
        """Add alert handler function.

        Handler should accept: (severity, alert_type, message)
        """
        self.handlers.append(handler)

    def create_alert(
        self,
        severity: str,
        alert_type: str,
        message: str,
        metadata: Dict = None,
    ):
        """Create and dispatch an alert.

        Args:
            severity: "low", "medium", "high", "critical"
            alert_type: Type identifier
            message: Human-readable message
            metadata: Additional context
        """
        alert = {
            "id": len(self.alerts) + 1,
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "type": alert_type,
            "message": message,
            "metadata": metadata or {},
            "acknowledged": False,
        }

        self.alerts.append(alert)
        self._save_alerts()

        # Dispatch to handlers
        for handler in self.handlers:
            try:
                handler(severity, alert_type, message)
            except Exception as e:
                log.error(f"Alert handler failed: {e}")

        # Log
        log_fn = log.critical if severity == "critical" else log.error if severity == "high" else log.warning
        log_fn(f"ALERT [{alert_type}]: {message}")

    def acknowledge_alert(self, alert_id: int):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                self._save_alerts()
                return True
        return False

    def get_active_alerts(self) -> List[Dict]:
        """Get unacknowledged alerts."""
        return [a for a in self.alerts if not a["acknowledged"]]

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from recent period."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        return [a for a in self.alerts if a["timestamp"] >= cutoff]
