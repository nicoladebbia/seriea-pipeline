#!/usr/bin/env python3
"""Feature Importance Analysis - Phase 1.3

Analyzes feature importance using SHAP and identifies noisy features.

Usage:
    python scripts/feature_importance_analysis.py
    python scripts/feature_importance_analysis.py --top 50  # Show top 50 features
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Analyze feature importance for ML models."""

    def __init__(self):
        self.df = None
        self.model = None
        self.feature_cols = []
        self.importance_scores = {}

    def load_data(self) -> bool:
        """Load feature data."""
        features_path = DATA_DIR / "features" / "features.parquet"
        if not features_path.exists():
            log.error(f"Features not found: {features_path}")
            return False

        self.df = pd.read_parquet(features_path)
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        self.df = self.df.dropna(subset=["result"])

        log.info(f"Loaded {len(self.df)} matches with {len(self.df.columns)} columns")
        return True

    def load_model(self) -> bool:
        """Load the ML classifier model."""
        if not CATBOOST_AVAILABLE:
            log.error("CatBoost not available")
            return False

        model_path = MODELS_DIR / "universal" / "classifier_extended.cbm"
        if not model_path.exists():
            log.error(f"Model not found: {model_path}")
            return False

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))

        # Get feature names from model
        self.feature_cols = self.model.feature_names_

        log.info(f"Loaded model with {len(self.feature_cols)} features")
        return True

    def get_catboost_importance(self) -> Dict[str, float]:
        """Get feature importance from CatBoost model."""
        if self.model is None:
            return {}

        # Get built-in feature importance
        importance = self.model.get_feature_importance()
        feature_importance = dict(zip(self.feature_cols, importance))

        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return sorted_importance

    def get_shap_importance(self, sample_size: int = 1000) -> Dict[str, float]:
        """Get SHAP-based feature importance."""
        if not SHAP_AVAILABLE or self.model is None:
            log.warning("SHAP analysis not available")
            return {}

        log.info(f"Computing SHAP values on {sample_size} samples...")

        # Prepare data
        X = self.df[self.feature_cols].fillna(0)
        X = X.tail(sample_size)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        # For multiclass, average across classes
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        feature_importance = dict(zip(self.feature_cols, mean_shap))

        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return sorted_importance

    def get_permutation_importance(self, sample_size: int = 500) -> Dict[str, float]:
        """Get permutation-based feature importance."""
        log.info(f"Computing permutation importance on {sample_size} samples...")

        # Prepare data
        test_df = self.df.tail(sample_size).copy()
        X = test_df[self.feature_cols].fillna(0)

        # Map result to numeric
        result_map = {"H": 0, "D": 1, "A": 2}
        y = test_df["result"].map(result_map).values

        # Baseline accuracy
        baseline_preds = np.array(self.model.predict(X))
        if baseline_preds.ndim > 1:
            baseline_preds = baseline_preds.argmax(axis=1)
        baseline_acc = (baseline_preds == y).mean()

        importance = {}
        for col in self.feature_cols:
            # Shuffle feature
            X_shuffled = X.copy()
            X_shuffled[col] = np.random.permutation(X_shuffled[col].values)

            # Measure accuracy drop
            shuffled_preds = np.array(self.model.predict(X_shuffled))
            if shuffled_preds.ndim > 1:
                shuffled_preds = shuffled_preds.argmax(axis=1)
            shuffled_acc = (shuffled_preds == y).mean()

            # Importance = accuracy drop when shuffled
            importance[col] = baseline_acc - shuffled_acc

        # Sort by importance
        sorted_importance = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return sorted_importance

    def analyze_feature_groups(self) -> Dict[str, Dict]:
        """Analyze importance by feature group."""
        groups = {
            "elo_strength": [],
            "form_rolling": [],
            "understat_xg": [],
            "momentum": [],
            "rest_congestion": [],
            "league_position": [],
            "h2h": [],
            "venue": [],
            "derived": [],
            "other": [],
        }

        # Categorize features
        for col in self.feature_cols:
            col_lower = col.lower()
            if "elo" in col_lower or "strength" in col_lower:
                groups["elo_strength"].append(col)
            elif "roll_" in col_lower or "form" in col_lower:
                groups["form_rolling"].append(col)
            elif "us_" in col_lower or "xg" in col_lower or "xa" in col_lower:
                groups["understat_xg"].append(col)
            elif "streak" in col_lower or "unbeaten" in col_lower or "winless" in col_lower:
                groups["momentum"].append(col)
            elif "rest" in col_lower or "congestion" in col_lower:
                groups["rest_congestion"].append(col)
            elif "league" in col_lower and ("pos" in col_lower or "point" in col_lower):
                groups["league_position"].append(col)
            elif "h2h" in col_lower:
                groups["h2h"].append(col)
            elif "venue" in col_lower or "stadium" in col_lower or "travel" in col_lower:
                groups["venue"].append(col)
            elif "adj_" in col_lower or "gd_" in col_lower or "opp_" in col_lower:
                groups["derived"].append(col)
            else:
                groups["other"].append(col)

        # Calculate group importance
        catboost_imp = self.get_catboost_importance()

        group_results = {}
        for group_name, features in groups.items():
            if not features:
                continue

            group_importance = sum(catboost_imp.get(f, 0) for f in features)
            avg_importance = group_importance / len(features) if features else 0

            group_results[group_name] = {
                "features": features,
                "count": len(features),
                "total_importance": group_importance,
                "avg_importance": avg_importance,
            }

        return group_results

    def identify_low_importance_features(
        self,
        threshold_percentile: float = 20
    ) -> List[str]:
        """Identify features with low importance."""
        catboost_imp = self.get_catboost_importance()

        if not catboost_imp:
            return []

        importances = list(catboost_imp.values())
        threshold = np.percentile(importances, threshold_percentile)

        low_importance = [
            f for f, imp in catboost_imp.items()
            if imp <= threshold
        ]

        return low_importance

    def run_full_analysis(self) -> Dict:
        """Run complete feature importance analysis."""
        log.info("Running full feature importance analysis...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_features": len(self.feature_cols),
        }

        # CatBoost importance
        results["catboost_importance"] = self.get_catboost_importance()

        # SHAP importance (if available)
        if SHAP_AVAILABLE:
            results["shap_importance"] = self.get_shap_importance(sample_size=500)
        else:
            results["shap_importance"] = {}

        # Permutation importance
        results["permutation_importance"] = self.get_permutation_importance(sample_size=300)

        # Group analysis
        results["group_analysis"] = self.analyze_feature_groups()

        # Low importance features
        results["low_importance_features"] = self.identify_low_importance_features(threshold_percentile=20)

        # Top features consensus
        top_catboost = list(results["catboost_importance"].keys())[:30]
        top_shap = list(results["shap_importance"].keys())[:30] if results["shap_importance"] else []
        top_perm = list(results["permutation_importance"].keys())[:30]

        # Features appearing in all top lists
        consensus = set(top_catboost)
        if top_shap:
            consensus &= set(top_shap)
        consensus &= set(top_perm)

        results["consensus_top_features"] = list(consensus)

        return results


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance")
    parser.add_argument("--top", type=int, default=30, help="Number of top features to show")
    parser.add_argument("--save-reduced", action="store_true", help="Save reduced feature list")
    args = parser.parse_args()

    analyzer = FeatureImportanceAnalyzer()

    if not analyzer.load_data():
        return

    if not analyzer.load_model():
        return

    results = analyzer.run_full_analysis()

    # Print results
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal features: {results['n_features']}")

    # Top CatBoost features
    print(f"\nTOP {args.top} FEATURES (CatBoost Importance)")
    print("-" * 60)
    for i, (feature, importance) in enumerate(list(results["catboost_importance"].items())[:args.top]):
        print(f"{i+1:3}. {feature:50} {importance:10.4f}")

    # Group analysis
    print("\n" + "=" * 70)
    print("FEATURE GROUP ANALYSIS")
    print("=" * 70)
    group_results = results["group_analysis"]
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]["total_importance"], reverse=True)

    for group_name, group_data in sorted_groups:
        print(f"\n{group_name.upper()}")
        print(f"  Features: {group_data['count']}")
        print(f"  Total Importance: {group_data['total_importance']:.4f}")
        print(f"  Avg Importance: {group_data['avg_importance']:.4f}")

    # Low importance features
    low_imp = results["low_importance_features"]
    print("\n" + "=" * 70)
    print("LOW IMPORTANCE FEATURES (Bottom 20%)")
    print("=" * 70)
    print(f"\nCount: {len(low_imp)} features")
    print("\nFeatures to consider removing:")
    for f in low_imp[:20]:
        imp = results["catboost_importance"].get(f, 0)
        print(f"  - {f} (importance: {imp:.4f})")

    # Consensus top features
    print("\n" + "=" * 70)
    print("CONSENSUS TOP FEATURES (appear in all importance methods)")
    print("=" * 70)
    for f in results["consensus_top_features"][:15]:
        print(f"  - {f}")

    # Save results
    results_dir = DATA_DIR / "optimization"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert to JSON-serializable format
    json_results = {
        k: v if not isinstance(v, (np.ndarray, np.floating)) else float(v)
        for k, v in results.items()
    }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Save reduced feature list if requested
    if args.save_reduced:
        # Keep top 80% features
        keep_features = [f for f in results["catboost_importance"].keys()
                        if f not in low_imp]

        reduced_file = MODELS_DIR / "reduced_features.json"
        with open(reduced_file, "w") as f:
            json.dump({"features": keep_features, "original_count": len(analyzer.feature_cols),
                      "reduced_count": len(keep_features)}, f, indent=2)
        print(f"Reduced feature list saved to: {reduced_file}")


if __name__ == "__main__":
    main()
