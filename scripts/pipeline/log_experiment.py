#!/usr/bin/env python3
"""Experiment tracking utility for the Serie A prediction pipeline.

Usage:
    python3 scripts/log_experiment.py \
        --description "Added xc_home_p4 feature" \
        --type feature \
        --model catboost_upcoming.cbm \
        --features 125 \
        --accuracy 0.556 \
        --log-loss 0.9494 \
        --brier 0.5660 \
        --yield 0.263 \
        --result SHIP \
        --files "features/build.py,scripts/ensemble_prediction_engine.py" \
        --notes "Walk-forward validated"

    python3 scripts/log_experiment.py --show  # Show last 10 experiments
    python3 scripts/log_experiment.py --compare  # Compare last experiment to baseline
"""

import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENT_LOG = os.path.join(PROJECT_ROOT, "data", "experiments", "experiment_log.jsonl")
DEPLOYMENT_STATE = os.path.join(PROJECT_ROOT, "data", "models", "deployment_state.json")


def log_experiment(args):
    """Append an experiment entry to the log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "description": args.description,
        "type": args.type,
        "metrics": {
            "accuracy": args.accuracy,
            "log_loss": args.log_loss,
            "brier": args.brier,
            "betting_yield": getattr(args, "yield_val", None),
        },
        "model": args.model,
        "features": args.features,
        "result": args.result,
        "files_modified": args.files.split(",") if args.files else [],
        "notes": args.notes or "",
    }

    os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Logged experiment: {entry['experiment_id']}")
    print(f"  Description: {entry['description']}")
    print(f"  Result: {entry['result']}")
    print(f"  Metrics: Acc={args.accuracy}, LL={args.log_loss}, Brier={args.brier}")

    # If SHIP, update deployment state
    if args.result == "SHIP" and args.accuracy and args.log_loss:
        update_deployment_state(entry)


def update_deployment_state(entry):
    """Update deployment_state.json when an experiment is shipped."""
    try:
        with open(DEPLOYMENT_STATE) as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {"history": []}

    # Save current as rollback target
    state["rollback"] = {
        "available": True,
        "rollback_model": state.get("current_model"),
        "rollback_reason": None,
        "last_rollback_at": None,
    }

    # Update current
    state["current_model"] = entry.get("model", state.get("current_model"))
    state["features_count"] = entry.get("features", state.get("features_count"))
    state["deployed_at"] = entry["timestamp"]
    state["metrics"] = entry["metrics"]

    # Append to history
    state.setdefault("history", []).append({
        "timestamp": entry["timestamp"],
        "action": "deploy",
        "model": entry.get("model"),
        "experiment_id": entry["experiment_id"],
        "metrics": entry["metrics"],
    })

    with open(DEPLOYMENT_STATE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"  Updated deployment_state.json → {entry.get('model')}")


def show_experiments(n=10):
    """Show last N experiments."""
    try:
        with open(EXPERIMENT_LOG) as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print("No experiments logged yet.")
        return

    for entry in entries[-n:]:
        metrics = entry.get("metrics", {})
        result = entry.get("result", "?")
        icon = "✅" if result in ("SHIP", "BASELINE") else "❌" if result == "REVERT" else "⏳"
        print(f"{icon} [{entry.get('timestamp', '?')[:16]}] {entry.get('description', '?')}")
        print(f"   Result: {result} | Acc={metrics.get('accuracy', '?')} | LL={metrics.get('log_loss', '?')}")
        print()


def compare_to_baseline():
    """Compare last experiment to baseline."""
    try:
        with open(EXPERIMENT_LOG) as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print("No experiments logged yet.")
        return

    baseline = next((e for e in entries if e.get("type") == "baseline"), None)
    if not baseline:
        print("No baseline found.")
        return

    last = entries[-1]
    if last == baseline:
        print("Only baseline exists — no experiments to compare.")
        return

    bm = baseline.get("metrics", {})
    lm = last.get("metrics", {})

    print(f"Baseline: {baseline.get('description', '?')}")
    print(f"Latest:   {last.get('description', '?')}")
    print()
    print(f"{'Metric':<15} {'Baseline':>10} {'Latest':>10} {'Delta':>10} {'Status':>8}")
    print("-" * 55)

    for key, label, better in [
        ("accuracy", "Accuracy", "higher"),
        ("log_loss", "Log-loss", "lower"),
        ("brier", "Brier", "lower"),
        ("betting_yield", "Yield", "higher"),
    ]:
        bv = bm.get(key)
        lv = lm.get(key)
        if bv is not None and lv is not None:
            delta = lv - bv
            is_better = (delta > 0 and better == "higher") or (delta < 0 and better == "lower")
            icon = "✅" if is_better else "❌"
            print(f"{label:<15} {bv:>10.4f} {lv:>10.4f} {delta:>+10.4f} {icon:>8}")


def main():
    parser = argparse.ArgumentParser(description="Serie A experiment tracker")
    parser.add_argument("--show", action="store_true", help="Show last 10 experiments")
    parser.add_argument("--compare", action="store_true", help="Compare last to baseline")
    parser.add_argument("--description", type=str, help="Experiment description")
    parser.add_argument("--type", type=str, choices=["feature", "model", "calibration", "ensemble", "bugfix", "baseline"], default="model")
    parser.add_argument("--model", type=str, default="catboost_upcoming.cbm")
    parser.add_argument("--features", type=int, default=125)
    parser.add_argument("--accuracy", type=float)
    parser.add_argument("--log-loss", type=float)
    parser.add_argument("--brier", type=float)
    parser.add_argument("--yield", type=float, dest="yield_val")
    parser.add_argument("--result", type=str, choices=["SHIP", "REVERT", "PENDING", "BASELINE"], default="PENDING")
    parser.add_argument("--files", type=str, help="Comma-separated list of modified files")
    parser.add_argument("--notes", type=str)

    args = parser.parse_args()

    if args.show:
        show_experiments()
    elif args.compare:
        compare_to_baseline()
    elif args.description:
        log_experiment(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
