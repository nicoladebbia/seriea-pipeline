#!/usr/bin/env python3
"""Train Deep Learning Models - Phase 5 Implementation.

Trains LSTM and Transformer models on historical match data
for integration into the ensemble prediction system.

Usage:
    python scripts/train_deep_models.py
    python scripts/train_deep_models.py --epochs 100
    python scripts/train_deep_models.py --quick  # Fast training for testing
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Run: pip install torch")
    sys.exit(1)

from config.settings import DATA_DIR, MODELS_DIR
from models.deep_learning import (
    LSTMFormModel,
    TransformerMatchModel,
    MatchSequenceDataset,
    DeepModelTrainer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    """Load historical match data for training."""
    features_path = DATA_DIR / "features" / "features.parquet"

    if not features_path.exists():
        log.error(f"Features file not found: {features_path}")
        return None

    df = pd.read_parquet(features_path)
    df = df.sort_values("match_date").reset_index(drop=True)

    # Filter to matches with scores
    df = df.dropna(subset=["home_score", "away_score"])

    log.info(f"Loaded {len(df)} matches for training")
    return df


def prepare_dataloaders(
    df: pd.DataFrame,
    sequence_length: int = 10,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> tuple:
    """Prepare train/val/test dataloaders."""
    log.info("Preparing datasets...")

    # Create dataset
    dataset = MatchSequenceDataset(df, sequence_length=sequence_length)

    log.info(f"Created dataset with {len(dataset)} samples")

    # Chronological split (NOT random — preserves temporal order)
    total = len(dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - test_size - val_size

    # Use Subset with sequential indices to maintain time order
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, total))

    log.info(f"Split: Train={train_size}, Val={val_size}, Test={test_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def evaluate_model(model, test_loader, device: str = "cpu") -> dict:
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            _, predicted = outputs.max(1)

            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = correct / total

    # Calculate per-class metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # High confidence accuracy
    max_probs = all_probs.max(axis=1)
    high_conf_mask = max_probs > 0.5
    if high_conf_mask.sum() > 0:
        high_conf_preds = all_probs[high_conf_mask].argmax(axis=1)
        high_conf_labels = all_labels[high_conf_mask]
        high_conf_acc = (high_conf_preds == high_conf_labels).mean()
    else:
        high_conf_acc = 0

    return {
        "accuracy": accuracy,
        "high_conf_accuracy": high_conf_acc,
        "high_conf_count": int(high_conf_mask.sum()),
        "total": total,
    }


def train_all_models(
    epochs: int = 50,
    batch_size: int = 32,
    sequence_length: int = 10,
    quick: bool = False,
) -> dict:
    """Train all deep learning models."""
    log.info("=" * 70)
    log.info("DEEP LEARNING MODEL TRAINING - PHASE 5")
    log.info("=" * 70)

    if quick:
        epochs = 10
        batch_size = 64
        log.info("Quick mode: 10 epochs")

    # Load data
    df = load_training_data()
    if df is None:
        return None

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        df,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    # Initialize trainer
    trainer = DeepModelTrainer()
    results = {"timestamp": datetime.now().isoformat()}

    # Train LSTM
    log.info("\n" + "=" * 50)
    log.info("Training LSTM Model")
    log.info("=" * 50)

    lstm_history = trainer.train_lstm(
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.001,
    )

    # Evaluate LSTM
    lstm_metrics = evaluate_model(trainer.lstm_model, test_loader)
    log.info(f"LSTM Test Accuracy: {lstm_metrics['accuracy']:.1%}")
    log.info(f"LSTM High Conf Accuracy: {lstm_metrics['high_conf_accuracy']:.1%} "
             f"({lstm_metrics['high_conf_count']} samples)")

    results["lstm"] = {
        "final_train_loss": lstm_history["train_loss"][-1],
        "final_val_loss": lstm_history["val_loss"][-1] if lstm_history["val_loss"] else None,
        "final_val_acc": lstm_history["val_acc"][-1] if lstm_history["val_acc"] else None,
        "test_accuracy": lstm_metrics["accuracy"],
        "test_high_conf_accuracy": lstm_metrics["high_conf_accuracy"],
    }

    # Train Transformer
    log.info("\n" + "=" * 50)
    log.info("Training Transformer Model")
    log.info("=" * 50)

    transformer_history = trainer.train_transformer(
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.0005,
    )

    # Evaluate Transformer
    transformer_metrics = evaluate_model(trainer.transformer_model, test_loader)
    log.info(f"Transformer Test Accuracy: {transformer_metrics['accuracy']:.1%}")
    log.info(f"Transformer High Conf Accuracy: {transformer_metrics['high_conf_accuracy']:.1%}")

    results["transformer"] = {
        "final_train_loss": transformer_history["train_loss"][-1],
        "final_val_loss": transformer_history["val_loss"][-1] if transformer_history["val_loss"] else None,
        "final_val_acc": transformer_history["val_acc"][-1] if transformer_history["val_acc"] else None,
        "test_accuracy": transformer_metrics["accuracy"],
        "test_high_conf_accuracy": transformer_metrics["high_conf_accuracy"],
    }

    # Save models
    trainer.save_models()

    # Save results
    results_path = MODELS_DIR / "deep" / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nResults saved to {results_path}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("TRAINING SUMMARY")
    log.info("=" * 70)
    log.info(f"LSTM Accuracy: {lstm_metrics['accuracy']:.1%} "
             f"(HC: {lstm_metrics['high_conf_accuracy']:.1%})")
    log.info(f"Transformer Accuracy: {transformer_metrics['accuracy']:.1%} "
             f"(HC: {transformer_metrics['high_conf_accuracy']:.1%})")

    # Average
    avg_acc = (lstm_metrics['accuracy'] + transformer_metrics['accuracy']) / 2
    avg_hc = (lstm_metrics['high_conf_accuracy'] + transformer_metrics['high_conf_accuracy']) / 2
    log.info(f"Ensemble Average: {avg_acc:.1%} (HC: {avg_hc:.1%})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train deep learning models")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=10, help="History length")
    parser.add_argument("--quick", action="store_true", help="Quick training (10 epochs)")
    args = parser.parse_args()

    results = train_all_models(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        quick=args.quick,
    )

    if results:
        print("\n" + "=" * 50)
        print("Training complete!")
        print("=" * 50)


if __name__ == "__main__":
    main()
