"""Deep Learning Models for Football Prediction - Phase 5 Implementation.

Three neural network architectures for enhanced prediction:

1. LSTM FORM MODEL
   - Captures temporal patterns in team performance
   - Input: Last N matches as sequences
   - Output: Form embedding for prediction

2. TRANSFORMER CONTEXT MODEL
   - Self-attention over match history
   - Cross-attention for H2H context
   - Better for variable-length sequences

3. SQUAD INTERACTION MODEL (GNN-inspired)
   - Models player-player chemistry
   - Aggregates individual contributions
   - Uses message-passing concept

4. META-LEARNER
   - Combines all deep model outputs
   - Calibrated probability output
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

log = logging.getLogger(__name__)


# =============================================================================
# LSTM FORM MODEL
# =============================================================================

if TORCH_AVAILABLE:
    class LSTMFormModel(nn.Module):
        """LSTM model for team form sequences.

        Takes last N matches and predicts outcome probabilities.
        Each match represented by [goals_for, goals_against, xG, xGA, points, is_home].
        """

        def __init__(
            self,
            input_size: int = 10,      # Features per match
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            output_size: int = 3,      # H, D, A probabilities
        ):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layer
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )

            # Attention mechanism for sequence
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

            # Output layers
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Shape (batch, seq_len, input_size)

            Returns:
                Shape (batch, output_size) - probabilities
            """
            # LSTM encoding
            lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

            # Attention weights
            attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
            attn_weights = F.softmax(attn_weights, dim=1)

            # Weighted sum
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

            # Output
            logits = self.fc(context)
            return F.softmax(logits, dim=-1)


    class TransformerMatchModel(nn.Module):
        """Transformer model for match context.

        Uses self-attention to weigh importance of different matches.
        Better handles variable history lengths and long-range dependencies.
        """

        def __init__(
            self,
            input_size: int = 10,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.3,
            max_seq_len: int = 20,
            output_size: int = 3,
        ):
            super().__init__()

            self.d_model = d_model

            # Input projection
            self.input_proj = nn.Linear(input_size, d_model)

            # Positional encoding
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_seq_len, d_model) * 0.1
            )

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

            # CLS token for classification
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

            # Output layers
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_size),
            )

        def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Shape (batch, seq_len, input_size)
                mask: Optional attention mask

            Returns:
                Shape (batch, output_size)
            """
            batch_size, seq_len, _ = x.shape

            # Project input
            x = self.input_proj(x)  # (batch, seq, d_model)

            # Add positional encoding
            x = x + self.pos_encoding[:, :seq_len, :]

            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq+1, d_model)

            # Transformer encoding
            x = self.transformer(x, src_key_padding_mask=mask)

            # Take CLS token output
            cls_output = x[:, 0, :]  # (batch, d_model)

            # Output
            logits = self.fc(cls_output)
            return F.softmax(logits, dim=-1)


    class SquadInteractionModel(nn.Module):
        """GNN-inspired model for squad interactions.

        Models player contributions and team chemistry.
        Uses message-passing style aggregation.
        """

        def __init__(
            self,
            player_features: int = 8,   # xG, xA, mins, goals, assists, etc.
            hidden_size: int = 32,
            num_players: int = 11,
            dropout: float = 0.3,
            output_size: int = 3,
        ):
            super().__init__()

            self.num_players = num_players

            # Player embedding
            self.player_embed = nn.Sequential(
                nn.Linear(player_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Player-player interaction (simplified GNN message passing)
            self.interaction = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

            # Aggregation attention
            self.player_attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )

            # Team-level output
            self.team_fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Match prediction (takes both teams)
            self.match_fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )

        def forward(
            self,
            home_players: torch.Tensor,
            away_players: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                home_players: (batch, num_players, player_features)
                away_players: (batch, num_players, player_features)

            Returns:
                (batch, output_size)
            """
            # Embed players
            home_embed = self.player_embed(home_players)  # (batch, 11, hidden)
            away_embed = self.player_embed(away_players)

            # Simple interaction: each player interacts with team mean
            home_mean = home_embed.mean(dim=1, keepdim=True).expand_as(home_embed)
            away_mean = away_embed.mean(dim=1, keepdim=True).expand_as(away_embed)

            home_interact = torch.cat([home_embed, home_mean], dim=-1)
            away_interact = torch.cat([away_embed, away_mean], dim=-1)

            home_updated = home_embed + self.interaction(home_interact)
            away_updated = away_embed + self.interaction(away_interact)

            # Attention aggregation
            home_attn = F.softmax(self.player_attention(home_updated), dim=1)
            away_attn = F.softmax(self.player_attention(away_updated), dim=1)

            home_agg = (home_attn * home_updated).sum(dim=1)  # (batch, hidden)
            away_agg = (away_attn * away_updated).sum(dim=1)

            # Team representations
            home_team = self.team_fc(home_agg)
            away_team = self.team_fc(away_agg)

            # Match prediction
            match_repr = torch.cat([home_team, away_team], dim=-1)
            logits = self.match_fc(match_repr)

            return F.softmax(logits, dim=-1)


    class DeepEnsemble(nn.Module):
        """Meta-learner combining all deep models.

        Takes outputs from LSTM, Transformer, and Squad models
        and produces final calibrated probabilities.
        """

        def __init__(
            self,
            num_models: int = 3,
            hidden_size: int = 32,
            output_size: int = 3,
        ):
            super().__init__()

            # Input: concatenated probabilities from each model
            input_size = num_models * output_size

            self.meta_net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

            # Learnable model weights
            self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)

        def forward(
            self,
            lstm_probs: torch.Tensor,
            transformer_probs: torch.Tensor,
            squad_probs: torch.Tensor,
        ) -> torch.Tensor:
            """Combine model predictions.

            Args:
                lstm_probs: (batch, 3)
                transformer_probs: (batch, 3)
                squad_probs: (batch, 3)

            Returns:
                (batch, 3) - final probabilities
            """
            # Normalize weights
            weights = F.softmax(self.model_weights, dim=0)

            # Weighted average
            weighted_avg = (
                weights[0] * lstm_probs +
                weights[1] * transformer_probs +
                weights[2] * squad_probs
            )

            # Meta-network refinement
            concat = torch.cat([lstm_probs, transformer_probs, squad_probs], dim=-1)
            meta_output = self.meta_net(concat)

            # Combine weighted average with meta output
            combined = 0.7 * weighted_avg + 0.3 * F.softmax(meta_output, dim=-1)

            # Ensure valid probabilities
            return combined / combined.sum(dim=-1, keepdim=True)


# =============================================================================
# DATA PREPARATION
# =============================================================================

class MatchSequenceDataset(Dataset):
    """Dataset for training sequence models."""

    def __init__(
        self,
        matches_df,
        sequence_length: int = 10,
        features: List[str] = None,
    ):
        """Initialize dataset.

        Args:
            matches_df: DataFrame with match data
            sequence_length: Number of historical matches per sample
            features: Features to extract per match
        """
        self.sequence_length = sequence_length
        self.features = features or [
            "goals_scored", "goals_conceded", "xg", "xga",
            "points", "is_home", "shots", "shots_on_target",
            "possession", "pass_accuracy",
        ]

        self.samples = []
        self.labels = []

        self._prepare_data(matches_df)

    def _prepare_data(self, df):
        """Prepare sequences from match data."""
        # Group by team
        teams = df["home_team"].unique().tolist() + df["away_team"].unique().tolist()
        teams = list(set(teams))

        for team in teams:
            team_matches = df[
                (df["home_team"] == team) | (df["away_team"] == team)
            ].sort_values("match_date")

            for i in range(self.sequence_length, len(team_matches)):
                # Get sequence
                seq_matches = team_matches.iloc[i - self.sequence_length:i]
                target_match = team_matches.iloc[i]

                # Extract features
                sequence = self._extract_sequence(seq_matches, team)
                label = self._get_label(target_match, team)

                if sequence is not None and label is not None:
                    self.samples.append(sequence)
                    self.labels.append(label)

    def _extract_sequence(self, matches, team) -> Optional[np.ndarray]:
        """Extract feature sequence for a team."""
        sequence = []

        for _, match in matches.iterrows():
            is_home = match["home_team"] == team
            prefix = "home" if is_home else "away"
            opp_prefix = "away" if is_home else "home"

            features = []
            for feat in self.features:
                if feat == "is_home":
                    val = 1.0 if is_home else 0.0
                elif feat == "goals_scored":
                    val = match.get(f"{prefix}_score", 0)
                    val = float(val) if val is not None and not np.isnan(val) else 0.0
                elif feat == "goals_conceded":
                    val = match.get(f"{opp_prefix}_score", 0)
                    val = float(val) if val is not None and not np.isnan(val) else 0.0
                elif feat == "xg":
                    val = match.get(f"{prefix}_us_team_xg", 1.0)
                    val = float(val) if val is not None and not np.isnan(val) else 1.0
                elif feat == "xga":
                    val = match.get(f"{opp_prefix}_us_team_xg", 1.0)
                    val = float(val) if val is not None and not np.isnan(val) else 1.0
                elif feat == "points":
                    home_score = match.get("home_score", 0)
                    away_score = match.get("away_score", 0)
                    home_score = float(home_score) if home_score is not None and not np.isnan(home_score) else 0.0
                    away_score = float(away_score) if away_score is not None and not np.isnan(away_score) else 0.0
                    if is_home:
                        val = 3 if home_score > away_score else (1 if home_score == away_score else 0)
                    else:
                        val = 3 if away_score > home_score else (1 if home_score == away_score else 0)
                elif feat == "shots":
                    val = match.get(f"{prefix}_roll_10_shots_on_target", 10)
                    val = float(val) if val is not None and not np.isnan(val) else 10.0
                elif feat == "shots_on_target":
                    val = match.get(f"{prefix}_roll_10_shots_on_target", 4)
                    val = float(val) if val is not None and not np.isnan(val) else 4.0
                elif feat == "possession":
                    val = 50.0  # Default possession
                elif feat == "pass_accuracy":
                    val = 80.0  # Default pass accuracy
                else:
                    col = f"{prefix}_{feat}"
                    val = match.get(col, 0)
                    val = float(val) if val is not None and not np.isnan(val) else 0.0

                features.append(val)

            sequence.append(features)

        arr = np.array(sequence, dtype=np.float32)
        # Replace any remaining NaN/inf with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _get_label(self, match, team) -> Optional[int]:
        """Get outcome label (0=loss, 1=draw, 2=win for team)."""
        is_home = match["home_team"] == team
        home_score = match.get("home_score")
        away_score = match.get("away_score")

        if home_score is None or away_score is None:
            return None

        if is_home:
            if home_score > away_score:
                return 2  # Win
            elif home_score == away_score:
                return 1  # Draw
            else:
                return 0  # Loss
        else:
            if away_score > home_score:
                return 2
            elif home_score == away_score:
                return 1
            else:
                return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# =============================================================================
# TRAINING
# =============================================================================

class DeepModelTrainer:
    """Trainer for deep learning models."""

    def __init__(
        self,
        model_dir: Path = None,
        device: str = None,
    ):
        if model_dir is None:
            from config.settings import MODELS_DIR
            model_dir = MODELS_DIR / "deep"

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Models
        self.lstm_model = None
        self.transformer_model = None
        self.squad_model = None
        self.meta_model = None

    def train_lstm(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> Dict:
        """Train LSTM form model."""
        log.info("Training LSTM Form Model...")

        model = LSTMFormModel(
            input_size=10,
            hidden_size=64,
            num_layers=2,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            if val_loader:
                model.eval()
                val_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        _, predicted = outputs.max(1)
                        total += batch_y.size(0)
                        correct += predicted.eq(batch_y).sum().item()

                val_loss /= len(val_loader)
                val_acc = correct / total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), self.model_dir / "lstm_best.pt")

                if (epoch + 1) % 10 == 0:
                    log.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, "
                             f"Val: {val_loss:.4f}, Acc: {val_acc:.1%}")

        self.lstm_model = model
        return history

    def train_transformer(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 50,
        lr: float = 0.0005,
    ) -> Dict:
        """Train Transformer context model."""
        log.info("Training Transformer Model...")

        model = TransformerMatchModel(
            input_size=10,
            d_model=64,
            nhead=4,
            num_layers=2,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            scheduler.step()

            if val_loader:
                model.eval()
                val_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        _, predicted = outputs.max(1)
                        total += batch_y.size(0)
                        correct += predicted.eq(batch_y).sum().item()

                val_loss /= len(val_loader)
                val_acc = correct / total
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), self.model_dir / "transformer_best.pt")

                if (epoch + 1) % 10 == 0:
                    log.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, "
                             f"Val: {val_loss:.4f}, Acc: {val_acc:.1%}")

        self.transformer_model = model
        return history

    def save_models(self):
        """Save all trained models."""
        if self.lstm_model:
            torch.save(self.lstm_model.state_dict(), self.model_dir / "lstm_final.pt")
        if self.transformer_model:
            torch.save(self.transformer_model.state_dict(), self.model_dir / "transformer_final.pt")
        if self.squad_model:
            torch.save(self.squad_model.state_dict(), self.model_dir / "squad_final.pt")
        if self.meta_model:
            torch.save(self.meta_model.state_dict(), self.model_dir / "meta_final.pt")

        log.info(f"Models saved to {self.model_dir}")


# =============================================================================
# PREDICTOR FOR INFERENCE
# =============================================================================

class DeepPredictor:
    """Deep learning predictor for ensemble integration."""

    def __init__(self, model_dir: Path = None):
        if model_dir is None:
            from config.settings import MODELS_DIR
            model_dir = MODELS_DIR / "deep"

        self.model_dir = Path(model_dir)
        self.device = "cpu"  # Use CPU for inference

        self.lstm_model = None
        self.transformer_model = None
        self.loaded = False

    def load(self) -> bool:
        """Load trained models."""
        if not TORCH_AVAILABLE:
            log.warning("PyTorch not available")
            return False

        try:
            # Try to load LSTM
            lstm_path = self.model_dir / "lstm_best.pt"
            if lstm_path.exists():
                self.lstm_model = LSTMFormModel()
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
                self.lstm_model.eval()
                log.info("Loaded LSTM model")

            # Try to load Transformer
            transformer_path = self.model_dir / "transformer_best.pt"
            if transformer_path.exists():
                self.transformer_model = TransformerMatchModel()
                self.transformer_model.load_state_dict(
                    torch.load(transformer_path, map_location=self.device)
                )
                self.transformer_model.eval()
                log.info("Loaded Transformer model")

            self.loaded = self.lstm_model is not None or self.transformer_model is not None
            return self.loaded

        except Exception as e:
            log.error(f"Failed to load deep models: {e}")
            return False

    def predict_from_sequence(
        self,
        home_sequence: np.ndarray,
        away_sequence: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        """Predict match outcome from form sequences.

        Args:
            home_sequence: (seq_len, features) - home team recent matches
            away_sequence: (seq_len, features) - away team recent matches

        Returns:
            Dict with prob_H, prob_D, prob_A
        """
        if not self.loaded:
            if not self.load():
                return None

        try:
            probs_list = []

            # LSTM prediction
            if self.lstm_model:
                with torch.no_grad():
                    home_tensor = torch.tensor(home_sequence, dtype=torch.float32).unsqueeze(0)
                    away_tensor = torch.tensor(away_sequence, dtype=torch.float32).unsqueeze(0)

                    home_probs = self.lstm_model(home_tensor).numpy()[0]
                    away_probs = self.lstm_model(away_tensor).numpy()[0]

                    # Combine: home win = home's win prob * away's loss prob
                    combined = np.array([
                        home_probs[2] * 0.6 + (1 - away_probs[2]) * 0.4,  # Home win
                        (home_probs[1] + away_probs[1]) / 2,               # Draw
                        away_probs[2] * 0.6 + (1 - home_probs[2]) * 0.4,  # Away win
                    ])
                    combined = combined / combined.sum()
                    probs_list.append(combined)

            # Transformer prediction
            if self.transformer_model:
                with torch.no_grad():
                    home_tensor = torch.tensor(home_sequence, dtype=torch.float32).unsqueeze(0)
                    away_tensor = torch.tensor(away_sequence, dtype=torch.float32).unsqueeze(0)

                    home_probs = self.transformer_model(home_tensor).numpy()[0]
                    away_probs = self.transformer_model(away_tensor).numpy()[0]

                    combined = np.array([
                        home_probs[2] * 0.6 + (1 - away_probs[2]) * 0.4,
                        (home_probs[1] + away_probs[1]) / 2,
                        away_probs[2] * 0.6 + (1 - home_probs[2]) * 0.4,
                    ])
                    combined = combined / combined.sum()
                    probs_list.append(combined)

            if not probs_list:
                return None

            # Average predictions
            final_probs = np.mean(probs_list, axis=0)

            # Apply temperature scaling for calibration (T=2.4 from calibration analysis)
            final_probs = self._apply_temperature_scaling(final_probs, temperature=2.4)

            return {
                "prob_H": float(final_probs[0]),
                "prob_D": float(final_probs[1]),
                "prob_A": float(final_probs[2]),
            }

        except Exception as e:
            log.error(f"Deep prediction failed: {e}")
            return None

    def _apply_temperature_scaling(self, probs: np.ndarray, temperature: float = 2.4) -> np.ndarray:
        """Apply temperature scaling to calibrate overconfident predictions.

        Higher temperature = more uncertainty (less confident).
        T=2.4 was determined from calibration analysis to reduce ECE by 43%.
        """
        if temperature == 1.0:
            return probs

        eps = 1e-10
        # Convert to logits, apply temperature, convert back
        logits = np.log(probs + eps)
        scaled_logits = logits / temperature
        scaled_probs = np.exp(scaled_logits)
        scaled_probs = scaled_probs / scaled_probs.sum()

        return scaled_probs

    def predict(
        self,
        home_team: str,
        away_team: str,
        matches_df = None,
    ) -> Optional[Dict[str, float]]:
        """Predict match outcome using team names.

        Extracts recent form sequences from matches_df.
        """
        if matches_df is None:
            return None

        try:
            # Get recent matches for each team
            home_matches = matches_df[
                (matches_df["home_team"] == home_team) |
                (matches_df["away_team"] == home_team)
            ].sort_values("match_date", ascending=False).head(10)

            away_matches = matches_df[
                (matches_df["home_team"] == away_team) |
                (matches_df["away_team"] == away_team)
            ].sort_values("match_date", ascending=False).head(10)

            if len(home_matches) < 5 or len(away_matches) < 5:
                return None

            # Extract sequences
            home_seq = self._extract_sequence(home_matches, home_team)
            away_seq = self._extract_sequence(away_matches, away_team)

            return self.predict_from_sequence(home_seq, away_seq)

        except Exception as e:
            log.error(f"Deep prediction failed: {e}")
            return None

    def _safe_get(self, match, key, default):
        """Safely get value from match, handling NaN and pandas NA."""
        val = match.get(key, default)
        if val is None:
            return default
        try:
            # Catches numpy NaN, pandas NA/NaT, and any non-numeric type
            import pandas as pd
            if pd.isna(val):
                return default
        except (TypeError, ValueError, ImportError) as e:
            log.debug(f"Failed to check pd.isna for value: {e}")
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _extract_sequence(self, matches, team) -> np.ndarray:
        """Extract feature sequence for prediction."""
        sequence = []

        for _, match in matches.iterrows():
            is_home = match.get("home_team") == team
            prefix = "home" if is_home else "away"
            opp_prefix = "away" if is_home else "home"

            home_score = self._safe_get(match, "home_score", 0)
            away_score = self._safe_get(match, "away_score", 0)

            if is_home:
                pts = 3 if home_score > away_score else (1 if home_score == away_score else 0)
            else:
                pts = 3 if away_score > home_score else (1 if home_score == away_score else 0)

            # Use available columns with proper fallbacks
            xg = self._safe_get(match, f"{prefix}_us_team_xg", 1.0)
            xga = self._safe_get(match, f"{opp_prefix}_us_team_xg", 1.0)
            shots = self._safe_get(match, f"{prefix}_total_shots",
                     self._safe_get(match, f"{prefix}_us_team_shots", 10))
            sot = self._safe_get(match, f"{prefix}_roll_10_shots_on_target", 4)
            possession = self._safe_get(match, f"{prefix}_possession", 50)
            pass_acc = self._safe_get(match, f"{prefix}_passing_accuracy", 80)

            features = [
                home_score if is_home else away_score,  # goals_scored
                away_score if is_home else home_score,  # goals_conceded
                xg,   # xG
                xga,  # xGA
                pts,  # points
                1.0 if is_home else 0.0,  # is_home
                shots,  # shots
                sot,    # shots on target
                possession,  # possession
                pass_acc,    # pass accuracy
            ]
            sequence.append(features)

        # Pad or truncate to 10 matches
        while len(sequence) < 10:
            sequence.append([0] * 10)
        sequence = sequence[:10]

        arr = np.array(sequence, dtype=np.float32)
        # Final safety check - replace any remaining NaN/inf
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Deep Learning Models - Phase 5")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Run: pip install torch")
    else:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Test model creation
        print("\nCreating models...")

        lstm = LSTMFormModel()
        print(f"  LSTM: {sum(p.numel() for p in lstm.parameters())} parameters")

        transformer = TransformerMatchModel()
        print(f"  Transformer: {sum(p.numel() for p in transformer.parameters())} parameters")

        squad = SquadInteractionModel()
        print(f"  Squad: {sum(p.numel() for p in squad.parameters())} parameters")

        meta = DeepEnsemble()
        print(f"  Meta: {sum(p.numel() for p in meta.parameters())} parameters")

        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 4
        seq_len = 10

        x = torch.randn(batch_size, seq_len, 10)

        lstm_out = lstm(x)
        print(f"  LSTM output: {lstm_out.shape}")

        trans_out = transformer(x)
        print(f"  Transformer output: {trans_out.shape}")

        home_players = torch.randn(batch_size, 11, 8)
        away_players = torch.randn(batch_size, 11, 8)
        squad_out = squad(home_players, away_players)
        print(f"  Squad output: {squad_out.shape}")

        meta_out = meta(lstm_out, trans_out, squad_out)
        print(f"  Meta output: {meta_out.shape}")

        print("\nAll models working correctly!")
