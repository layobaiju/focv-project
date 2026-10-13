"""CNN + LSTM: an LSTM reads the ordered frame features and predicts the activity.

Only the LSTM and its linear head are trained; the CNN features come from the
cache. Training is small enough for a laptop CPU (seconds per run).
"""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .config import Config
from .dataset import experiment_record, load_manifest, records_for_split
from .features import augmentation_settings, load_augmented_features, load_features, preprocessing_settings
from .registry import checkpoint_path, get_spec
from .utils import ModelNotTrainedError, configure_torch_threads, get_device, get_logger, set_seed, write_json

log = get_logger(__name__)

MODEL_KEY = "cnn_lstm"
AUGMENTED_KEY = "cnn_lstm_aug"
DISPLAY_NAME = get_spec(MODEL_KEY).display_name
_MIN_IMPROVEMENT = 1e-4  # validation-loss decrease that counts as progress


class LSTMClassifier(nn.Module):
    """Standardise features -> dropout -> LSTM -> dropout -> linear layer on the last hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        # Per-dimension statistics of the *training* features, stored with the weights.
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def set_normalisation(self, train_features: np.ndarray) -> None:
        flat = torch.as_tensor(train_features, dtype=torch.float32).reshape(-1, train_features.shape[-1])
        self.feature_mean.copy_(flat.mean(dim=0))
        self.feature_std.copy_(flat.std(dim=0).clamp_min(1e-6))

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """``sequences``: (batch, time, input_dim) with frames in temporal order. Returns class logits."""
        if sequences.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, time, features), got {tuple(sequences.shape)}.")
        normalised = (sequences - self.feature_mean) / self.feature_std
        _, (hidden, _) = self.lstm(self.input_dropout(normalised))
        return self.head(self.output_dropout(hidden[-1]))  # last layer's state after the final frame


@dataclass(frozen=True)
class LSTMHyperparameters:
    hidden_dim: int
    num_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int

    @classmethod
    def from_config(cls, config: Config, **overrides) -> "LSTMHyperparameters":
        values = {name: getattr(config, name) for name in cls.__dataclass_fields__}
        values.update(overrides)
        return cls(**values)


def hyperparameter_grid(config: Config) -> list[LSTMHyperparameters]:
    return [
        LSTMHyperparameters.from_config(config, hidden_dim=hidden, dropout=dropout, learning_rate=rate)
        for hidden, dropout, rate in itertools.product(
            config.search_hidden_dims, config.search_dropouts, config.search_learning_rates
        )
    ]


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Weights inversely proportional to class frequency (1.0 for every class when balanced)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = len(labels) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.inference_mode()
def predict_logits(model: LSTMClassifier, features: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    inputs = torch.as_tensor(features, dtype=torch.float32, device=get_device())
    return torch.cat([model(inputs[i : i + batch_size]) for i in range(0, len(inputs), batch_size)]).numpy()


def predict_probabilities(model: LSTMClassifier, features: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(predict_logits(model, features)), dim=1).numpy()


def fit_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hp: LSTMHyperparameters,
    num_classes: int,
    seed: int,
) -> dict:
    """Train one LSTM with early stopping on validation loss; the best epoch's weights are restored."""
    set_seed(seed)
    device = get_device()
    model = LSTMClassifier(x_train.shape[2], hp.hidden_dim, num_classes, hp.num_layers, hp.dropout).to(device)
    model.set_normalisation(x_train)
    train_x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=balanced_class_weights(y_train, num_classes).to(device))
    optimiser = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    shuffle = torch.Generator().manual_seed(seed)
    labels = list(range(num_classes))

    history: list[dict] = []
    best: dict | None = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, hp.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=shuffle)
        running_loss = 0.0
        for start in range(0, len(order), hp.batch_size):
            batch = order[start : start + hp.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(train_x[batch]), train_y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            running_loss += loss.item() * len(batch)

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x)
            val_loss = loss_fn(val_logits, val_y).item()
        val_predicted = val_logits.argmax(dim=1).numpy()
        entry = {
            "epoch": epoch,
            "train_loss": running_loss / len(order),
            "val_loss": val_loss,
            "val_accuracy": float((val_predicted == y_val).mean()),
            "val_macro_f1": float(f1_score(y_val, val_predicted, labels=labels, average="macro", zero_division=0)),
        }
        history.append(entry)

        if best is None or val_loss < best["val_loss"] - _MIN_IMPROVEMENT:
            best = {**entry, "state_dict": copy.deepcopy(model.state_dict())}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= hp.patience:
                break

    seconds = time.perf_counter() - started
    model.load_state_dict(best["state_dict"])
    model.eval()
    return {
        "model": model,
        "history": history,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "val_loss": best["val_loss"],
        "val_accuracy": best["val_accuracy"],
        "val_macro_f1": best["val_macro_f1"],
        "training_seconds": seconds,
    }


def train_lstm(config: Config, search: bool = False, augment: bool = False) -> dict:
    """Train on the training split, select on the validation split, save the best checkpoint.

    With ``search=True`` every combination in the config's search grid is
    trained and the one with the highest validation macro-F1 is kept (ties:
    lower validation loss). With ``augment=True`` the augmented copies of the
    training videos are added to the training set and the result is saved as a
    separate variant. The test split is never read here.
    """
    key = AUGMENTED_KEY if augment else MODEL_KEY
    configure_torch_threads(config.torch_threads)
    records = load_manifest(config)
    train_records = records_for_split(records, "train")
    x_train, y_train = load_features(config, train_records)
    x_val, y_val = load_features(config, records_for_split(records, "val"))
    train_videos = len(y_train)
    if augment:
        x_augmented, _ = load_augmented_features(config, train_records)  # (N, copies, T, D)
        copies = x_augmented.shape[1]
        x_train = np.concatenate([x_train, x_augmented.reshape(-1, *x_augmented.shape[2:])])
        y_train = np.concatenate([y_train, np.repeat(y_train, copies)])
    num_classes = len(config.classes)

    candidates = hyperparameter_grid(config) if search else [LSTMHyperparameters.from_config(config)]
    runs: list[dict] = []
    best: tuple[LSTMHyperparameters, dict] | None = None
    for hp in candidates:
        result = fit_lstm(x_train, y_train, x_val, y_val, hp, num_classes, config.seed)
        runs.append(
            {
                **asdict(hp),
                "best_epoch": result["best_epoch"],
                "epochs_run": result["epochs_run"],
                "val_loss": result["val_loss"],
                "val_accuracy": result["val_accuracy"],
                "val_macro_f1": result["val_macro_f1"],
                "training_seconds": round(result["training_seconds"], 3),
            }
        )
        log.info(
            "hidden=%-4d dropout=%.1f lr=%-7g -> best epoch %2d/%2d, val loss %.4f, accuracy %.3f, macro-F1 %.3f (%.1f s)",
            hp.hidden_dim, hp.dropout, hp.learning_rate, result["best_epoch"], result["epochs_run"],
            result["val_loss"], result["val_accuracy"], result["val_macro_f1"], result["training_seconds"],
        )  # fmt: skip
        rank = (result["val_macro_f1"], -result["val_loss"])
        if best is None or rank > (best[1]["val_macro_f1"], -best[1]["val_loss"]):
            best = (hp, result)

    hp, result = best
    model: LSTMClassifier = result["model"]
    architecture = {
        "input_dim": int(x_train.shape[2]),
        "hidden_dim": hp.hidden_dim,
        "num_classes": num_classes,
        "num_layers": hp.num_layers,
        "dropout": hp.dropout,
    }
    config.models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "classes": list(config.classes),
            "feature_key": config.feature_key,
            "num_frames": config.num_frames,
            "seed": config.seed,
            "best_epoch": result["best_epoch"],
            "val_loss": result["val_loss"],
            "val_accuracy": result["val_accuracy"],
            "val_macro_f1": result["val_macro_f1"],
        },
        checkpoint_path(config, key),
    )
    summary = {
        "model": get_spec(key).display_name,
        "selected": {**asdict(hp), **{k: result[k] for k in ("best_epoch", "epochs_run", "val_loss", "val_accuracy", "val_macro_f1")}},
        "architecture": architecture,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selection_metric": "validation macro-F1, ties broken by lower validation loss",
        "early_stopping": f"validation loss, patience {hp.patience} epochs",
        "searched": search,
        "candidates": runs,
        "training_seconds": round(result["training_seconds"], 3),
        "search_seconds": round(sum(run["training_seconds"] for run in runs), 3),
        "history": result["history"],
        "train_videos": int(train_videos),
        "training_samples": int(len(y_train)),
        "val_videos": int(len(y_val)),
        "seed": config.seed,
        "experiment": experiment_record(
            config,
            features=config.feature_key,
            preprocessing=preprocessing_settings(config),
            augmentation=augmentation_settings(config) if augment else None,
        ),
    }
    write_json(config.results_dir / key / "training.json", summary)

    from .plots import plot_training_curves

    plot_training_curves(result["history"], result["best_epoch"], config.results_dir / key / "training_curves.png")
    log.info("Best checkpoint (epoch %d) saved to %s", result["best_epoch"], checkpoint_path(config, key))
    return summary


def load_lstm(config: Config, key: str = MODEL_KEY) -> tuple[LSTMClassifier, dict]:
    """Load a saved checkpoint. Returns the model in eval mode and the checkpoint metadata."""
    path = checkpoint_path(config, key)
    if not path.is_file():
        flag = " --augment" if get_spec(key).augmented else ""
        raise ModelNotTrainedError(
            f"No trained CNN + LSTM checkpoint at {path}. Run `python -m visionact train-lstm{flag}` first."
        )
    checkpoint = torch.load(path, map_location=get_device(), weights_only=True)
    if checkpoint["classes"] != list(config.classes):
        raise ValueError(
            f"The checkpoint was trained for classes {checkpoint['classes']}, but the config lists "
            f"{list(config.classes)}. Retrain with `python -m visionact train-lstm`."
        )
    model = LSTMClassifier(**checkpoint["architecture"]).to(get_device())
    model.load_state_dict(checkpoint.pop("state_dict"))
    model.eval()
    return model, checkpoint
