"""A small training loop with validation-based early stopping, shared by the R3D-18 models.

(The LSTM keeps its own loop in :mod:`visionact.lstm`, unchanged, so that the
baseline remains exactly reproducible.)
"""

from __future__ import annotations

import copy
import time
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .utils import get_device

MIN_IMPROVEMENT = 1e-4  # validation-loss decrease that counts as progress


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Weights inversely proportional to class frequency (1.0 for every class when balanced)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    return torch.tensor(len(labels) / (num_classes * np.maximum(counts, 1.0)), dtype=torch.float32)


def fit_with_early_stopping(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_labels: np.ndarray,
    validation_logits: Callable[[nn.Module], torch.Tensor],
    val_labels: np.ndarray,
    *,
    num_classes: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    parameters=None,
) -> dict:
    """Train ``model`` on ``train_inputs`` and keep the weights of the best validation epoch.

    ``validation_logits(model)`` must return one row of logits per validation
    *video* (the caller decides how clips are combined). Training stops when
    the validation loss has not improved for ``patience`` epochs. Only
    ``parameters`` are optimised (default: every parameter that requires
    gradients), so a frozen backbone stays frozen.
    """
    device = get_device()
    model.to(device)
    labels = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    val_targets = torch.as_tensor(val_labels, dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=balanced_class_weights(train_labels, num_classes).to(device))
    trainable = list(parameters) if parameters is not None else [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=learning_rate, weight_decay=weight_decay)
    shuffle = torch.Generator().manual_seed(seed)
    class_ids = list(range(num_classes))

    history: list[dict] = []
    best: dict | None = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_inputs), generator=shuffle)
        running_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(train_inputs[batch].to(device)), labels[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimiser.step()
            running_loss += loss.item() * len(batch)

        model.eval()
        with torch.inference_mode():
            logits = validation_logits(model)
            val_loss = loss_fn(logits, val_targets).item()
        predicted = logits.argmax(dim=1).cpu().numpy()
        entry = {
            "epoch": epoch,
            "train_loss": running_loss / len(order),
            "val_loss": val_loss,
            "val_accuracy": float((predicted == val_labels).mean()),
            "val_macro_f1": float(f1_score(val_labels, predicted, labels=class_ids, average="macro", zero_division=0)),
        }
        history.append(entry)
        if best is None or val_loss < best["val_loss"] - MIN_IMPROVEMENT:
            best = {**entry, "state_dict": copy.deepcopy(model.state_dict())}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
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
