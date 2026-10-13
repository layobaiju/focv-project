"""Matplotlib figures for results: confusion matrices, training curves, class scores.

Functions return the figure and, when ``path`` is given, also save it. The
non-interactive Agg backend is used so they work without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def _finish(figure: Figure, path: str | Path | None) -> Figure:
    figure.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
        plt.close(figure)
    return figure


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]], class_names: Sequence[str], title: str = "", path: str | Path | None = None
) -> Figure:
    """Rows are true classes, columns are predicted classes; cells show video counts."""
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(1.0 * len(class_names) + 2.0, 0.9 * len(class_names) + 1.4))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="videos")
    ticks = np.arange(len(class_names))
    axis.set_xticks(ticks, class_names, rotation=30, ha="right")
    axis.set_yticks(ticks, class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    if title:
        axis.set_title(title)
    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0.5
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            colour = "white" if matrix[row, column] > threshold else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=colour)
    return _finish(figure, path)


def plot_training_curves(history: Sequence[dict], best_epoch: int, path: str | Path | None = None) -> Figure:
    """Loss and validation accuracy per epoch, with the early-stopping checkpoint marked."""
    epochs = [entry["epoch"] for entry in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 3.8))
    loss_axis.plot(epochs, [entry["train_loss"] for entry in history], label="training loss")
    loss_axis.plot(epochs, [entry["val_loss"] for entry in history], label="validation loss")
    loss_axis.set_ylabel("cross-entropy loss")
    accuracy_axis.plot(epochs, [entry["val_accuracy"] for entry in history], color="tab:green", label="validation accuracy")
    accuracy_axis.set_ylabel("accuracy")
    accuracy_axis.set_ylim(0, 1.02)
    for axis in (loss_axis, accuracy_axis):
        axis.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"saved checkpoint (epoch {best_epoch})")
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    return _finish(figure, path)


def plot_class_scores(scores: dict[str, float], title: str = "", path: str | Path | None = None) -> Figure:
    """Horizontal bar chart of per-class scores in [0, 1], highest at the top."""
    names = sorted(scores, key=scores.get)
    values = [scores[name] for name in names]
    figure, axis = plt.subplots(figsize=(6, 0.5 * len(names) + 1.2))
    bars = axis.barh(names, values, color=["tab:blue" if v == max(values) else "lightgrey" for v in values])
    axis.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3)
    axis.set_xlim(0, 1.15)
    axis.set_xlabel("softmax score")
    if title:
        axis.set_title(title)
    return _finish(figure, path)


def plot_metric_comparison(
    models: dict[str, dict], metrics: Sequence[tuple[str, str]], path: str | Path | None = None
) -> Figure:
    """Grouped bars comparing models. ``metrics`` is a list of (key in the model dict, axis label)."""
    many = len(models) > 3
    figure, axis = plt.subplots(figsize=(10, 4.4) if many else (9, 3.6))
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(metrics))
    for index, (name, values) in enumerate(models.items()):
        heights = [values[key] for key, _ in metrics]
        bars = axis.bar(positions + index * width, heights, width, label=name)
        # With many models the value labels are turned upright so neighbours do not overlap.
        axis.bar_label(bars, labels=[f"{h:.2f}" if many else f"{h:.3f}" for h in heights], padding=2,
                       fontsize=7 if many else 8, rotation=90 if many else 0)  # fmt: skip
    axis.set_xticks(positions + width * (len(models) - 1) / 2, [label for _, label in metrics])
    axis.set_ylim(0, 1.18 if many else 1.12)
    axis.set_ylabel("score on the test set")
    axis.grid(axis="y", alpha=0.3)
    # Above the axes, so the legend never hides a bar.
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2 if many else max(len(models), 1), frameon=False)
    return _finish(figure, path)
