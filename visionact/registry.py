"""The models the project can train, in order of increasing complexity.

Evaluation, the command line and the app all iterate over this registry, so
every trained model is scored on the same test videos in the same way. The
module deliberately imports neither PyTorch nor scikit-learn: it only knows
names and file locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .utils import read_json_if_exists


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str  # "svm", "lstm" or "r3d_head": decides how the model is loaded and applied
    backbone: str  # "mobilenet_v3_small" or "r3d_18"
    filename: str  # checkpoint file inside <artifacts>/models
    augmented: bool  # trained with augmented copies of the training videos
    summary: str


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "cnn_svm", "CNN + mean pooling + linear SVM", "svm", "mobilenet_v3_small", "cnn_svm_baseline.joblib",
            False, "Baseline. MobileNetV3-Small frame features averaged over time, then a linear SVM.",
        ),
        ModelSpec(
            "cnn_lstm", "CNN + LSTM", "lstm", "mobilenet_v3_small", "cnn_lstm_best.pt",
            False, "MobileNetV3-Small frame features read in order by an LSTM.",
        ),
        ModelSpec(
            "cnn_svm_aug", "CNN + SVM, augmented training", "svm", "mobilenet_v3_small", "cnn_svm_aug.joblib",
            True, "The baseline, trained on the original and augmented training videos.",
        ),
        ModelSpec(
            "cnn_lstm_aug", "CNN + LSTM, augmented training", "lstm", "mobilenet_v3_small", "cnn_lstm_aug.pt",
            True, "The CNN + LSTM, trained on the original and augmented training videos.",
        ),
        ModelSpec(
            "r3d18_head", "R3D-18 (Kinetics-400, frozen) + linear head", "r3d_head", "r3d_18", "r3d18_head_best.pt",
            False, "Frozen 3D CNN pretrained on action videos, with a new classification head.",
        ),
    )
}  # fmt: skip

REFERENCE_MODEL = "cnn_svm"  # every other model is compared against this one


def get_spec(key: str) -> ModelSpec:
    if key not in MODELS:
        raise ValueError(f"Unknown model {key!r}. Known models: {', '.join(MODELS)}.")
    return MODELS[key]


def checkpoint_path(config: Config, key: str) -> Path:
    return config.models_dir / get_spec(key).filename


def trained_models(config: Config) -> list[str]:
    """Keys of the models whose checkpoint exists, in registry order."""
    return [key for key in MODELS if checkpoint_path(config, key).is_file()]


def validation_score(config: Config, key: str) -> dict | None:
    """The selected candidate's validation results, as recorded when the model was trained."""
    training = read_json_if_exists(config.results_dir / key / "training.json")
    return training.get("selected") if training else None


def best_validated_model(config: Config) -> str | None:
    """The trained model with the best validation macro-F1 (ties: accuracy, then registry order).

    This is the model used by default for predictions. Test-set results are
    never consulted, so the choice cannot be tuned to the test set.
    """
    best_key, best_rank = None, None
    for key in trained_models(config):
        score = validation_score(config, key)
        if not score or "val_macro_f1" not in score:
            continue
        rank = (score["val_macro_f1"], score.get("val_accuracy", 0.0))
        if best_rank is None or rank > best_rank:
            best_key, best_rank = key, rank
    return best_key
