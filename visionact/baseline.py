"""Baseline: CNN features -> temporal mean pooling -> linear SVM.

Averaging the frame features over time discards their order, so this model
sees *what* is in the video but not *in which order* things happen. Comparing it
with the LSTM, which reads the same features in order, isolates the value of
temporal modelling.
"""

from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import Config
from .dataset import experiment_record, load_manifest, records_for_split
from .features import augmentation_settings, load_augmented_features, load_features, preprocessing_settings
from .registry import checkpoint_path, get_spec
from .utils import ModelNotTrainedError, get_logger, write_json

log = get_logger(__name__)

MODEL_KEY = "cnn_svm"
AUGMENTED_KEY = "cnn_svm_aug"
DISPLAY_NAME = get_spec(MODEL_KEY).display_name


def temporal_mean_pool(features: np.ndarray) -> np.ndarray:
    """(N, T, D) -> (N, D): average over the time axis. The result ignores frame order."""
    if features.ndim != 3:
        raise ValueError(f"Expected features of shape (N, T, D), got {features.shape}.")
    return features.mean(axis=1)


def build_svm(c: float, seed: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=20000),
    )


def class_scores(model: Pipeline, pooled: np.ndarray, num_classes: int) -> np.ndarray:
    """SVM decision values arranged as (N, num_classes); higher means more confident."""
    decision = model.decision_function(pooled)
    if decision.ndim == 1:  # binary problems return one column
        decision = np.column_stack([-decision, decision])
    scores = np.full((len(pooled), num_classes), -np.inf, dtype=np.float64)
    scores[:, model.classes_] = decision
    return scores


def fit_and_select_svm(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, config: Config
) -> tuple[Pipeline, dict, list[dict]]:
    """Fit one SVM per C on the training data and keep the one with the best validation macro-F1."""
    labels = list(range(len(config.classes)))
    candidates = []
    best = None
    for c in config.svm_c_grid:
        started = time.perf_counter()
        model = build_svm(c, config.seed).fit(train_x, train_y)
        seconds = time.perf_counter() - started
        predicted = model.predict(val_x)
        result = {
            "C": c,
            "val_accuracy": float(accuracy_score(val_y, predicted)),
            "val_macro_f1": float(f1_score(val_y, predicted, labels=labels, average="macro", zero_division=0)),
            "training_seconds": round(seconds, 4),
        }
        candidates.append(result)
        log.info("C=%-7g val accuracy %.3f, val macro-F1 %.3f", c, result["val_accuracy"], result["val_macro_f1"])
        # The grid is ascending, so on ties the smaller C (stronger regularisation) is kept.
        if best is None or result["val_macro_f1"] > best[0]["val_macro_f1"]:
            best = (result, model)
    return best[1], best[0], candidates


def train_baseline(config: Config, augment: bool = False) -> dict:
    """Fit on the training split, choose C on the validation split, save the selected model.

    With ``augment=True`` the augmented copies of the training videos are added
    to the training set and the model is saved as a separate variant, so the
    original baseline is never overwritten.
    """
    key = AUGMENTED_KEY if augment else MODEL_KEY
    records = load_manifest(config)
    train_records = records_for_split(records, "train")
    x_train, y_train = load_features(config, train_records)
    x_val, y_val = load_features(config, records_for_split(records, "val"))
    pooled_train, labels_train = temporal_mean_pool(x_train), y_train
    if augment:
        x_augmented, _ = load_augmented_features(config, train_records)  # (N, copies, T, D)
        copies = x_augmented.shape[1]
        pooled_augmented = x_augmented.mean(axis=2).reshape(-1, x_augmented.shape[-1])
        pooled_train = np.concatenate([pooled_train, pooled_augmented])
        labels_train = np.concatenate([y_train, np.repeat(y_train, copies)])

    model, selected, candidates = fit_and_select_svm(pooled_train, labels_train, temporal_mean_pool(x_val), y_val, config)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "classes": list(config.classes),
            "feature_key": config.feature_key,
            "num_frames": config.num_frames,
            "C": selected["C"],
        },
        checkpoint_path(config, key),
    )
    summary = {
        "model": get_spec(key).display_name,
        "pooling": "mean over time",
        "selected": selected,
        "candidates": candidates,
        "selection_metric": "validation macro-F1",
        "training_seconds": selected["training_seconds"],
        "search_seconds": round(sum(c["training_seconds"] for c in candidates), 4),
        "train_videos": int(len(y_train)),
        "training_samples": int(len(labels_train)),
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
    log.info("Selected C=%g. Model saved to %s", selected["C"], checkpoint_path(config, key))
    return summary


def load_baseline(config: Config, key: str = MODEL_KEY) -> dict:
    """Load a saved SVM. Only load model files you created yourself (joblib uses pickle)."""
    path = checkpoint_path(config, key)
    if not path.is_file():
        flag = " --augment" if get_spec(key).augmented else ""
        raise ModelNotTrainedError(
            f"No trained model at {path}. Run `python -m visionact train-baseline{flag}` first."
        )
    bundle = joblib.load(path)
    if bundle["classes"] != list(config.classes):
        raise ValueError(
            f"The saved SVM was trained for classes {bundle['classes']}, but the config lists "
            f"{list(config.classes)}. Retrain with `python -m visionact train-baseline`."
        )
    return bundle
