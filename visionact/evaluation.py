"""Evaluate every trained model on the held-out test split and export the results.

This is the only module that scores models on the test split. Run it after
all models have been selected on the validation split. Everything written to
``results/`` is computed from the models' actual predictions.

Outputs::

    results/comparison.json           all numbers shown in the app and the reports
    results/comparison.md             the same comparison as a Markdown table
    results/<model>/metrics.json      accuracy, macro-F1, per-class scores, confusion matrix
    results/<model>/predictions.csv   one row per test video
    results/<model>/confusion_matrix.png
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import baseline, lstm, r3d
from .config import CNN_WEIGHTS, Config
from .dataset import VideoRecord, load_manifest, load_split_summary, records_for_split
from .features import load_features
from .inference import get_backbone, load_predictor
from .plots import plot_confusion_matrix
from .registry import REFERENCE_MODEL, best_validated_model, checkpoint_path, get_spec, trained_models, validation_score
from .utils import (
    ModelNotTrainedError,
    configure_torch_threads,
    file_size_mb,
    get_logger,
    read_json,
    read_json_if_exists,
    reported_interval,
    write_json,
)

log = get_logger(__name__)


# --- metrics -----------------------------------------------------------------------------------


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, treating every video as independent."""
    if total == 0:
        return (0.0, 1.0)
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def group_bootstrap_interval(
    correct: np.ndarray, group_keys: list, seed: int, repetitions: int = 2000
) -> tuple[float, float]:
    """95% interval for accuracy that resamples source videos, not clips.

    Clips cut from one source video tend to succeed or fail together, so
    treating each clip as an independent trial (as the Wilson interval does)
    understates the uncertainty. Resampling whole groups respects that
    dependence and is the interval to trust when errors are clustered.
    """
    correct = np.asarray(correct, dtype=float)
    by_group = defaultdict(list)
    for index, key in enumerate(group_keys):
        by_group[key].append(index)
    clusters = [np.array(indices) for indices in by_group.values()]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        chosen = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        estimates.append(correct[chosen].mean())
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def source_video_breakdown(correct: np.ndarray, group_keys: list) -> dict:
    """How the results are distributed over source videos (are the errors clustered?)."""
    by_group = defaultdict(list)
    for flag, key in zip(correct, group_keys):
        by_group[key].append(bool(flag))
    rates = np.array([np.mean(flags) for flags in by_group.values()])
    wrong_clips = sum(len(flags) - sum(flags) for flags in by_group.values())
    mostly_wrong = [flags for flags in by_group.values() if np.mean(flags) < 0.5]
    return {
        "total": len(rates),
        "all_clips_correct": int((rates == 1).sum()),
        "all_clips_wrong": int((rates == 0).sum()),
        "mixed": int(((rates > 0) & (rates < 1)).sum()),
        "share_of_errors_in_mostly_wrong_source_videos": (
            sum(len(flags) - sum(flags) for flags in mostly_wrong) / wrong_clips if wrong_clips else 0.0
        ),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy, macro scores, per-class precision/recall/F1 and the confusion matrix.

    ``labels`` is passed explicitly so a class with no test videos or no
    predictions still gets a row (scored 0 rather than raising).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics without any test videos.")
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    correct = int((y_true == y_pred).sum())
    return {
        "num_videos": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci95": list(wilson_interval(correct, len(y_true))),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def paired_comparison(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Compare two models on the same videos with an exact McNemar test.

    Only videos where exactly one model is right carry information about which
    model is better. A large p-value means the observed difference is
    compatible with chance. The test treats videos as independent, so with
    clustered errors it overstates the evidence.
    """
    a_right, b_right = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    discordant = only_a + only_b
    p_value = float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "both_correct": int((a_right & b_right).sum()),
        "only_first_correct": only_a,
        "only_second_correct": only_b,
        "both_wrong": int((~a_right & ~b_right).sum()),
        "mcnemar_exact_p": p_value,
    }


def write_predictions(
    path: Path, records: list[VideoRecord], y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, class_names: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "group", "true_label", "predicted_label", "correct", *(f"score_{c}" for c in class_names)])
        for record, truth, predicted, row in zip(records, y_true, y_pred, scores):
            writer.writerow(
                [record.video_id, record.group, class_names[truth], class_names[predicted], int(truth == predicted)]
                + [f"{value:.6f}" for value in row]
            )


def temporal_order_ablation(model: lstm.LSTMClassifier, x_test: np.ndarray, y_test: np.ndarray, seed: int) -> dict:
    """Re-test a trained LSTM with the frame order destroyed.

    If accuracy barely changes, the LSTM is classifying from appearance and is
    not relying on the order of the frames.
    """
    rng = np.random.default_rng(seed)
    shuffled = np.stack([video[rng.permutation(len(video))] for video in x_test])
    reversed_order = np.ascontiguousarray(x_test[:, ::-1])

    def accuracy(features: np.ndarray) -> float:
        return float((lstm.predict_logits(model, features).argmax(axis=1) == y_test).mean())

    return {
        "original_order_accuracy": accuracy(x_test),
        "reversed_order_accuracy": accuracy(reversed_order),
        "shuffled_order_accuracy": accuracy(shuffled),
    }


# --- scoring the trained models -------------------------------------------------------------------


def _check_feature_key(name: str, trained_on: str, current: str) -> None:
    if trained_on != current:
        raise ValueError(
            f"{name} was trained on features {trained_on!r} but the config now produces {current!r}. "
            "Retrain the model or restore the previous settings."
        )


def cached_scores(config: Config, key: str, records: list[VideoRecord], cache: dict) -> tuple[np.ndarray, dict]:
    """Class scores (N, classes) for ``records`` from cached features, plus model-specific extras.

    ``cache`` holds the loaded features of these records, so models that share a backbone load them once.
    """
    spec = get_spec(key)
    extras: dict = {}
    if spec.backbone == "mobilenet_v3_small":
        if "mobilenet" not in cache:
            cache["mobilenet"] = load_features(config, records)[0]
        features = cache["mobilenet"]
        if spec.family == "svm":
            bundle = baseline.load_baseline(config, key)
            _check_feature_key(spec.display_name, bundle["feature_key"], config.feature_key)
            scores = baseline.class_scores(bundle["pipeline"], baseline.temporal_mean_pool(features), len(config.classes))
        else:
            model, checkpoint = lstm.load_lstm(config, key)
            _check_feature_key(spec.display_name, checkpoint["feature_key"], config.feature_key)
            scores = lstm.predict_probabilities(model, features)
            extras["lstm_model"] = model
    else:
        if "r3d" not in cache:
            cache["r3d"] = r3d.load_r3d_features(config, records)[0]
        head, checkpoint = r3d.load_r3d_head(config)
        _check_feature_key(spec.display_name, checkpoint["feature_key"], config.r3d_feature_key)
        scores = r3d.video_probabilities(head, cache["r3d"])
    return scores, extras


def benchmark_inference(config: Config, keys: list[str], records: list[VideoRecord], num_videos: int, backbones: dict) -> dict:
    """Time the full CPU pipeline of each model on a spread of test videos (mean milliseconds per video)."""
    step = max(len(records) // max(num_videos, 1), 1)
    chosen = records[::step][:num_videos]
    predictors = {key: load_predictor(config, key, backbones) for key in keys}
    for predictor in predictors.values():
        predictor.predict(chosen[0].path)  # warm-up: the first call pays one-off costs
    # Models take turns on each video, so a slow spell of the machine affects all of them alike;
    # the median then ignores the occasional outlier.
    stages = {key: defaultdict(list) for key in keys}
    for record in chosen:
        for key, predictor in predictors.items():
            for stage, seconds in predictor.predict(record.path).timings.items():
                stages[key][stage].append(seconds)
    per_model = {}
    for key, predictor in predictors.items():
        median_ms = {f"{stage}_ms": 1000 * float(np.median(values)) for stage, values in stages[key].items()}
        per_model[key] = {**median_ms, "total_ms": sum(median_ms.values()), "input": predictor.description}
    return {"videos_timed": len(chosen), "statistic": "median per stage", "per_model": per_model}


def _feature_extractor_info(backbones: dict, name: str) -> dict:
    extractor = get_backbone(backbones, name)
    pretrained = {
        "mobilenet_v3_small": f"ImageNet ({CNN_WEIGHTS})",
        "r3d_18": "Kinetics-400 (KINETICS400_V1)",
    }[name]
    return {
        "name": name,
        "pretrained_on": pretrained,
        "frozen": True,
        "feature_dim": extractor.feature_dim,
        "parameters": extractor.num_parameters(),
        "size_mb": extractor.size_mb(),
    }


def evaluate_models(config: Config, timing_videos: int = 20) -> dict:
    """Score every trained model on the same test videos and write all result files."""
    configure_torch_threads(config.torch_threads)
    class_names = list(config.classes)
    records = records_for_split(load_manifest(config), "test")
    keys = trained_models(config)
    if not keys:
        raise ModelNotTrainedError("No trained model was found. Train at least one model before evaluating.")
    y_test = np.array([class_names.index(record.label) for record in records], dtype=np.int64)
    group_keys = [record.group_key for record in records]

    cache: dict = {}
    predictions: dict[str, np.ndarray] = {}
    ablations: dict[str, dict] = {}
    models: dict[str, dict] = {}
    for key in keys:
        spec = get_spec(key)
        scores, extras = cached_scores(config, key, records, cache)
        y_pred = scores.argmax(axis=1)
        predictions[key] = y_pred
        metrics = compute_metrics(y_test, y_pred, class_names)
        metrics["accuracy_ci95_group_bootstrap"] = list(group_bootstrap_interval(y_pred == y_test, group_keys, config.seed))
        metrics["source_videos"] = source_video_breakdown(y_pred == y_test, group_keys)
        if "lstm_model" in extras:
            ablations[key] = temporal_order_ablation(extras["lstm_model"], cache["mobilenet"], y_test, config.seed)

        output_dir = config.results_dir / key
        training = read_json(output_dir / "training.json")
        write_json(output_dir / "metrics.json", {"model": spec.display_name, "split": "test", **metrics})
        write_predictions(output_dir / "predictions.csv", records, y_test, y_pred, scores, class_names)
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, f"{spec.display_name} (test set)", output_dir / "confusion_matrix.png")
        selected = validation_score(config, key) or {}
        models[key] = {
            "display_name": spec.display_name,
            "backbone": spec.backbone,
            "augmented_training": bool(training["experiment"]["augmentation"]),
            **metrics,
            "validation": {name: selected.get(name) for name in ("val_accuracy", "val_macro_f1")},
            "selected_hyperparameters": training["selected"],
            "training_seconds": training["training_seconds"],
            "search_seconds": training["search_seconds"],
            "trainable_parameters": training.get("trainable_parameters"),
            "model_size_mb": file_size_mb(checkpoint_path(config, key)),
        }
        log.info(
            "%-44s accuracy %.3f (source-video 95%% CI %.3f-%.3f), macro-F1 %.3f",
            spec.display_name, metrics["accuracy"], *metrics["accuracy_ci95_group_bootstrap"], metrics["macro_f1"],
        )  # fmt: skip

    backbones: dict = {}
    extractors = {name: _feature_extractor_info(backbones, name) for name in dict.fromkeys(m["backbone"] for m in models.values())}
    timing = None
    if timing_videos > 0:
        log.info("Timing the CPU pipeline of %d models on %d test videos...", len(keys), min(timing_videos, len(records)))
        timing = benchmark_inference(config, keys, records, timing_videos, backbones)
    for key, model in models.items():
        model["total_size_mb"] = model["model_size_mb"] + extractors[model["backbone"]]["size_mb"]
        model["inference_ms_per_video"] = timing["per_model"][key]["total_ms"] if timing else None

    reference = REFERENCE_MODEL if REFERENCE_MODEL in keys else keys[0]
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "class_names": class_names,
        "test_videos": int(len(y_test)),
        "test_source_videos": len(set(group_keys)),
        "split": load_split_summary(config),
        "feature_extractors": extractors,
        "models": models,
        "reference_model": reference,
        "best_validated_model": best_validated_model(config),
        "paired_comparisons": {
            key: paired_comparison(y_test, predictions[reference], predictions[key]) for key in keys if key != reference
        },
        "temporal_order_ablation": ablations,
        "timing": timing,
        "config": config.to_dict(),
    }
    write_json(config.results_dir / "comparison.json", results)
    (config.results_dir / "comparison.md").write_text(comparison_markdown(results), encoding="utf-8")
    log.info("Results written to %s", config.results_dir)
    return results


def comparison_markdown(results: dict) -> str:
    """Render the model comparison as a Markdown table (also used in the reports)."""
    lines = [
        f"Test set: {results['test_videos']} videos, classes: {', '.join(results['class_names'])}.",
        "",
        "| Model | Test accuracy | 95% interval | Macro F1 | Validation macro F1 "
        "| Training (s) | Inference (ms/video) | Size with backbone (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    fallback_used = False
    for key, model in results["models"].items():
        low, high, basis = reported_interval(model)
        mark = "" if basis == "source videos" else " *"
        fallback_used |= bool(mark)
        inference = f"{model['inference_ms_per_video']:.0f}" if model["inference_ms_per_video"] is not None else "not measured"
        validation = model["validation"]["val_macro_f1"]
        name = model["display_name"] + (" (best on validation)" if key == results.get("best_validated_model") else "")
        lines.append(
            f"| {name} | {model['accuracy']:.1%} | {low:.1%} to {high:.1%}{mark} | {model['macro_f1']:.3f} "
            f"| {validation:.3f} | {model['training_seconds']:.2f} | {inference} | {model['total_size_mb']:.1f} |"
        )
    lines += ["", "Intervals resample whole source videos, since clips of one source video succeed or fail together."]
    if fallback_used:
        lines.append(
            "\\* No errors, so resampling cannot show uncertainty; this interval treats the videos as independent "
            "and is therefore too narrow."
        )
    return "\n".join(lines) + "\n"


def load_results(config: Config) -> dict | None:
    """The saved comparison, or ``None`` if ``evaluate`` has not been run yet."""
    return read_json_if_exists(config.results_dir / "comparison.json")
