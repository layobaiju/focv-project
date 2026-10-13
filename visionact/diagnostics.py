"""Diagnostics: why is the accuracy what it is?

``python -m visionact diagnose`` gathers evidence about the data and the two
baseline models and writes it to ``results/diagnostics.json``. It answers:

* Is the data sound? (labels, paths, decodability, duplicates, class balance)
* Is the test split independent of training at the source-video level?
* Do the saved models reproduce the reported predictions?
* Are the models overfitting? (training against validation against test accuracy)
* How much can a validation score from 4 groups per class be trusted?
* Would more training data help? (learning curve over training groups)

Every experiment that fits a model here uses the training and validation
groups only. Test videos are used solely to re-score the already selected
models, which changes nothing about how they were chosen.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import baseline
from .config import Config
from .dataset import SPLITS, VideoRecord, assert_no_group_leakage, load_manifest, parse_ucf101_name, validate_records
from .evaluation import cached_scores, group_bootstrap_interval, source_video_breakdown
from .features import FeatureExtractor, load_features
from .registry import MODELS, trained_models
from .utils import get_logger, read_json_if_exists, write_json
from .video import read_metadata

log = get_logger(__name__)


# --- data audit ----------------------------------------------------------------------


def audit_dataset(config: Config, records: list[VideoRecord]) -> dict:
    """Counts, group structure, clip lengths and frame rates per class."""
    per_class = {}
    for label in config.classes:
        items = [r for r in records if r.label == label]
        groups = Counter(r.group for r in items)
        frames = [r.num_frames for r in items]
        rates = Counter(round(read_metadata(r.path).fps, 2) for r in items)
        per_class[label] = {
            "videos": len(items),
            "groups": len(groups),
            "clips_per_group": {"min": min(groups.values()), "max": max(groups.values())},
            "frames": {"min": min(frames), "median": float(np.median(frames)), "max": max(frames)},
            "frame_rates": {str(rate): count for rate, count in sorted(rates.items())},
            "split_videos": {s: sum(r.split == s for r in items) for s in SPLITS},
            "split_groups": {s: len({r.group for r in items if r.split == s}) for s in SPLITS},
        }
    train_counts = [per_class[label]["split_videos"]["train"] for label in config.classes]
    return {
        "per_class": per_class,
        "train_imbalance_ratio": max(train_counts) / min(train_counts),
        "chance_accuracy": 1 / len(config.classes),
    }


def check_labels_and_files(records: list[VideoRecord]) -> dict:
    """Every file exists, decodes, and its name agrees with its class folder."""
    name_mismatch = [
        r.video_id for r in records if (parsed := parse_ucf101_name(r.path.stem)) and parsed[0] != r.label
    ]
    valid, problems = validate_records(records)
    return {
        "videos_checked": len(records),
        "decodable": len(valid),
        "undecodable": problems,
        "name_label_mismatches": name_mismatch,
    }


def check_split_independence(records: list[VideoRecord]) -> dict:
    """A (class, group) pair is one source video; it must appear in exactly one split."""
    assert_no_group_leakage(records)  # raises on leakage
    splits_of = defaultdict(set)
    for record in records:
        splits_of[record.group_key].add(record.split)
    return {
        "source_videos": len(splits_of),
        "source_videos_in_more_than_one_split": sum(len(s) > 1 for s in splits_of.values()),
        "source_videos_per_split": {s: sum(s in v for v in splits_of.values()) for s in SPLITS},
    }


def find_exact_duplicates(records: list[VideoRecord]) -> list[list[str]]:
    """Groups of videos whose files are byte-identical."""
    by_hash = defaultdict(list)
    for record in records:
        by_hash[hashlib.sha256(record.path.read_bytes()).hexdigest()].append(record.video_id)
    return [ids for ids in by_hash.values() if len(ids) > 1]


def near_duplicate_report(records: list[VideoRecord], pooled: np.ndarray, top: int = 10) -> dict:
    """Look for videos from *different* source videos that are suspiciously alike.

    Clips of one source video (same class and group) are expected to be similar
    and always share a split. The risk is two different groups that are really
    the same footage and sit in different splits. Similarity is the cosine of
    the mean-pooled CNN features; a cross-group pair is flagged when it is more
    similar than the typical (median) pair of clips from one source video.
    """
    unit = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    similarity = unit @ unit.T
    keys = [r.group_key for r in records]
    same_source = np.array([[a == b for b in keys] for a in keys])
    upper = np.triu(np.ones_like(similarity, dtype=bool), k=1)
    within = similarity[same_source & upper]
    threshold = float(np.median(within))

    cross = ~same_source & upper
    cross_split = cross & np.array([[a.split != b.split for b in records] for a in records])
    flagged = np.argwhere(cross_split & (similarity > threshold))
    flagged = sorted(flagged, key=lambda ij: -similarity[ij[0], ij[1]])
    return {
        "method": "cosine similarity of mean-pooled MobileNetV3 features",
        "within_source_video_similarity": {"median": threshold, "p10": float(np.percentile(within, 10))},
        "cross_source_similarity": {"median": float(np.median(similarity[cross])), "max": float(similarity[cross].max())},
        "cross_split_pairs_above_within_source_median": len(flagged),
        "most_similar_cross_split_pairs": [
            {
                "a": records[i].video_id, "a_split": records[i].split,
                "b": records[j].video_id, "b_split": records[j].split,
                "similarity": float(similarity[i, j]),
            }
            for i, j in flagged[:top]
        ],  # fmt: skip
    }


def check_feature_cache(config: Config, records: list[VideoRecord], sample: int = 6) -> dict:
    """Re-extract a few videos and compare with the cache (same code path as inference)."""
    extractor = FeatureExtractor()
    step = max(len(records) // sample, 1)
    chosen = records[::step][:sample]
    cached, _ = load_features(config, chosen)
    worst = 0.0
    for record, expected in zip(chosen, cached):
        fresh, _ = extractor.extract_video(record.path, config.num_frames)
        worst = max(worst, float(np.abs(fresh - expected).max()))
    return {"videos_rechecked": len(chosen), "max_abs_difference": worst, "consistent": worst < 1e-3}


# --- the saved models ------------------------------------------------------------------


def verify_saved_models(config: Config, records: list[VideoRecord]) -> dict:
    """Re-score every saved model on each split and compare with the stored test predictions.

    A large gap between training and test accuracy is the signature of
    overfitting. The stored predictions must be reproduced exactly, which shows
    that the reported results really come from the saved checkpoints.
    """
    class_names = list(config.classes)
    by_split = {split: [r for r in records if r.split == split] for split in SPLITS}
    caches = {split: {} for split in SPLITS}
    report = {}
    for key in trained_models(config):
        accuracy, predictions = {}, {}
        for split, items in by_split.items():
            truth = np.array([class_names.index(r.label) for r in items])
            predictions[split] = cached_scores(config, key, items, caches[split])[0].argmax(axis=1)
            accuracy[split] = float((predictions[split] == truth).mean())
        stored_path = config.results_dir / key / "predictions.csv"
        matches = None
        if stored_path.is_file():
            with stored_path.open(newline="", encoding="utf-8") as handle:
                stored = {row["video_id"]: row["predicted_label"] for row in csv.DictReader(handle)}
            matches = stored == {r.video_id: class_names[p] for r, p in zip(by_split["test"], predictions["test"])}
        correct = predictions["test"] == np.array([class_names.index(r.label) for r in by_split["test"]])
        group_keys = [r.group_key for r in by_split["test"]]
        report[key] = {
            "accuracy": accuracy,
            "train_minus_test_accuracy": accuracy["train"] - accuracy["test"],
            "stored_test_predictions_reproduced": matches,
            "test_accuracy_ci95_group_bootstrap": group_bootstrap_interval(correct, group_keys, config.seed),
            "test_accuracy_by_source_video": source_video_breakdown(correct, group_keys),
        }
    return report


# --- how reliable is validation, and would more data help? ---------------------------------


def _group_numbers(records: list[VideoRecord]) -> np.ndarray:
    return np.array([parse_ucf101_name(r.path.stem)[1] for r in records])


def validation_variability(
    pooled: np.ndarray, labels: np.ndarray, groups: np.ndarray, c: float, seed: int, val_groups: int, repetitions: int = 200
) -> dict:
    """Repeat the train/validation split with different random validation groups.

    Shows how much a validation accuracy measured on ``val_groups`` groups moves
    purely because of which groups were drawn.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    scores = []
    for _ in range(repetitions):
        held_out = rng.choice(unique, size=val_groups, replace=False)
        mask = np.isin(groups, held_out)
        model = baseline.build_svm(c, seed).fit(pooled[~mask], labels[~mask])
        scores.append(float((model.predict(pooled[mask]) == labels[mask]).mean()))
    scores = np.array(scores)
    return {
        "repetitions": repetitions,
        "validation_groups": val_groups,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "p05": float(np.percentile(scores, 5)),
        "p95": float(np.percentile(scores, 95)),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


def learning_curve(
    pooled: np.ndarray, labels: np.ndarray, groups: np.ndarray, c: float, seed: int, eval_groups: int, repetitions: int = 60
) -> list[dict]:
    """Accuracy on held-out groups as a function of the number of training groups."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    max_train = len(unique) - eval_groups
    sizes = [n for n in (1, 2, 4, 6, 8, 10, 12, 14, 16) if n < max_train] + [max_train]
    curve = []
    for size in sizes:
        scores = []
        for _ in range(repetitions):
            order = rng.permutation(unique)
            held_out, training = order[:eval_groups], order[eval_groups : eval_groups + size]
            train_mask, eval_mask = np.isin(groups, training), np.isin(groups, held_out)
            model = baseline.build_svm(c, seed).fit(pooled[train_mask], labels[train_mask])
            scores.append(float((model.predict(pooled[eval_mask]) == labels[eval_mask]).mean()))
        curve.append({"training_groups_per_class": int(size), "mean_accuracy": float(np.mean(scores)), "std": float(np.std(scores))})
    return curve


def label_permutation_test(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, c: float, seed: int, repetitions: int = 100
) -> dict:
    """Does validation accuracy really come from the training labels?

    The classifier is refitted many times with the training labels shuffled.
    If the pipeline is sound, those fits score at chance on the real validation
    labels. A real result must clearly beat them; if shuffled labels scored
    well too, the validation labels would be reaching the model some other way.
    The p-value is the share of shuffled fits that match or beat the real one.
    """
    rng = np.random.default_rng(seed)
    true_accuracy = float((baseline.build_svm(c, seed).fit(train_x, train_y).predict(val_x) == val_y).mean())
    shuffled = []
    for _ in range(repetitions):
        model = baseline.build_svm(c, seed).fit(train_x, rng.permutation(train_y))
        shuffled.append(float((model.predict(val_x) == val_y).mean()))
    p_value = (sum(score >= true_accuracy for score in shuffled) + 1) / (repetitions + 1)
    return {
        "true_labels_accuracy": true_accuracy,
        "shuffled_labels_mean_accuracy": float(np.mean(shuffled)),
        "shuffled_labels_max_accuracy": float(np.max(shuffled)),
        "chance": 1 / len(np.unique(val_y)),
        "repetitions": repetitions,
        "p_value": p_value,
        "passes": p_value < 0.05,
    }


def summarise_learning_curve(history: list[dict]) -> dict:
    """Where validation loss bottomed out, and what happened afterwards.

    Overfitting shows as training loss that keeps falling after the epoch with
    the best validation loss while validation loss rises. Early stopping keeps
    the weights of the best epoch, so this describes what was avoided.
    """
    best = min(history, key=lambda entry: entry["val_loss"])
    last = history[-1]
    return {
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "at_best_epoch": {"train_loss": best["train_loss"], "val_loss": best["val_loss"]},
        "at_last_epoch": {"train_loss": last["train_loss"], "val_loss": last["val_loss"]},
        "gap_at_best_epoch": best["val_loss"] - best["train_loss"],
        "overfitting_after_best_epoch": last["val_loss"] > best["val_loss"] and last["train_loss"] < best["train_loss"],
    }


def learning_curve_summaries(config: Config) -> dict[str, dict]:
    """Learning-curve summary for every trained model that was fitted by gradient descent."""
    summaries = {}
    for key in MODELS:
        training = read_json_if_exists(config.results_dir / key / "training.json")
        if training and training.get("history"):
            summaries[key] = summarise_learning_curve(training["history"])
    return summaries


# --- entry point -------------------------------------------------------------------------


def run_diagnostics(config: Config) -> dict:
    records = load_manifest(config)
    log.info("Auditing %d videos...", len(records))
    all_features, _ = load_features(config, records)
    pooled_all = baseline.temporal_mean_pool(all_features)

    report = {
        "dataset": audit_dataset(config, records),
        "labels_and_files": check_labels_and_files(records),
        "split_independence": check_split_independence(records),
        "exact_duplicate_files": find_exact_duplicates(records),
        "near_duplicates": near_duplicate_report(records, pooled_all),
        "feature_cache": check_feature_cache(config, records),
        "saved_models": verify_saved_models(config, records),
        "training_curves": learning_curve_summaries(config),
        "validation_variability": {},
        "learning_curve": {},
        "label_permutation_test": {},
    }

    # Repeated group hold-out on the training and validation videos only: a far steadier
    # measure of how well a feature set generalises to new source videos than one split.
    development = [r for r in records if r.split in ("train", "val")]
    is_val = np.array([r.split == "val" for r in development])
    dev_groups = _group_numbers(development)
    val_groups = len({r.group for r in records if r.split == "val"})
    for name, (dev_pooled, dev_y) in _development_features(config, development).items():
        log.info("Repeated hold-out of %d validation groups with %s features...", val_groups, name)
        _, selected, _ = baseline.fit_and_select_svm(
            dev_pooled[~is_val], dev_y[~is_val], dev_pooled[is_val], dev_y[is_val], config
        )
        report["validation_variability"][name] = {
            "svm_C": selected["C"],
            **validation_variability(dev_pooled, dev_y, dev_groups, selected["C"], config.seed, val_groups),
        }
        report["learning_curve"][name] = learning_curve(dev_pooled, dev_y, dev_groups, selected["C"], config.seed, val_groups)
        report["label_permutation_test"][name] = label_permutation_test(
            dev_pooled[~is_val], dev_y[~is_val], dev_pooled[is_val], dev_y[is_val], selected["C"], config.seed
        )
    write_json(config.results_dir / "diagnostics.json", report)
    _log_summary(report)
    return report


def _development_features(config: Config, development: list[VideoRecord]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One vector per video for each cached feature set (frame or clip features averaged)."""
    features, labels = load_features(config, development)
    feature_sets = {"mobilenet_v3_small": (baseline.temporal_mean_pool(features), labels)}
    try:
        from .r3d import load_r3d_features

        clip_features, labels = load_r3d_features(config, development)
        feature_sets["r3d_18"] = (clip_features.mean(axis=1), labels)
    except FileNotFoundError:
        log.info("R3D-18 features are not cached, so only the MobileNetV3 features are analysed.")
    return feature_sets


def _log_summary(report: dict) -> None:
    files, split = report["labels_and_files"], report["split_independence"]
    log.info("Files: %d checked, %d undecodable, %d name/label mismatches, %d exact duplicate sets",
             files["videos_checked"], len(files["undecodable"]), len(files["name_label_mismatches"]),
             len(report["exact_duplicate_files"]))  # fmt: skip
    log.info("Split: %d source videos, %d in more than one split", split["source_videos"],
             split["source_videos_in_more_than_one_split"])  # fmt: skip
    near = report["near_duplicates"]
    log.info("Near duplicates: %d cross-split pairs above the within-source median similarity %.3f",
             near["cross_split_pairs_above_within_source_median"], near["within_source_video_similarity"]["median"])  # fmt: skip
    for key, model in report["saved_models"].items():
        accuracy = model["accuracy"]
        log.info("%-9s train %.3f | val %.3f | test %.3f | reproduced stored predictions: %s | group-bootstrap CI %.3f-%.3f",
                 key, accuracy["train"], accuracy["val"], accuracy["test"], model["stored_test_predictions_reproduced"],
                 *model["test_accuracy_ci95_group_bootstrap"])  # fmt: skip
    for name, variability in report["validation_variability"].items():
        log.info("%s features: accuracy over %d random draws of %d validation groups: mean %.3f, 5th-95th percentile %.3f-%.3f",
                 name, variability["repetitions"], variability["validation_groups"], variability["mean"],
                 variability["p05"], variability["p95"])  # fmt: skip
        for point in report["learning_curve"][name]:
            log.info("  %2d training groups per class -> %.3f (+/- %.3f)", point["training_groups_per_class"],
                     point["mean_accuracy"], point["std"])  # fmt: skip
        permutation = report["label_permutation_test"][name]
        log.info("  label permutation test: validation accuracy %.3f with true labels, %.3f on average with shuffled "
                 "training labels (chance %.2f), p = %.3f", permutation["true_labels_accuracy"],
                 permutation["shuffled_labels_mean_accuracy"], permutation["chance"], permutation["p_value"])  # fmt: skip
