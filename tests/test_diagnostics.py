"""Known-answer tests for the diagnostic checks (planted leakage, duplicates and clustered errors)."""

from pathlib import Path

import numpy as np
import pytest

from visionact import diagnostics
from visionact.dataset import VideoRecord


def record(label: str, group: int, clip: int, split: str, folder: Path = Path(".")) -> VideoRecord:
    stem = f"v_{label}_g{group:02d}_c{clip:02d}"
    return VideoRecord(f"{label}/{stem}", label, f"g{group:02d}", folder / f"{stem}.avi", split=split)


def test_split_independence_counts_source_videos_and_rejects_leakage():
    clean = [record("A", 1, 1, "test"), record("A", 1, 2, "test"), record("A", 8, 1, "train"), record("B", 1, 1, "val")]
    report = diagnostics.check_split_independence(clean)
    assert report["source_videos"] == 3 and report["source_videos_in_more_than_one_split"] == 0
    assert report["source_videos_per_split"] == {"train": 1, "val": 1, "test": 1}
    with pytest.raises(ValueError, match="leakage"):
        diagnostics.check_split_independence(clean + [record("A", 8, 2, "test")])


def test_exact_duplicate_files_are_reported(tmp_path):
    items = [record("A", group, 1, "train", tmp_path) for group in (1, 2, 3)]
    for item, payload in zip(items, (b"same bytes", b"same bytes", b"different")):
        item.path.write_bytes(payload)
    assert diagnostics.find_exact_duplicates(items) == [["A/v_A_g01_c01", "A/v_A_g02_c01"]]


def test_near_duplicates_across_splits_are_flagged():
    rng = np.random.default_rng(0)
    base = {group: rng.normal(0, 1, 32) for group in (1, 8, 9)}
    items, features = [], []
    for group, split in ((1, "test"), (8, "train"), (9, "train")):
        for clip in (1, 2, 3):
            items.append(record("A", group, clip, split))
            features.append(base[group] + rng.normal(0, 0.05, 32))  # clips of one source video look alike
    clean = diagnostics.near_duplicate_report(items, np.array(features))
    assert clean["cross_split_pairs_above_within_source_median"] == 0

    # Plant the same footage under another group number, in another split.
    items.append(record("A", 10, 1, "train"))
    features.append(base[1] + rng.normal(0, 0.001, 32))
    planted = diagnostics.near_duplicate_report(items, np.array(features))
    assert planted["cross_split_pairs_above_within_source_median"] == 3
    assert {pair["b"] for pair in planted["most_similar_cross_split_pairs"]} == {"A/v_A_g10_c01"}


def test_source_video_breakdown_shows_whether_errors_are_clustered():
    from visionact.evaluation import source_video_breakdown

    keys = [record("A", group, clip, "test").group_key for group in (1, 2, 3, 4) for clip in (1, 2, 3, 4)]
    correct = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])  # group 1 all wrong, one stray error
    summary = source_video_breakdown(correct, keys)
    assert summary["total"] == 4
    assert (summary["all_clips_wrong"], summary["all_clips_correct"], summary["mixed"]) == (1, 2, 1)
    assert summary["share_of_errors_in_mostly_wrong_source_videos"] == pytest.approx(4 / 5)
    perfect = source_video_breakdown(np.ones(16), keys)
    assert perfect["all_clips_correct"] == 4 and perfect["share_of_errors_in_mostly_wrong_source_videos"] == 0.0


def epoch(number: int, train_loss: float, val_loss: float) -> dict:
    return {"epoch": number, "train_loss": train_loss, "val_loss": val_loss}


def test_learning_curve_summary_flags_training_past_the_best_validation_epoch():
    overfitting = [epoch(1, 1.0, 0.9), epoch(2, 0.5, 0.6), epoch(3, 0.2, 0.7), epoch(4, 0.05, 0.9)]
    summary = diagnostics.summarise_learning_curve(overfitting)
    assert summary["best_epoch"] == 2 and summary["epochs_run"] == 4
    assert summary["gap_at_best_epoch"] == pytest.approx(0.1)
    assert summary["overfitting_after_best_epoch"] is True, "training loss fell while validation loss rose"

    still_improving = [epoch(1, 1.0, 0.9), epoch(2, 0.6, 0.7), epoch(3, 0.4, 0.5)]
    summary = diagnostics.summarise_learning_curve(still_improving)
    assert summary["best_epoch"] == 3 and summary["overfitting_after_best_epoch"] is False


def group_effect_data(groups: int = 12, clips: int = 4, seed: int = 0):
    """Two classes whose features carry a class signal plus a strong per-group offset."""
    rng = np.random.default_rng(seed)
    features, labels, group_ids = [], [], []
    for label in (0, 1):
        for group in range(groups):
            offset = rng.normal(0, 1.5, 6)
            for _ in range(clips):
                features.append(np.r_[2.0 * label - 1.0, np.zeros(5)] + offset + rng.normal(0, 0.2, 6))
                labels.append(label)
                group_ids.append(group)
    return np.array(features), np.array(labels), np.array(group_ids)


def test_validation_variability_reports_the_spread_over_random_validation_groups():
    features, labels, groups = group_effect_data()
    report = diagnostics.validation_variability(features, labels, groups, c=1.0, seed=0, val_groups=2, repetitions=40)
    assert report["repetitions"] == 40 and report["validation_groups"] == 2
    assert report["min"] <= report["p05"] <= report["mean"] <= report["p95"] <= report["max"]
    assert report["max"] - report["min"] > 0.1, "which groups are held out changes the score"
    again = diagnostics.validation_variability(features, labels, groups, c=1.0, seed=0, val_groups=2, repetitions=40)
    assert again == report, "the experiment is seeded"


def test_label_permutation_test_passes_when_accuracy_comes_from_the_training_labels():
    rng = np.random.default_rng(0)
    train_y, val_y = np.arange(80) % 2, np.arange(40) % 2
    train_x = np.column_stack([train_y * 4.0, rng.normal(0, 1, (80, 3))])
    val_x = np.column_stack([val_y * 4.0, rng.normal(0, 1, (40, 3))])
    result = diagnostics.label_permutation_test(train_x, train_y, val_x, val_y, c=1.0, seed=0, repetitions=40)
    assert result["true_labels_accuracy"] > 0.95
    assert abs(result["shuffled_labels_mean_accuracy"] - 0.5) < 0.2, "training on shuffled labels gives chance accuracy"
    assert result["chance"] == 0.5 and result["repetitions"] == 40
    assert result["p_value"] == pytest.approx(1 / 41) and result["passes"] is True


def test_label_permutation_test_fails_when_there_is_nothing_real_to_learn():
    rng = np.random.default_rng(1)
    train_x, val_x = rng.normal(0, 1, (80, 4)), rng.normal(0, 1, (40, 4))
    train_y, val_y = np.arange(80) % 2, np.arange(40) % 2
    result = diagnostics.label_permutation_test(train_x, train_y, val_x, val_y, c=1.0, seed=0, repetitions=40)
    assert result["p_value"] > 0.05 and result["passes"] is False


def test_learning_curve_rises_with_more_training_groups():
    features, labels, groups = group_effect_data()
    curve = diagnostics.learning_curve(features, labels, groups, c=1.0, seed=0, eval_groups=2, repetitions=30)
    sizes = [point["training_groups_per_class"] for point in curve]
    assert sizes == sorted(sizes) and sizes[0] == 1 and sizes[-1] == 10, "from one group up to all that remain"
    assert curve[-1]["mean_accuracy"] > curve[0]["mean_accuracy"] + 0.05
