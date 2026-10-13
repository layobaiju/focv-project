import numpy as np
import pytest

from visionact import evaluation

CLASSES = ["A", "B", "C"]


def test_metrics_on_a_known_example():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    metrics = evaluation.compute_metrics(y_true, y_pred, CLASSES)
    assert metrics["num_videos"] == 6
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["confusion_matrix"] == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert metrics["per_class"]["A"] == {"precision": pytest.approx(2 / 3), "recall": pytest.approx(2 / 3), "f1": pytest.approx(2 / 3), "support": 3}
    assert metrics["per_class"]["B"]["recall"] == 1.0
    assert metrics["per_class"]["C"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 0.0) / 3)
    low, high = metrics["accuracy_ci95"]
    assert low < metrics["accuracy"] < high


def test_metrics_handle_a_class_with_no_test_videos():
    metrics = evaluation.compute_metrics(np.array([0, 1, 1]), np.array([0, 1, 1]), CLASSES)
    assert metrics["accuracy"] == 1.0
    assert metrics["per_class"]["C"]["support"] == 0 and metrics["per_class"]["C"]["f1"] == 0.0
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["macro_f1"] == pytest.approx(2 / 3), "an empty class counts as zero in the macro average"


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        evaluation.compute_metrics(np.array([]), np.array([]), CLASSES)


def test_wilson_interval():
    assert evaluation.wilson_interval(0, 0) == (0.0, 1.0)
    low, high = evaluation.wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=0.002) and high == pytest.approx(0.596, abs=0.002)
    low, high = evaluation.wilson_interval(125, 125)
    assert high == 1.0 and 0.96 < low < 1.0, "a perfect score on 125 videos still leaves uncertainty"


def test_paired_comparison_counts_and_p_value():
    y_true = np.zeros(10, dtype=int)
    first = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    second = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    result = evaluation.paired_comparison(y_true, first, second)
    assert (result["both_correct"], result["only_first_correct"], result["only_second_correct"], result["both_wrong"]) == (4, 2, 3, 1)
    assert result["mcnemar_exact_p"] == pytest.approx(1.0)
    assert evaluation.paired_comparison(y_true, first, first)["mcnemar_exact_p"] == 1.0
    one_sided = evaluation.paired_comparison(y_true, np.ones(10, dtype=int), y_true)
    assert one_sided["only_second_correct"] == 10 and one_sided["mcnemar_exact_p"] == pytest.approx(2 / 2**10)


def test_group_bootstrap_interval_reflects_clustered_errors():
    """Same 80% accuracy twice: errors spread evenly, or concentrated in whole source videos."""
    groups = [index // 5 for index in range(100)]  # 20 source videos with 5 clips each
    scattered = np.array([0, 1, 1, 1, 1] * 20)
    clustered = np.array([0] * 20 + [1] * 80)  # four source videos entirely wrong
    low_s, high_s = evaluation.group_bootstrap_interval(scattered, groups, seed=0)
    low_c, high_c = evaluation.group_bootstrap_interval(clustered, groups, seed=0)
    assert (low_s, high_s) == pytest.approx((0.8, 0.8)), "every resample of whole groups scores exactly 80%"
    assert low_c < 0.7 and high_c > 0.9, "clustered errors leave much more uncertainty"
    assert evaluation.group_bootstrap_interval(clustered, groups, seed=0) == (low_c, high_c)
    # The clip-level interval cannot tell the two situations apart.
    assert evaluation.wilson_interval(80, 100) == evaluation.wilson_interval(int(clustered.sum()), 100)


def test_reported_interval_falls_back_when_resampling_cannot_show_uncertainty():
    usual = {"accuracy_ci95": [0.69, 0.84], "accuracy_ci95_group_bootstrap": [0.63, 0.91]}
    assert evaluation.reported_interval(usual) == (0.63, 0.91, "source videos")
    # With no errors every resample scores 100%, so the bootstrap interval collapses to a point.
    perfect = {"accuracy_ci95": [0.97, 1.0], "accuracy_ci95_group_bootstrap": [1.0, 1.0]}
    assert evaluation.reported_interval(perfect) == (0.97, 1.0, "independent videos")


def test_comparison_markdown_lists_every_model_and_marks_the_default():
    model = {
        "display_name": "Model X",
        "accuracy": 0.9,
        "accuracy_ci95": [0.8, 0.95],
        "accuracy_ci95_group_bootstrap": [0.7, 1.0],
        "macro_f1": 0.89,
        "validation": {"val_macro_f1": 0.91},
        "training_seconds": 1.5,
        "inference_ms_per_video": None,
        "total_size_mb": 6.4,
    }
    text = evaluation.comparison_markdown(
        {
            "test_videos": 10,
            "class_names": CLASSES,
            "best_validated_model": "y",
            "models": {"x": model, "y": {**model, "display_name": "Model Y"}},
        }
    )
    assert "Model X" in text and "Model Y" in text and "not measured" in text
    assert "70.0% to 100.0%" in text, "the source-video interval is reported"
    row_x = next(line for line in text.splitlines() if line.startswith("| Model X"))
    row_y = next(line for line in text.splitlines() if line.startswith("| Model Y"))
    assert "best on validation" in row_y and "best on validation" not in row_x
