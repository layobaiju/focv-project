"""End-to-end run on a tiny synthetic dataset: every stage, every model, the CLI and inference."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from tests.conftest import make_video
from visionact import dataset, evaluation, inference, registry
from visionact.cli import main
from visionact.motion_experiment import run_motion_experiment, video_motion_descriptor
from visionact.utils import ModelNotTrainedError

pytestmark = pytest.mark.slow  # every test here uses the session-wide ``trained`` fixture from conftest

ALL_MODELS = ["cnn_svm", "cnn_lstm", "cnn_svm_aug", "cnn_lstm_aug", "r3d18_head"]


def read_predictions(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_model_is_trained_and_scored_on_the_same_test_videos(trained):
    config, _ = trained
    results = evaluation.load_results(config)
    assert list(results["models"]) == ALL_MODELS == registry.trained_models(config)

    records = dataset.load_manifest(config)
    test_ids = {r.video_id for r in dataset.records_for_split(records, "test")}
    seen_in_training = {r.video_id for r in records if r.split in ("train", "val")}
    fingerprint = dataset.experiment_record(config)["manifest_sha256"]
    for key, model in results["models"].items():
        folder = config.results_dir / key
        assert (folder / "confusion_matrix.png").stat().st_size > 0
        rows = read_predictions(folder / "predictions.csv")
        assert {row["video_id"] for row in rows} == test_ids
        assert not {row["video_id"] for row in rows} & seen_in_training
        # Reported accuracy is recomputed here from the saved per-video predictions.
        assert model["accuracy"] == pytest.approx(np.mean([int(row["correct"]) for row in rows]))
        assert model["num_videos"] == len(test_ids) == results["test_videos"]
        assert np.array(model["confusion_matrix"]).sum() == len(test_ids)
        low, high = model["accuracy_ci95_group_bootstrap"]
        assert 0.0 <= low <= model["accuracy"] <= high <= 1.0
        sources = model["source_videos"]
        assert sources["total"] == results["test_source_videos"] == 14
        assert sources["all_clips_correct"] + sources["all_clips_wrong"] + sources["mixed"] == 14
        assert read_json(folder / "metrics.json")["accuracy"] == model["accuracy"]
        # Cost figures are measured, and the total size includes the model's backbone.
        assert model["training_seconds"] > 0 and model["inference_ms_per_video"] > 0
        assert model["backbone"] in results["feature_extractors"]
        assert model["total_size_mb"] > model["model_size_mb"] > 0
        # Every model records the validation score it was selected with and the exact split it used.
        assert model["validation"]["val_macro_f1"] == registry.validation_score(config, key)["val_macro_f1"]
        assert read_json(folder / "training.json")["experiment"]["manifest_sha256"] == fingerprint
    assert "R3D-18" in (config.results_dir / "comparison.md").read_text()


def test_augmentation_is_recorded_only_for_the_models_that_used_it(trained):
    config, _ = trained
    results = evaluation.load_results(config)
    for key in ("cnn_svm", "cnn_lstm"):
        assert read_json(config.results_dir / key / "training.json")["experiment"]["augmentation"] is None
        assert results["models"][key]["augmented_training"] is False
    for key in ("cnn_svm_aug", "cnn_lstm_aug"):
        augmentation = read_json(config.results_dir / key / "training.json")["experiment"]["augmentation"]
        assert augmentation["applied_to"] == "training split only"
        assert results["models"][key]["augmented_training"] is True
    r3d_training = read_json(config.results_dir / "r3d18_head" / "training.json")
    assert results["models"]["r3d18_head"]["augmented_training"] is r3d_training["selected"]["augmented"]


def test_comparisons_and_default_model_follow_the_protocol(trained):
    config, _ = trained
    results = evaluation.load_results(config)
    test_videos = results["test_videos"]
    assert results["reference_model"] == "cnn_svm"
    assert set(results["paired_comparisons"]) == set(ALL_MODELS) - {"cnn_svm"}
    for comparison in results["paired_comparisons"].values():
        counted = sum(comparison[k] for k in ("both_correct", "only_first_correct", "only_second_correct", "both_wrong"))
        assert counted == test_videos and 0.0 <= comparison["mcnemar_exact_p"] <= 1.0
    # The default model is the best on validation; test accuracy plays no part.
    assert results["best_validated_model"] == registry.best_validated_model(config)

    assert set(results["temporal_order_ablation"]) == {"cnn_lstm", "cnn_lstm_aug"}
    ablation = results["temporal_order_ablation"]["cnn_lstm"]
    assert ablation["original_order_accuracy"] == pytest.approx(results["models"]["cnn_lstm"]["accuracy"])

    timing = results["timing"]
    assert timing["videos_timed"] == 2 and set(timing["per_model"]) == set(ALL_MODELS)
    for key, stages in timing["per_model"].items():
        assert stages["total_ms"] == pytest.approx(stages["decode_ms"] + stages["backbone_ms"] + stages["classifier_ms"])
        assert results["models"][key]["inference_ms_per_video"] == stages["total_ms"]
    assert (config.results_dir / "cnn_lstm" / "training_curves.png").is_file()
    assert (config.results_dir / "r3d18_head" / "training_curves.png").is_file()


def test_every_trained_model_classifies_a_new_video(trained, tmp_path):
    config, config_path = trained
    clip = make_video(tmp_path / "new_clip.avi", num_frames=5)
    backbones: dict = {}
    for key in ALL_MODELS:
        prediction = inference.load_predictor(config, key, backbones).predict(clip)
        assert prediction.model_key == key and prediction.label in config.classes
        assert set(prediction.scores) == set(config.classes)
        assert prediction.label == max(prediction.scores, key=prediction.scores.get)
        if prediction.score_kind == "softmax":
            assert sum(prediction.scores.values()) == pytest.approx(1.0, abs=1e-5)
            assert prediction.confidence == max(prediction.scores.values())
        else:
            assert registry.get_spec(key).family == "svm" and prediction.confidence is None
        assert prediction.total_seconds > 0 and prediction.frame_indices
    assert set(backbones) == {"mobilenet_v3_small", "r3d_18"}, "each backbone is loaded once and shared"

    default = inference.load_predictor(config, backbones=backbones)
    assert default.spec.key == registry.best_validated_model(config)
    assert main(["predict", str(clip), "--config", str(config_path)]) == 0
    assert main(["predict", str(clip), "--config", str(config_path), "--model", "cnn_lstm"]) == 0


def test_predictor_errors_are_actionable(trained, tmp_path):
    config, _ = trained
    with pytest.raises(ValueError, match="Unknown model"):
        inference.load_predictor(config, "resnet152")
    untrained = replace(config, artifacts_dir=tmp_path / "empty")
    with pytest.raises(ModelNotTrainedError, match="train-lstm"):
        inference.load_predictor(untrained, "cnn_lstm")
    with pytest.raises(ModelNotTrainedError, match="No trained model"):
        inference.load_predictor(untrained)


def test_cli_reports_errors_without_a_traceback(trained, tmp_path):
    _, config_path = trained
    assert main(["predict", str(tmp_path / "missing.avi"), "--config", str(config_path)]) == 1
    assert main(["predict", str(tmp_path / "missing.avi"), "--config", str(config_path), "--model", "nope"]) == 1
    assert main(["evaluate", "--config", str(tmp_path / "no_such_config.json")]) == 1


def test_evaluation_refuses_models_trained_on_other_features(trained):
    config, _ = trained
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.evaluate_models(replace(config, num_frames=config.num_frames + 2), timing_videos=0)


def test_motion_experiment_compares_flow_fusion_on_validation(trained):
    config, _ = trained
    results = run_motion_experiment(config)
    assert set(results["models"]) == {"flow_svm", "cnn_pooled_svm", "cnn_flow_svm", "r3d_pooled_svm", "r3d_flow_svm"}
    assert results["test_videos"] == 28 and results["val_videos"] == 8
    for model in results["models"].values():
        assert 0.0 <= model["accuracy"] <= 1.0 and model["num_videos"] == 28
        assert 0.0 <= model["validation"]["val_accuracy"] <= 1.0
    assert results["flow_ms_per_video"] > 0
    # One decision per visual feature set, each derived from validation results only.
    assert set(results["decisions"]) == {"cnn", "r3d"}
    for name, decision in results["decisions"].items():
        visual = results["models"][f"{name}_pooled_svm"]["validation"]
        with_flow = results["models"][f"{name}_flow_svm"]["validation"]
        gained = round((with_flow["val_accuracy"] - visual["val_accuracy"]) * results["val_videos"])
        assert decision["validation_videos_gained"] == gained
        assert isinstance(decision["retained"], bool) and decision["reason"]
    assert (config.results_dir / "motion_experiment.json").is_file()


def test_diagnostics_verify_the_baselines_and_compare_feature_sets(trained):
    from visionact.diagnostics import run_diagnostics

    config, _ = trained
    report = run_diagnostics(config)
    assert report["labels_and_files"]["videos_checked"] == 48 and not report["labels_and_files"]["undecodable"]
    assert report["split_independence"]["source_videos_in_more_than_one_split"] == 0
    assert report["exact_duplicate_files"] == []
    assert list(report["saved_models"]) == ALL_MODELS
    results = evaluation.load_results(config)
    for key, model in report["saved_models"].items():
        assert model["stored_test_predictions_reproduced"] is True
        assert set(model["accuracy"]) == {"train", "val", "test"}
        assert model["accuracy"]["test"] == pytest.approx(results["models"][key]["accuracy"])
    assert set(report["training_curves"]) == {"cnn_lstm", "cnn_lstm_aug", "r3d18_head"}
    assert set(report["label_permutation_test"]) == {"mobilenet_v3_small", "r3d_18"}
    # Both feature sets get the same repeated-holdout analysis, on training and validation data only.
    assert set(report["validation_variability"]) == set(report["learning_curve"]) == {"mobilenet_v3_small", "r3d_18"}
    for name in ("mobilenet_v3_small", "r3d_18"):
        assert 0.0 <= report["validation_variability"][name]["mean"] <= 1.0
        assert [p["training_groups_per_class"] for p in report["learning_curve"][name]] == [1, 2, 3]
    assert (config.results_dir / "diagnostics.json").is_file()


def test_motion_descriptor_shape_and_short_video(tmp_path):
    clip = make_video(tmp_path / "clip.avi", num_frames=12)
    descriptor = video_motion_descriptor(clip, num_pairs=4)
    assert descriptor.shape == (4, 36) and np.isfinite(descriptor).all()
    assert video_motion_descriptor(make_video(tmp_path / "two.avi", num_frames=2), num_pairs=4).shape == (4, 36)
