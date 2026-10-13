"""Baseline and LSTM tests on synthetic feature sequences (no CNN, no dataset)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from visionact import baseline, lstm
from visionact.config import Config
from visionact.utils import ModelNotTrainedError

HP = lstm.LSTMHyperparameters(
    hidden_dim=16, num_layers=1, dropout=0.0, learning_rate=1e-2, weight_decay=0.0, batch_size=16, epochs=40, patience=40
)


def order_task(num_videos: int, seed: int, time_steps: int = 6, dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Two classes built from the same frames in opposite order.

    Class 0 shows pattern A then pattern B; class 1 shows B then A. The mean
    over time is the same for both, so only a model that uses order can tell
    them apart.
    """
    rng = np.random.default_rng(seed)
    pattern_a, pattern_b = np.eye(dim, dtype=np.float32)[0] * 3, np.eye(dim, dtype=np.float32)[1] * 3
    half = time_steps // 2
    labels = np.arange(num_videos) % 2
    sequences = np.empty((num_videos, time_steps, dim), np.float32)
    for index, label in enumerate(labels):
        first, second = (pattern_a, pattern_b) if label == 0 else (pattern_b, pattern_a)
        sequences[index, :half], sequences[index, half:] = first, second
    sequences += rng.normal(0, 0.3, sequences.shape).astype(np.float32)
    return sequences, labels.astype(np.int64)


# --- baseline --------------------------------------------------------------------


def test_mean_pooling_ignores_frame_order():
    x, _ = order_task(4, seed=0)
    np.testing.assert_allclose(baseline.temporal_mean_pool(x), baseline.temporal_mean_pool(x[:, ::-1]), atol=1e-6)
    assert baseline.temporal_mean_pool(x).shape == (4, 8)
    with pytest.raises(ValueError):
        baseline.temporal_mean_pool(x[0])


def test_svm_scores_cover_all_classes_including_binary_and_unseen():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 1, (30, 5)), rng.normal(2, 1, (30, 5))])
    y = np.repeat([0, 1], 30)
    model = baseline.build_svm(1.0, seed=0).fit(x, y)
    scores = baseline.class_scores(model, x, num_classes=3)  # class 2 never seen in training
    assert scores.shape == (60, 3)
    assert (scores.argmax(axis=1) == y).mean() > 0.95
    assert np.isneginf(scores[:, 2]).all(), "an unseen class can never be predicted"


# --- LSTM ------------------------------------------------------------------------


def test_lstm_forward_shape_and_input_check():
    model = lstm.LSTMClassifier(input_dim=8, hidden_dim=16, num_classes=3)
    assert model(torch.zeros(5, 6, 8)).shape == (5, 3)
    assert model(torch.zeros(1, 1, 8)).shape == (1, 3), "a single-frame sequence is still valid"
    with pytest.raises(ValueError):
        model(torch.zeros(6, 8))


def test_lstm_output_depends_on_frame_order():
    torch.manual_seed(0)
    model = lstm.LSTMClassifier(8, 16, 2, dropout=0.0).eval()
    x, _ = order_task(4, seed=1)
    forward = lstm.predict_logits(model, x)
    backward = lstm.predict_logits(model, np.ascontiguousarray(x[:, ::-1]))
    assert not np.allclose(forward, backward, atol=1e-3)


def test_lstm_learns_an_order_only_task_that_defeats_mean_pooling():
    x_train, y_train = order_task(120, seed=0)
    x_val, y_val = order_task(60, seed=1)
    result = lstm.fit_lstm(x_train, y_train, x_val, y_val, HP, num_classes=2, seed=0)
    assert result["val_accuracy"] > 0.95

    svm = baseline.build_svm(1.0, seed=0).fit(baseline.temporal_mean_pool(x_train), y_train)
    svm_accuracy = (svm.predict(baseline.temporal_mean_pool(x_val)) == y_val).mean()
    assert svm_accuracy < 0.7, "order-blind features should be near chance on this task"


def test_training_is_reproducible_for_a_fixed_seed():
    x_train, y_train = order_task(40, seed=0)
    x_val, y_val = order_task(20, seed=1)
    short = replace(HP, epochs=4)
    first = lstm.fit_lstm(x_train, y_train, x_val, y_val, short, 2, seed=7)
    second = lstm.fit_lstm(x_train, y_train, x_val, y_val, short, 2, seed=7)
    assert first["history"] == second["history"]
    other = lstm.fit_lstm(x_train, y_train, x_val, y_val, short, 2, seed=8)
    assert first["history"] != other["history"]


def test_early_stopping_restores_the_best_epoch():
    x_train, y_train = order_task(40, seed=0)
    rng = np.random.default_rng(5)
    x_val, y_val = order_task(20, seed=1)
    y_val = rng.permutation(y_val)  # unlearnable validation labels: validation loss cannot keep improving
    result = lstm.fit_lstm(x_train, y_train, x_val, y_val, replace(HP, epochs=60, patience=3), 2, seed=0)
    assert result["epochs_run"] < 60, "training should stop early"
    assert result["epochs_run"] == result["best_epoch"] + 3
    losses = [entry["val_loss"] for entry in result["history"]]
    assert result["val_loss"] == pytest.approx(min(losses), abs=1e-3)
    # The returned model is the best checkpoint, not the last epoch.
    logits = torch.from_numpy(lstm.predict_logits(result["model"], x_val))
    restored = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y_val)).item()
    assert restored == pytest.approx(result["val_loss"], abs=1e-4)


def test_normalisation_uses_training_statistics():
    model = lstm.LSTMClassifier(4, 8, 2)
    features = np.random.default_rng(0).normal(5.0, 2.0, (50, 6, 4)).astype(np.float32)
    model.set_normalisation(features)
    np.testing.assert_allclose(model.feature_mean.numpy(), features.reshape(-1, 4).mean(axis=0), atol=1e-4)
    assert "feature_mean" in model.state_dict(), "statistics must be saved with the checkpoint"


def test_balanced_class_weights():
    np.testing.assert_allclose(lstm.balanced_class_weights(np.array([0, 1, 0, 1]), 2), [1.0, 1.0])
    weights = lstm.balanced_class_weights(np.array([0, 0, 0, 1]), 3)  # class 2 absent
    assert weights[1] > weights[0] and torch.isfinite(weights).all()


def test_hyperparameter_grid_size():
    config = Config(search_hidden_dims=(32, 64), search_dropouts=(0.2,), search_learning_rates=(1e-3, 1e-4))
    grid = lstm.hyperparameter_grid(config)
    assert len(grid) == 4 and {hp.hidden_dim for hp in grid} == {32, 64}
    assert all(hp.epochs == config.epochs for hp in grid)


def test_loading_untrained_models_raises_a_clear_error(tmp_path):
    config = Config(artifacts_dir=tmp_path)
    with pytest.raises(ModelNotTrainedError, match="train-lstm"):
        lstm.load_lstm(config)
    with pytest.raises(ModelNotTrainedError, match="train-baseline"):
        baseline.load_baseline(config)
