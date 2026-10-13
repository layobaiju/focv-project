import pytest

from visionact import registry
from visionact.config import Config
from visionact.utils import write_json


def touch(config: Config, key: str, val_macro_f1: float | None = None, val_accuracy: float = 0.5) -> None:
    path = registry.checkpoint_path(config, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if val_macro_f1 is not None:
        selected = {"val_macro_f1": val_macro_f1, "val_accuracy": val_accuracy}
        write_json(config.results_dir / key / "training.json", {"selected": selected})


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(artifacts_dir=tmp_path / "artifacts", results_dir=tmp_path / "results")


def test_registry_lists_the_baselines_first_and_keeps_their_checkpoint_paths(config):
    assert list(registry.MODELS)[:2] == ["cnn_svm", "cnn_lstm"]
    assert registry.checkpoint_path(config, "cnn_svm") == config.baseline_model_path
    assert registry.checkpoint_path(config, "cnn_lstm") == config.lstm_checkpoint_path
    assert registry.checkpoint_path(config, "r3d18_head") == config.r3d_checkpoint_path
    assert len({registry.checkpoint_path(config, key) for key in registry.MODELS}) == len(registry.MODELS)
    with pytest.raises(ValueError, match="Unknown model"):
        registry.checkpoint_path(config, "resnet152")


def test_trained_models_are_those_with_a_checkpoint_in_registry_order(config):
    assert registry.trained_models(config) == []
    touch(config, "r3d18_head")
    touch(config, "cnn_svm")
    assert registry.trained_models(config) == ["cnn_svm", "r3d18_head"]


def test_best_model_is_chosen_by_validation_score_never_by_test(config):
    assert registry.best_validated_model(config) is None
    touch(config, "cnn_svm", val_macro_f1=0.80)
    touch(config, "cnn_lstm", val_macro_f1=0.90)
    touch(config, "r3d18_head")  # trained but without a recorded validation score: cannot be chosen
    assert registry.best_validated_model(config) == "cnn_lstm"
    # Test results on disk must not influence the choice.
    write_json(config.results_dir / "comparison.json", {"models": {"cnn_svm": {"accuracy": 1.0}}})
    assert registry.best_validated_model(config) == "cnn_lstm"


def test_ties_on_macro_f1_fall_back_to_accuracy_then_to_the_simpler_model(config):
    touch(config, "cnn_svm", val_macro_f1=0.9, val_accuracy=0.90)
    touch(config, "cnn_lstm", val_macro_f1=0.9, val_accuracy=0.92)
    assert registry.best_validated_model(config) == "cnn_lstm"
    touch(config, "cnn_lstm", val_macro_f1=0.9, val_accuracy=0.90)
    assert registry.best_validated_model(config) == "cnn_svm", "on a complete tie the earlier (simpler) model wins"
