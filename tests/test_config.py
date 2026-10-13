import json

import pytest

from visionact.config import DEVICE, PROJECT_ROOT, Config


def test_defaults_are_valid_and_cpu_only():
    config = Config()
    assert DEVICE == "cpu"
    assert len(config.classes) == 4
    assert config.features_dir.name == f"mobilenet_v3_small-IMAGENET1K_V1-T{config.num_frames}"


def test_feature_key_changes_with_sequence_length():
    assert Config(num_frames=8).feature_key != Config(num_frames=16).feature_key


def test_overrides_are_coerced_to_field_types():
    config = Config().with_overrides(classes="JumpingJack, PushUps", dataset_dir="some/folder", num_frames="8", seed=None)
    assert config.classes == ("JumpingJack", "PushUps")
    assert config.dataset_dir == PROJECT_ROOT / "some" / "folder"
    assert config.num_frames == 8 and config.seed == 42
    with pytest.raises(KeyError):
        Config().with_overrides(not_a_field=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"classes": ("OnlyOne",)},
        {"classes": ("A", "A")},
        {"num_frames": 0},
        {"dropout": 1.0},
        {"learning_rate": 0.0},
        {"split_mode": "by_frame"},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_load_reads_json_then_applies_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("VISIONACT_DATASET_DIR", raising=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"hidden_dim": 32, "epochs": 5, "svm_c_grid": [0.5, 2]}))
    config = Config.load(path, epochs=7)
    assert config.hidden_dim == 32 and config.epochs == 7 and config.svm_c_grid == (0.5, 2.0)
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "missing.json")
    path.write_text("{not json")
    with pytest.raises(ValueError):
        Config.load(path)


def test_environment_variable_sets_dataset_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONACT_CONFIG", str(tmp_path / "absent.json"))
    monkeypatch.setenv("VISIONACT_DATASET_DIR", str(tmp_path / "videos"))
    assert Config.load().dataset_dir == tmp_path / "videos"


def test_example_config_is_valid_and_matches_the_defaults(monkeypatch):
    for variable in ("VISIONACT_DATASET_DIR", "VISIONACT_ARTIFACTS_DIR", "VISIONACT_RESULTS_DIR"):
        monkeypatch.delenv(variable, raising=False)
    assert Config.load(PROJECT_ROOT / "config.example.json") == Config()


def test_to_dict_is_json_serialisable():
    assert json.loads(json.dumps(Config().to_dict()))["classes"] == list(Config().classes)
