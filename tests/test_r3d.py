"""R3D-18 tests. Those marked slow run the real pretrained network (127 MB of cached weights)."""

from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch
from torchvision.models.video import R3D_18_Weights

from tests.conftest import make_video
from visionact import dataset, r3d
from visionact.augment import ClipAugmentation
from visionact.config import Config
from visionact.utils import ModelNotTrainedError

HP = r3d.HeadHyperparameters(dropout=0.0, weight_decay=1e-4, learning_rate=1e-2, batch_size=32, epochs=200, patience=10)


def smooth_clip(length: int = 16, seed: int = 0) -> list[np.ndarray]:
    """BGR frames (240 x 320) of slowly drifting smooth texture."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 256, (240, 320 + length, 3), dtype=np.uint8), (0, 0), 4.0)
    return [np.ascontiguousarray(base[:, shift : shift + 320]) for shift in range(length)]


def clip_task(num_videos: int, clips: int, seed: int, dim: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable clip features: (N, clips, dim) and labels for two classes."""
    rng = np.random.default_rng(seed)
    labels = np.arange(num_videos) % 2
    centres = np.where(labels[:, None, None] == 0, -1.0, 1.0)
    features = centres + rng.normal(0, 0.8, (num_videos, clips, dim))
    return features.astype(np.float32), labels.astype(np.int64)


# --- classification head (no pretrained weights needed) -------------------------------------


def test_video_logits_are_the_mean_of_the_clip_logits():
    torch.manual_seed(0)
    head = r3d.ClipHead(input_dim=12, num_classes=3).eval()
    features, _ = clip_task(5, clips=4, seed=0)
    video_level = r3d.video_logits(head, features)
    assert video_level.shape == (5, 3)
    per_clip = np.stack([head(torch.from_numpy(features[:, k])).detach().numpy() for k in range(4)], axis=1)
    np.testing.assert_allclose(video_level, per_clip.mean(axis=1), atol=1e-5)


def test_head_learns_separable_features_and_restores_the_best_epoch():
    train_x, train_y = clip_task(80, clips=3, seed=0)
    val_x, val_y = clip_task(40, clips=3, seed=1)
    result = r3d.fit_head(train_x.reshape(-1, 12), np.repeat(train_y, 3), val_x, val_y, HP, num_classes=2, seed=0)
    assert result["val_accuracy"] > 0.95
    assert result["epochs_run"] < HP.epochs, "early stopping should end training"
    assert result["epochs_run"] == result["best_epoch"] + HP.patience
    logits = torch.from_numpy(r3d.video_logits(result["model"], val_x))
    restored = torch.nn.functional.cross_entropy(logits, torch.from_numpy(val_y)).item()
    assert restored == pytest.approx(result["val_loss"], abs=1e-4)
    assert [entry["epoch"] for entry in result["history"]] == list(range(1, result["epochs_run"] + 1))


def test_head_training_is_reproducible():
    train_x, train_y = clip_task(40, clips=2, seed=0)
    val_x, val_y = clip_task(20, clips=2, seed=1)
    short = replace(HP, epochs=5)
    runs = [r3d.fit_head(train_x.reshape(-1, 12), np.repeat(train_y, 2), val_x, val_y, short, 2, seed=3) for _ in range(2)]
    assert runs[0]["history"] == runs[1]["history"]


def test_head_standardises_with_training_statistics():
    head = r3d.ClipHead(4, 2)
    features = np.random.default_rng(0).normal(3.0, 2.0, (200, 4)).astype(np.float32)
    head.set_normalisation(features)
    np.testing.assert_allclose(head.feature_mean.numpy(), features.mean(axis=0), atol=1e-4)
    assert {"feature_mean", "feature_std"} <= set(head.state_dict())


def test_loading_an_untrained_head_raises_a_clear_error(tmp_path):
    with pytest.raises(ModelNotTrainedError, match="train-r3d"):
        r3d.load_r3d_head(Config(artifacts_dir=tmp_path))


def test_feature_keys_identify_the_extraction_settings():
    config = Config()
    assert "r3d_18" in config.r3d_feature_key and "KINETICS400" in config.r3d_feature_key
    assert replace(config, r3d_clips_per_video=3).r3d_feature_key != config.r3d_feature_key
    assert replace(config, r3d_clip_len=8).r3d_feature_key != config.r3d_feature_key
    assert config.augmentation_key != replace(config, augment_hflip=False).augmentation_key
    assert config.augmentation_key != replace(config, seed=7).augmentation_key


def test_missing_weights_are_not_downloaded_without_consent(monkeypatch):
    monkeypatch.setattr(r3d, "weights_are_cached", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    with pytest.raises(RuntimeError, match="127 MB"):
        r3d.ensure_weights(assume_yes=False)
    r3d.ensure_weights(assume_yes=True)  # explicit consent: no prompt, no error


# --- the pretrained network -----------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor() -> r3d.R3DFeatureExtractor:
    return r3d.R3DFeatureExtractor()


@pytest.mark.slow
def test_extractor_is_frozen_on_cpu_without_the_kinetics_classifier(extractor):
    assert not extractor.model.training
    assert all(not p.requires_grad for p in extractor.model.parameters())
    assert {p.device.type for p in extractor.model.parameters()} == {"cpu"}
    assert isinstance(extractor.model.fc, torch.nn.Identity), "the 400-way Kinetics layer must be removed"
    assert extractor.preprocessing["resize_size"] == 128 and extractor.preprocessing["crop_size"] == 112


@pytest.mark.slow
def test_clip_preprocessing_agrees_with_the_weights_transform(extractor):
    frames = smooth_clip()
    ours = extractor.preprocess_clip(frames)
    assert ours.shape == (3, 16, 112, 112) and ours.dtype == np.float32
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    expected = R3D_18_Weights.KINETICS400_V1.transforms()(torch.from_numpy(rgb).permute(0, 3, 1, 2)).numpy()
    difference = np.abs(ours - expected)
    assert difference.mean() < 0.02 and np.percentile(difference, 99) < 0.1


@pytest.mark.slow
def test_clip_features_are_deterministic_and_respond_to_augmentation(extractor):
    clips = [smooth_clip(seed=0), smooth_clip(seed=1)]
    features = extractor.extract_clips(clips)
    assert features.shape == (2, r3d.FEATURE_DIM) and features.dtype == np.float32
    np.testing.assert_array_equal(features, extractor.extract_clips(clips))
    augmented = extractor.extract_clips(clips, [ClipAugmentation(crop_u=0.0, flip=True, brightness=1.2)] * 2)
    assert not np.allclose(features, augmented, atol=1e-3)


@pytest.mark.slow
def test_extract_video_copes_with_a_video_shorter_than_one_clip(extractor, tmp_path):
    short = make_video(tmp_path / "short.avi", num_frames=3)
    features, timing = extractor.extract_video(short, clip_len=16, target_fps=15.0, num_clips=2)
    assert features.shape == (2, r3d.FEATURE_DIM)
    assert timing["indices"].shape == (2, 16) and timing["backbone_seconds"] > 0


@pytest.mark.slow
def test_full_model_equals_backbone_followed_by_head(extractor):
    torch.manual_seed(0)
    head = r3d.ClipHead(r3d.FEATURE_DIM, 4).eval()
    clip = smooth_clip()
    full = r3d.build_full_model(head)
    assert full.fc is head and not full.training
    with torch.inference_mode():
        direct = full(torch.from_numpy(extractor.preprocess_clip(clip))[None]).numpy()
    np.testing.assert_allclose(direct, r3d.video_logits(head, extractor.extract_clips([clip])[None]), atol=1e-4)


@pytest.mark.slow
def test_dataset_features_are_cached_and_only_training_videos_are_augmented(tiny_config):
    dataset.prepare_dataset(tiny_config)
    records = dataset.load_manifest(tiny_config)
    with pytest.raises(FileNotFoundError, match="extract-r3d"):
        r3d.load_r3d_features(tiny_config, records)

    summary = r3d.extract_r3d_features(tiny_config)
    train = dataset.records_for_split(records, "train")
    assert summary["extracted_now"] == {"clean": len(records), "augmented": len(train)}
    assert r3d.extract_r3d_features(tiny_config)["extracted_now"] == {"clean": 0, "augmented": 0}

    features, labels = r3d.load_r3d_features(tiny_config, records)
    assert features.shape == (len(records), tiny_config.r3d_clips_per_video, r3d.FEATURE_DIM)
    assert sorted(set(labels.tolist())) == [0, 1]
    augmented, augmented_labels = r3d.load_r3d_features(tiny_config, train, augmented=True)
    assert augmented.shape == (len(train), tiny_config.augment_copies, r3d.FEATURE_DIM)
    np.testing.assert_array_equal(augmented_labels, r3d.load_r3d_features(tiny_config, train)[1])
    with pytest.raises(FileNotFoundError):
        r3d.load_r3d_features(tiny_config, dataset.records_for_split(records, "test"), augmented=True)

    result = r3d.train_r3d(tiny_config)
    assert tiny_config.r3d_checkpoint_path.is_file()
    assert {candidate["augmented"] for candidate in result["candidates"]} == {False, True}
    head, checkpoint = r3d.load_r3d_head(tiny_config)
    assert checkpoint["classes"] == list(tiny_config.classes)
    assert checkpoint["feature_key"] == tiny_config.r3d_feature_key
    assert r3d.video_logits(head, features).shape == (len(records), 2)
