"""Tests that run the real pretrained CNN (marked slow: needs the cached or downloadable weights)."""

import cv2
import numpy as np
import pytest
import torch
from torchvision.models import MobileNet_V3_Small_Weights

from visionact import dataset, features, video

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def extractor() -> features.FeatureExtractor:
    return features.FeatureExtractor()


def random_frames(count: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (240, 320, 3), dtype=np.uint8) for _ in range(count)]


def test_extractor_is_frozen_cpu_and_has_no_imagenet_layer(extractor):
    assert not extractor.model.training
    assert all(not p.requires_grad for p in extractor.model.parameters())
    assert {p.device.type for p in extractor.model.parameters()} == {"cpu"}
    out_features = [m.out_features for m in extractor.model.classifier if isinstance(m, torch.nn.Linear)]
    assert out_features == [features.FEATURE_DIM], "the 1000-way ImageNet layer must be removed"


def test_features_have_expected_shape_and_are_deterministic(extractor):
    frames = random_frames(3)
    first = extractor.extract_frames(frames)
    assert first.shape == (3, features.FEATURE_DIM) and first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, extractor.extract_frames(frames))


def test_feature_order_follows_frame_order(extractor):
    frames = random_frames(3, seed=1)
    forward = extractor.extract_frames(frames)
    backward = extractor.extract_frames(frames[::-1])
    np.testing.assert_allclose(forward, backward[::-1], atol=1e-4)
    assert not np.allclose(forward[0], forward[2], atol=1e-3)


def test_opencv_preprocessing_matches_the_weights_transform(extractor):
    """On a 256x256 image no resizing happens, so our pipeline must equal torchvision's exactly."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    assert extractor.preprocessing["crop_size"] == 224 and extractor.preprocessing["resize_size"] == 256
    rgb = np.random.default_rng(2).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    expected = weights.transforms()(torch.from_numpy(rgb).permute(2, 0, 1)).numpy()
    ours = video.preprocess_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), **extractor.preprocessing)
    np.testing.assert_allclose(ours, expected, atol=1e-5)


def test_extract_video_handles_short_videos(extractor, tmp_path):
    from tests.conftest import make_video

    short = make_video(tmp_path / "short.avi", num_frames=3)
    feats, timing = extractor.extract_video(short, 8)
    assert feats.shape == (8, features.FEATURE_DIM)
    assert list(timing["indices"]) == sorted(timing["indices"])
    assert timing["backbone_seconds"] > 0 and timing["decode_seconds"] > 0


def test_augmented_features_exist_only_for_training_videos(tiny_config):
    dataset.prepare_dataset(tiny_config)
    records = dataset.load_manifest(tiny_config)
    train = dataset.records_for_split(records, "train")
    features.extract_dataset_features(tiny_config)
    with pytest.raises(FileNotFoundError, match="extract-augmented"):
        features.load_augmented_features(tiny_config, train)

    summary = features.extract_augmented_features(tiny_config)
    assert summary["extracted_now"] == len(train)
    assert features.extract_augmented_features(tiny_config)["extracted_now"] == 0, "second run must use the cache"

    augmented, labels = features.load_augmented_features(tiny_config, train)
    clean, clean_labels = features.load_features(tiny_config, train)
    assert augmented.shape == (len(train), tiny_config.augment_copies, tiny_config.num_frames, features.FEATURE_DIM)
    np.testing.assert_array_equal(labels, clean_labels)
    assert not np.allclose(augmented[:, 0], clean, atol=1e-3), "augmented clips must differ from the clean ones"
    # Validation and test videos are never augmented.
    assert not list(tiny_config.augmented_features_dir.rglob("*g01_c01.npy"))
    with pytest.raises(FileNotFoundError):
        features.load_augmented_features(tiny_config, dataset.records_for_split(records, "test"))


def test_augmentation_is_reproducible_for_a_seed(tiny_config, tmp_path):
    from dataclasses import replace

    dataset.prepare_dataset(tiny_config)
    train = dataset.records_for_split(dataset.load_manifest(tiny_config), "train")[:2]
    extractor_ = features.FeatureExtractor()
    first = features.augmented_video_features(extractor_, tiny_config, train[0])
    again = features.augmented_video_features(extractor_, tiny_config, train[0])
    other_seed = features.augmented_video_features(extractor_, replace(tiny_config, seed=1), train[0])
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other_seed, atol=1e-3)


def test_dataset_features_are_cached_and_reused(tiny_config):
    dataset.prepare_dataset(tiny_config)
    records = dataset.load_manifest(tiny_config)
    with pytest.raises(FileNotFoundError, match="visionact extract"):
        features.load_features(tiny_config, records)

    first = features.extract_dataset_features(tiny_config)
    assert first["extracted_now"] == 48 and (tiny_config.features_dir / "meta.json").is_file()
    assert features.extract_dataset_features(tiny_config)["extracted_now"] == 0, "second run must use the cache"

    x, y = features.load_features(tiny_config, records)
    assert x.shape == (48, tiny_config.num_frames, features.FEATURE_DIM)
    assert sorted(set(y.tolist())) == [0, 1]
    # Labels follow the order of config.classes.
    assert all(tiny_config.classes[label] == record.label for label, record in zip(y, records))

    # A cache entry with the wrong sequence length is not trusted.
    stale = features.feature_path(tiny_config, records[0])
    np.save(stale, np.zeros((2, features.FEATURE_DIM), np.float32))
    assert features.extract_dataset_features(tiny_config)["extracted_now"] == 1
