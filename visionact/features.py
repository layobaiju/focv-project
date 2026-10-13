"""MobileNetV3-Small as a frozen feature extractor, with an on-disk feature cache.

The CNN is pretrained on ImageNet and never trained here. Its final 1000-way
ImageNet layer is removed, so each frame becomes a 1024-dimensional vector that
describes its appearance. A video becomes an ordered ``(num_frames, 1024)``
array, which is cached as ``<features_dir>/<Class>/<clip>.npy`` so that training
experiments never have to run the CNN again.

Training videos can additionally be cached in augmented form (see
:mod:`visionact.augment`): ``(augment_copies, num_frames, 1024)`` per video.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .augment import ClipAugmentation, video_rng
from .config import CNN_NAME, CNN_WEIGHTS, Config
from .dataset import VideoRecord, load_manifest, records_for_split
from .utils import configure_torch_threads, get_device, get_logger, write_json
from .video import VideoError, preprocess_frames, sample_frames

log = get_logger(__name__)

FEATURE_DIM = 1024


class FeatureExtractor:
    """Frozen MobileNetV3-Small without its ImageNet classification layer."""

    def __init__(self) -> None:
        weights = MobileNet_V3_Small_Weights[CNN_WEIGHTS]
        try:
            model = mobilenet_v3_small(weights=weights)
        except Exception as exc:  # network errors surface as several exception types
            raise RuntimeError(
                "Could not load the pretrained MobileNetV3-Small weights. The first run downloads about "
                f"10 MB from download.pytorch.org into the torch cache; check the internet connection. ({exc})"
            ) from exc
        # classifier = [Linear(576, 1024), Hardswish, Dropout, Linear(1024, 1000)].
        # Keep the first two layers: the 1024-d representation that fed the ImageNet layer.
        model.classifier = nn.Sequential(*list(model.classifier.children())[:2])
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(get_device())
        # Input size and normalisation come from the weights themselves, not from constants.
        preset = weights.transforms()
        self.preprocessing = {
            "resize_size": int(preset.resize_size[0]),
            "crop_size": int(preset.crop_size[0]),
            "mean": tuple(float(v) for v in preset.mean),
            "std": tuple(float(v) for v in preset.std),
        }
        self.feature_dim = FEATURE_DIM

    @torch.inference_mode()
    def extract_frames(self, frames: Sequence[np.ndarray], augmentation: ClipAugmentation | None = None) -> np.ndarray:
        """BGR frames in temporal order -> ``(len(frames), 1024)`` features in the same order."""
        batch = preprocess_frames(frames, augmentation=augmentation, **self.preprocessing)
        return self.model(torch.from_numpy(batch).to(get_device())).numpy().astype(np.float32, copy=False)

    def extract_video(
        self,
        path: str | Path,
        num_frames: int,
        rng: np.random.Generator | None = None,
        augmentation: ClipAugmentation | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Sample, preprocess and encode one video. Also returns timings and the sampled indices.

        ``rng`` and ``augmentation`` are for training data only; inference leaves them unset.
        """
        start = time.perf_counter()
        frames, indices, _ = sample_frames(path, num_frames, rng)
        decoded = time.perf_counter()
        features = self.extract_frames(frames, augmentation)
        done = time.perf_counter()
        return features, {"decode_seconds": decoded - start, "backbone_seconds": done - decoded, "indices": indices}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def size_mb(self) -> float:
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getbuffer().nbytes / 1e6


# --- cache helpers (shared with the R3D-18 features) -----------------------------------------


def is_cached(path: Path, shape: tuple[int, ...]) -> bool:
    """True if ``path`` holds an array of exactly ``shape`` (a stale or truncated file is not trusted)."""
    if not path.is_file():
        return False
    try:
        return np.load(path, mmap_mode="r").shape == tuple(shape)
    except (OSError, ValueError):
        return False


def save_features(path: Path, features: np.ndarray) -> None:
    """Write atomically, so an interrupted run never leaves a half-written cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    with partial.open("wb") as handle:
        np.save(handle, features)
    partial.replace(path)


def load_cached(
    paths: list[Path], shape: tuple[int, ...], records: list[VideoRecord], config: Config, command: str
) -> tuple[np.ndarray, np.ndarray]:
    """Stack cached arrays for ``records`` and build the label vector (order of ``config.classes``)."""
    if not records:
        raise ValueError("No videos to load features for.")
    missing = [r.video_id for r, path in zip(records, paths) if not is_cached(path, shape)]
    if missing:
        raise FileNotFoundError(
            f"Cached features are missing for {len(missing)} video(s), e.g. {missing[0]} "
            f"(expected under {paths[0].parent.parent}). Run `python -m visionact {command}` first."
        )
    class_index = {label: index for index, label in enumerate(config.classes)}
    features = np.stack([np.load(path) for path in paths]).astype(np.float32)
    labels = np.array([class_index[record.label] for record in records], dtype=np.int64)
    return features, labels


# --- clean features (all videos) ---------------------------------------------------------------


def feature_path(config: Config, record: VideoRecord) -> Path:
    return config.features_dir / f"{record.video_id}.npy"


def extract_dataset_features(config: Config, force: bool = False) -> dict:
    """Extract features for every video in the manifest, skipping videos already cached."""
    records = load_manifest(config)
    configure_torch_threads(config.torch_threads)
    shape = (config.num_frames, FEATURE_DIM)
    todo = [r for r in records if force or not is_cached(feature_path(config, r), shape)]
    log.info("%d of %d videos already cached in %s", len(records) - len(todo), len(records), config.features_dir)
    failures: list[str] = []
    seconds = 0.0
    if todo:
        extractor = FeatureExtractor()
        started = time.perf_counter()
        for index, record in enumerate(todo, start=1):
            try:
                features, _ = extractor.extract_video(record.path, config.num_frames)
            except VideoError as exc:
                failures.append(f"{record.video_id}: {exc}")
                continue
            save_features(feature_path(config, record), features)
            if index % 50 == 0 or index == len(todo):
                elapsed = time.perf_counter() - started
                log.info("%d/%d videos (%.2f s per video)", index, len(todo), elapsed / index)
        seconds = time.perf_counter() - started
    if failures:
        raise RuntimeError(
            "Feature extraction failed for:\n  " + "\n  ".join(failures) + "\nRun `python -m visionact prepare` "
            "again to exclude unreadable videos."
        )
    summary = {
        "feature_key": config.feature_key,
        "cnn": CNN_NAME,
        "weights": CNN_WEIGHTS,
        "feature_dim": FEATURE_DIM,
        "num_frames": config.num_frames,
        "num_videos": len(records),
        "extracted_now": len(todo),
        "extraction_seconds": round(seconds, 2),
        "torch_version": torch.__version__,
        "device": str(get_device()),
    }
    if todo or not (config.features_dir / "meta.json").is_file():
        write_json(config.features_dir / "meta.json", summary)
    return summary


def load_features(config: Config, records: list[VideoRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Load cached features: ``X`` of shape (N, num_frames, 1024) and integer labels ``y``."""
    paths = [feature_path(config, record) for record in records]
    return load_cached(paths, (config.num_frames, FEATURE_DIM), records, config, "extract")


# --- augmented features (training videos only) ----------------------------------------------------


def augmented_feature_path(config: Config, record: VideoRecord) -> Path:
    return config.augmented_features_dir / f"{record.video_id}.npy"


def augmented_video_features(extractor: FeatureExtractor, config: Config, record: VideoRecord) -> np.ndarray:
    """``augment_copies`` augmented versions of one video: shape (copies, num_frames, 1024).

    Each copy uses jittered frame positions and its own crop position,
    brightness and (optionally) mirroring, drawn from a generator that depends
    only on the seed and the video.
    """
    rng = video_rng(config.seed, record.video_id)
    copies = []
    for _ in range(config.augment_copies):
        augmentation = ClipAugmentation.sample(rng, config.augment_hflip, config.augment_brightness)
        copies.append(extractor.extract_video(record.path, config.num_frames, rng, augmentation)[0])
    return np.stack(copies)


def extract_augmented_features(config: Config) -> dict:
    """Cache augmented features for the training split. Validation and test videos are left alone."""
    if config.augment_copies < 1:
        raise ValueError("Augmentation is disabled (augment_copies is 0); there is nothing to extract.")
    records = records_for_split(load_manifest(config), "train")
    configure_torch_threads(config.torch_threads)
    shape = (config.augment_copies, config.num_frames, FEATURE_DIM)
    todo = [r for r in records if not is_cached(augmented_feature_path(config, r), shape)]
    log.info("%d of %d training videos already augmented in %s", len(records) - len(todo), len(records),
             config.augmented_features_dir)  # fmt: skip
    seconds = 0.0
    if todo:
        extractor = FeatureExtractor()
        started = time.perf_counter()
        for index, record in enumerate(todo, start=1):
            save_features(augmented_feature_path(config, record), augmented_video_features(extractor, config, record))
            if index % 50 == 0 or index == len(todo):
                log.info("%d/%d training videos augmented", index, len(todo))
        seconds = time.perf_counter() - started
    summary = {
        "feature_key": config.feature_key,
        "augmentation": augmentation_settings(config),
        "training_videos": len(records),
        "extracted_now": len(todo),
        "extraction_seconds": round(seconds, 2),
    }
    if todo or not (config.augmented_features_dir / "meta.json").is_file():
        write_json(config.augmented_features_dir / "meta.json", summary)
    return summary


def load_augmented_features(config: Config, records: list[VideoRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Augmented training features: ``X`` of shape (N, augment_copies, num_frames, 1024) and labels."""
    paths = [augmented_feature_path(config, record) for record in records]
    shape = (config.augment_copies, config.num_frames, FEATURE_DIM)
    return load_cached(paths, shape, records, config, "extract-augmented")


def preprocessing_settings(config: Config) -> dict:
    """How a video is turned into MobileNetV3-Small input, for experiment records."""
    preset = MobileNet_V3_Small_Weights[CNN_WEIGHTS].transforms()
    return {
        "frames_per_video": config.num_frames,
        "frame_positions": "middle frame of equal segments spanning the video",
        "resize_shorter_side": int(preset.resize_size[0]),
        "crop": f"centre {preset.crop_size[0]} x {preset.crop_size[0]}",
        "normalisation_mean": [float(v) for v in preset.mean],
        "normalisation_std": [float(v) for v in preset.std],
    }


def augmentation_settings(config: Config) -> dict:
    """The augmentation applied to training data, for experiment records."""
    return {
        "copies_per_training_video": config.augment_copies,
        "random_crop_position": True,
        "temporal_jitter": True,
        "brightness_range": [1 - config.augment_brightness, 1 + config.augment_brightness],
        "horizontal_flip": config.augment_hflip,
        "seed": config.seed,
        "applied_to": "training split only",
    }
