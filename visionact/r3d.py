"""R3D-18 pretrained on Kinetics-400, used as a frozen video feature extractor.

Unlike MobileNetV3-Small, which sees one frame at a time, R3D-18 applies 3D
convolutions to a short clip, so its features describe motion as well as
appearance. It was trained on Kinetics-400, a large collection of human action
videos, rather than on ImageNet photographs.

The backbone stays frozen. Its 400-way Kinetics layer is replaced by a new
linear head for the project's classes, and only that head is trained. Because
the backbone never changes, its 512-dimensional clip features are computed once
and cached; training the head then takes seconds on a CPU.

A video is represented by ``r3d_clips_per_video`` clips spread over its length.
The video-level prediction averages the clip logits.
"""

from __future__ import annotations

import io
import itertools
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from .augment import ClipAugmentation, video_rng
from .config import Config
from .dataset import VideoRecord, experiment_record, load_manifest, records_for_split
from .features import augmentation_settings, is_cached, load_cached, save_features
from .training import fit_with_early_stopping
from .utils import ModelNotTrainedError, configure_torch_threads, get_device, get_logger, set_seed, write_json
from .video import preprocess_frames, read_clips

log = get_logger(__name__)

MODEL_KEY = "r3d18_head"
DISPLAY_NAME = "R3D-18 (Kinetics-400, frozen) + linear head"
BACKBONE_NAME = "r3d_18"
WEIGHTS = R3D_18_Weights.KINETICS400_V1
WEIGHTS_SIZE_MB = 127
FEATURE_DIM = 512


# --- pretrained weights ----------------------------------------------------------------------


def weights_are_cached() -> bool:
    filename = WEIGHTS.url.rsplit("/", 1)[-1]
    return (Path(torch.hub.get_dir()) / "checkpoints" / filename).is_file()


def ensure_weights(assume_yes: bool = False) -> None:
    """Make sure the user has agreed to the weight download before it can happen."""
    if assume_yes or weights_are_cached():
        return
    explanation = (
        f"The pretrained R3D-18 weights are not on this computer. They are a one-time download of "
        f"{WEIGHTS_SIZE_MB} MB from {WEIGHTS.url} into the torch cache."
    )
    try:
        answer = input(f"{explanation}\nDownload now? [y/N] ")
    except (EOFError, OSError):  # not an interactive terminal
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise RuntimeError(f"{explanation} Re-run the command with --yes to allow the download.")


def _load_backbone() -> nn.Module:
    try:
        return r3d_18(weights=WEIGHTS)
    except Exception as exc:  # network errors surface as several exception types
        raise RuntimeError(
            f"Could not load the pretrained R3D-18 weights ({WEIGHTS_SIZE_MB} MB from download.pytorch.org). "
            f"Check the internet connection. ({exc})"
        ) from exc


# --- feature extractor -------------------------------------------------------------------------


class R3DFeatureExtractor:
    """Frozen R3D-18 without its Kinetics classification layer: one 512-d vector per clip."""

    def __init__(self) -> None:
        model = _load_backbone()
        model.fc = nn.Identity()
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(get_device())
        # Input size and normalisation come from the weights (128 x 171 resize, 112 x 112 crop).
        preset = WEIGHTS.transforms()
        self.preprocessing = {
            "resize_size": int(min(preset.resize_size)),
            "crop_size": int(preset.crop_size[0]),
            "mean": tuple(float(v) for v in preset.mean),
            "std": tuple(float(v) for v in preset.std),
        }
        self.feature_dim = FEATURE_DIM

    def preprocess_clip(self, frames: Sequence[np.ndarray], augmentation: ClipAugmentation | None = None) -> np.ndarray:
        """BGR frames in temporal order -> normalised array of shape (3, T, 112, 112)."""
        per_frame = preprocess_frames(frames, augmentation=augmentation, **self.preprocessing)
        return np.ascontiguousarray(per_frame.transpose(1, 0, 2, 3))

    @torch.inference_mode()
    def extract_clips(
        self, clips: Sequence[Sequence[np.ndarray]], augmentations: Sequence[ClipAugmentation | None] | None = None,
        batch_size: int = 4,
    ) -> np.ndarray:  # fmt: skip
        """Clips (each a list of BGR frames) -> ``(len(clips), 512)`` features."""
        augmentations = augmentations or [None] * len(clips)
        batch = torch.from_numpy(np.stack([self.preprocess_clip(c, a) for c, a in zip(clips, augmentations)]))
        outputs = [self.model(batch[i : i + batch_size].to(get_device())) for i in range(0, len(batch), batch_size)]
        return torch.cat(outputs).numpy().astype(np.float32, copy=False)

    def extract_video(
        self,
        path: str | Path,
        clip_len: int,
        target_fps: float,
        num_clips: int,
        rng: np.random.Generator | None = None,
        augmentations: Sequence[ClipAugmentation] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Decode ``num_clips`` clips and encode them. ``rng``/``augmentations`` are for training data only."""
        start = time.perf_counter()
        clips, indices, _ = read_clips(path, clip_len, target_fps, num_clips, rng)
        decoded = time.perf_counter()
        features = self.extract_clips(clips, augmentations)
        done = time.perf_counter()
        return features, {"decode_seconds": decoded - start, "backbone_seconds": done - decoded, "indices": indices}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def size_mb(self) -> float:
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getbuffer().nbytes / 1e6


# --- feature cache --------------------------------------------------------------------------------


def clip_feature_path(config: Config, record: VideoRecord, augmented: bool = False) -> Path:
    directory = config.r3d_augmented_features_dir if augmented else config.r3d_features_dir
    return directory / f"{record.video_id}.npy"


def preprocessing_settings(config: Config) -> dict:
    """How a video is turned into R3D-18 input, for experiment records."""
    preset = WEIGHTS.transforms()
    return {
        "clip_length_frames": config.r3d_clip_len,
        "target_frame_rate": config.r3d_target_fps,
        "clips_per_video": config.r3d_clips_per_video,
        "clip_positions": "evenly spaced over the video",
        "resize_shorter_side": int(min(preset.resize_size)),
        "crop": f"centre {preset.crop_size[0]} x {preset.crop_size[1]}",
        "normalisation_mean": [float(v) for v in preset.mean],
        "normalisation_std": [float(v) for v in preset.std],
        "video_prediction": "mean of the clip logits",
    }


def extract_r3d_features(config: Config, assume_yes: bool = False) -> dict:
    """Cache clip features for every video, plus augmented clips for the training split."""
    records = load_manifest(config)
    configure_torch_threads(config.torch_threads)
    clean_shape = (config.r3d_clips_per_video, FEATURE_DIM)
    augmented_shape = (config.augment_copies, FEATURE_DIM)
    clean_todo = [r for r in records if not is_cached(clip_feature_path(config, r), clean_shape)]
    augmented_todo = []
    if config.augment_copies > 0:
        augmented_todo = [
            r for r in records_for_split(records, "train")
            if not is_cached(clip_feature_path(config, r, augmented=True), augmented_shape)
        ]  # fmt: skip
    log.info("R3D-18 features to extract: %d videos (clean), %d training videos (augmented)",
             len(clean_todo), len(augmented_todo))  # fmt: skip

    seconds = 0.0
    if clean_todo or augmented_todo:
        ensure_weights(assume_yes)
        extractor = R3DFeatureExtractor()
        started = time.perf_counter()
        total = len(clean_todo) + len(augmented_todo)
        for index, record in enumerate(clean_todo, start=1):
            features, _ = extractor.extract_video(
                record.path, config.r3d_clip_len, config.r3d_target_fps, config.r3d_clips_per_video
            )
            save_features(clip_feature_path(config, record), features)
            _log_progress(index, total, started)
        for index, record in enumerate(augmented_todo, start=len(clean_todo) + 1):
            rng = video_rng(config.seed, record.video_id)
            augmentations = [
                ClipAugmentation.sample(rng, config.augment_hflip, config.augment_brightness)
                for _ in range(config.augment_copies)
            ]
            features, _ = extractor.extract_video(
                record.path, config.r3d_clip_len, config.r3d_target_fps, config.augment_copies, rng, augmentations
            )
            save_features(clip_feature_path(config, record, augmented=True), features)
            _log_progress(index, total, started)
        seconds = time.perf_counter() - started

    summary = {
        "feature_key": config.r3d_feature_key,
        "backbone": BACKBONE_NAME,
        "weights": "KINETICS400_V1",
        "feature_dim": FEATURE_DIM,
        "preprocessing": preprocessing_settings(config),
        "augmentation": augmentation_settings(config) if config.augment_copies > 0 else None,
        "num_videos": len(records),
        "extracted_now": {"clean": len(clean_todo), "augmented": len(augmented_todo)},
        "extraction_seconds": round(seconds, 2),
    }
    if clean_todo or augmented_todo or not (config.r3d_features_dir / "meta.json").is_file():
        write_json(config.r3d_features_dir / "meta.json", summary)
    return summary


def _log_progress(index: int, total: int, started: float) -> None:
    if index % 25 == 0 or index == total:
        elapsed = time.perf_counter() - started
        log.info("%d/%d videos (%.1f s per video, about %.0f min left)", index, total, elapsed / index,
                 (total - index) * elapsed / index / 60)  # fmt: skip


def load_r3d_features(config: Config, records: list[VideoRecord], augmented: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Cached clip features: ``X`` of shape (N, clips, 512) and integer labels ``y``."""
    paths = [clip_feature_path(config, record, augmented) for record in records]
    clips = config.augment_copies if augmented else config.r3d_clips_per_video
    return load_cached(paths, (clips, FEATURE_DIM), records, config, "extract-r3d")


# --- classification head -----------------------------------------------------------------------------


class ClipHead(nn.Module):
    """New classifier for R3D-18: standardise the clip feature, dropout, one linear layer."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)

    def set_normalisation(self, train_features: np.ndarray | torch.Tensor) -> None:
        flat = torch.as_tensor(train_features, dtype=torch.float32).reshape(-1, self.feature_mean.numel())
        self.feature_mean.copy_(flat.mean(dim=0))
        self.feature_std.copy_(flat.std(dim=0).clamp_min(1e-6))

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """(batch, 512) clip features -> (batch, num_classes) logits."""
        return self.linear(self.dropout((clip_features - self.feature_mean) / self.feature_std))


def _video_logits(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    videos, clips, dim = features.shape
    return head(features.reshape(videos * clips, dim)).reshape(videos, clips, -1).mean(dim=1)


@torch.inference_mode()
def video_logits(head: nn.Module, features: np.ndarray) -> np.ndarray:
    """(N, clips, 512) clip features -> (N, num_classes): the mean of each video's clip logits."""
    head.eval()
    return _video_logits(head, torch.as_tensor(features, dtype=torch.float32, device=get_device())).cpu().numpy()


def video_probabilities(head: nn.Module, features: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(video_logits(head, features)), dim=1).numpy()


def build_full_model(head: ClipHead) -> nn.Module:
    """The complete network: pretrained R3D-18 with its classifier replaced by the trained head."""
    model = _load_backbone()
    model.requires_grad_(False)
    model.fc = head
    return model.eval().to(get_device())


@dataclass(frozen=True)
class HeadHyperparameters:
    dropout: float
    weight_decay: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int


def fit_head(
    train_clips: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    hp: HeadHyperparameters,
    num_classes: int,
    seed: int,
) -> dict:
    """Train a head on clip features ``(M, D)``; validate on videos ``(N, clips, D)``."""
    set_seed(seed)
    head = ClipHead(train_clips.shape[1], num_classes, hp.dropout)
    head.set_normalisation(train_clips)
    val_tensor = torch.as_tensor(val_features, dtype=torch.float32, device=get_device())
    return fit_with_early_stopping(
        head,
        torch.as_tensor(train_clips, dtype=torch.float32),
        train_labels,
        lambda model: _video_logits(model, val_tensor),
        val_labels,
        num_classes=num_classes,
        learning_rate=hp.learning_rate,
        weight_decay=hp.weight_decay,
        batch_size=hp.batch_size,
        epochs=hp.epochs,
        patience=hp.patience,
        seed=seed,
    )


def train_r3d(config: Config) -> dict:
    """Train the head on the training split and select it on the validation split.

    Candidates: with and without the augmented training clips, for each dropout
    and weight-decay value in the config. The best validation macro-F1 wins
    (ties: lower validation loss). The test split is never read here.
    """
    configure_torch_threads(config.torch_threads)
    records = load_manifest(config)
    train_records, val_records = records_for_split(records, "train"), records_for_split(records, "val")
    x_train, y_train = load_r3d_features(config, train_records)
    x_val, y_val = load_r3d_features(config, val_records)
    num_classes = len(config.classes)

    clean_clips = x_train.reshape(-1, FEATURE_DIM)
    clean_labels = np.repeat(y_train, x_train.shape[1])
    training_sets = {False: (clean_clips, clean_labels)}
    if config.augment_copies > 0:
        x_augmented, _ = load_r3d_features(config, train_records, augmented=True)
        training_sets[True] = (
            np.concatenate([clean_clips, x_augmented.reshape(-1, FEATURE_DIM)]),
            np.concatenate([clean_labels, np.repeat(y_train, x_augmented.shape[1])]),
        )

    runs: list[dict] = []
    best: tuple[dict, dict] | None = None
    grid = itertools.product(training_sets, config.r3d_head_dropouts, config.r3d_head_weight_decays)
    for augmented, dropout, weight_decay in grid:
        hp = HeadHyperparameters(
            dropout, weight_decay, config.r3d_head_learning_rate, config.r3d_head_batch_size,
            config.r3d_head_epochs, config.r3d_head_patience,
        )  # fmt: skip
        clips, labels = training_sets[augmented]
        result = fit_head(clips, labels, x_val, y_val, hp, num_classes, config.seed)
        run = {
            "augmented": augmented,
            **asdict(hp),
            "training_clips": int(len(labels)),
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "val_loss": result["val_loss"],
            "val_accuracy": result["val_accuracy"],
            "val_macro_f1": result["val_macro_f1"],
            "training_seconds": round(result["training_seconds"], 3),
        }
        runs.append(run)
        log.info(
            "augmented=%-5s dropout=%.1f weight_decay=%-6g -> best epoch %3d/%3d, val loss %.4f, accuracy %.3f, macro-F1 %.3f",
            augmented, dropout, weight_decay, run["best_epoch"], run["epochs_run"], run["val_loss"],
            run["val_accuracy"], run["val_macro_f1"],
        )  # fmt: skip
        if best is None or (run["val_macro_f1"], -run["val_loss"]) > (best[0]["val_macro_f1"], -best[0]["val_loss"]):
            best = (run, result)

    selected, result = best
    head: ClipHead = result["model"]
    architecture = {"input_dim": FEATURE_DIM, "num_classes": num_classes, "dropout": selected["dropout"]}
    config.models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "architecture": architecture,
            "classes": list(config.classes),
            "feature_key": config.r3d_feature_key,
            "clip_len": config.r3d_clip_len,
            "target_fps": config.r3d_target_fps,
            "clips_per_video": config.r3d_clips_per_video,
            "trained_with_augmentation": selected["augmented"],
            "seed": config.seed,
            "best_epoch": selected["best_epoch"],
            "val_loss": selected["val_loss"],
            "val_accuracy": selected["val_accuracy"],
            "val_macro_f1": selected["val_macro_f1"],
        },
        config.r3d_checkpoint_path,
    )
    summary = {
        "model": DISPLAY_NAME,
        "selected": selected,
        "architecture": architecture,
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "frozen_backbone": f"{BACKBONE_NAME} with KINETICS400_V1 weights",
        "selection_metric": "validation macro-F1, ties broken by lower validation loss",
        "early_stopping": f"validation loss, patience {config.r3d_head_patience} epochs",
        "candidates": runs,
        "training_seconds": selected["training_seconds"],
        "search_seconds": round(sum(run["training_seconds"] for run in runs), 3),
        "history": result["history"],
        "train_videos": int(len(y_train)),
        "val_videos": int(len(y_val)),
        "seed": config.seed,
        "experiment": experiment_record(
            config,
            features=config.r3d_feature_key,
            preprocessing=preprocessing_settings(config),
            augmentation=augmentation_settings(config) if selected["augmented"] else None,
        ),
    }
    write_json(config.results_dir / MODEL_KEY / "training.json", summary)

    from .plots import plot_training_curves

    plot_training_curves(result["history"], selected["best_epoch"], config.results_dir / MODEL_KEY / "training_curves.png")
    log.info("Selected: augmented=%s, dropout=%.1f, weight decay=%g. Head saved to %s",
             selected["augmented"], selected["dropout"], selected["weight_decay"], config.r3d_checkpoint_path)  # fmt: skip
    return summary


def load_r3d_head(config: Config) -> tuple[ClipHead, dict]:
    """Load the trained head. Returns it in eval mode with the checkpoint metadata."""
    path = config.r3d_checkpoint_path
    if not path.is_file():
        raise ModelNotTrainedError(f"No trained R3D-18 head at {path}. Run `python -m visionact train-r3d` first.")
    checkpoint = torch.load(path, map_location=get_device(), weights_only=True)
    if checkpoint["classes"] != list(config.classes):
        raise ValueError(
            f"The R3D-18 head was trained for classes {checkpoint['classes']}, but the config lists "
            f"{list(config.classes)}. Retrain with `python -m visionact train-r3d`."
        )
    head = ClipHead(**checkpoint["architecture"]).to(get_device())
    head.load_state_dict(checkpoint.pop("state_dict"))
    head.eval()
    return head, checkpoint
