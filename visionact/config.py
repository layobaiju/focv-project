"""Central configuration for VisionAct.

Every module reads its paths and hyperparameters from :class:`Config`, so an
experiment is described by one object that can be saved next to its results.

Override order (later wins):
1. the defaults below,
2. ``config.json`` in the project root (or the file named by ``VISIONACT_CONFIG``),
3. the ``VISIONACT_DATASET_DIR`` / ``VISIONACT_ARTIFACTS_DIR`` / ``VISIONACT_RESULTS_DIR``
   environment variables,
4. keyword overrides passed to :meth:`Config.load` (the CLI flags).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Four body-weight exercise classes that exist in UCF101 (verified against the
# dataset listing: 437 clips, 25 groups per class).
DEFAULT_CLASSES = ("BodyWeightSquats", "JumpingJack", "PullUps", "PushUps")

# The project targets CPU-only hardware. Nothing in the code selects CUDA.
DEVICE = "cpu"

CNN_NAME = "mobilenet_v3_small"
CNN_WEIGHTS = "IMAGENET1K_V1"

SPLIT_MODES = ("ucf101_split1", "random_group")

_ENV_PATHS = {
    "dataset_dir": "VISIONACT_DATASET_DIR",
    "artifacts_dir": "VISIONACT_ARTIFACTS_DIR",
    "results_dir": "VISIONACT_RESULTS_DIR",
}


@dataclass(frozen=True)
class Config:
    # --- reproducibility -------------------------------------------------
    seed: int = 42

    # --- data ------------------------------------------------------------
    classes: tuple[str, ...] = DEFAULT_CLASSES
    dataset_dir: Path = PROJECT_ROOT / "data" / "UCF101"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    results_dir: Path = PROJECT_ROOT / "results"

    # "ucf101_split1": test = the official split-1 test groups, validation =
    # ``num_val_groups`` groups drawn (seeded) from the remaining groups.
    # "random_group": seeded group-level split using the two fractions below,
    # for custom datasets without an official protocol.
    split_mode: str = "ucf101_split1"
    test_groups: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    num_val_groups: int = 4
    val_fraction: float = 0.15
    test_fraction: float = 0.25

    # --- frame sampling / CNN --------------------------------------------
    num_frames: int = 16  # frames sampled per video = LSTM sequence length

    # --- CNN + LSTM ------------------------------------------------------
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10  # early-stopping patience, in epochs

    # Small validation grid used by ``train-lstm --search``.
    search_hidden_dims: tuple[int, ...] = (64, 128, 256)
    search_dropouts: tuple[float, ...] = (0.3, 0.5)
    search_learning_rates: tuple[float, ...] = (1e-3, 3e-4)

    # --- CNN + SVM baseline ----------------------------------------------
    # Regularisation strengths tried on the validation split (ascending).
    svm_c_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.1, 1.0, 10.0)

    # --- training-only augmentation ----------------------------------------
    # Augmented copies cached per *training* video: random crop position,
    # temporal jitter, brightness change and (optionally) a mirror image.
    # Validation and test videos are never augmented. 0 disables augmentation.
    augment_copies: int = 4
    augment_brightness: float = 0.2  # brightness factor drawn from [1 - b, 1 + b]
    augment_hflip: bool = True  # mirror clips; valid only if a mirrored video keeps its label

    # --- R3D-18 pretrained on Kinetics-400 ---------------------------------
    # Clip length, frame rate and clips per video follow the settings the
    # published weights were evaluated with (16 frames at 15 fps, 5 clips).
    r3d_clip_len: int = 16
    r3d_target_fps: float = 15.0
    r3d_clips_per_video: int = 5
    r3d_head_learning_rate: float = 1e-3
    r3d_head_batch_size: int = 32
    r3d_head_epochs: int = 200
    r3d_head_patience: int = 20
    # Validation grid for the classification head.
    r3d_head_dropouts: tuple[float, ...] = (0.0, 0.5)
    r3d_head_weight_decays: tuple[float, ...] = (1e-4, 1e-2)

    # --- runtime ---------------------------------------------------------
    torch_threads: int = 0  # 0 keeps PyTorch's default thread count

    def __post_init__(self) -> None:
        if len(self.classes) < 2:
            raise ValueError("Config.classes needs at least two activity classes.")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"Config.classes contains duplicates: {self.classes}")
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {self.split_mode!r}.")
        positive = ("num_frames", "hidden_dim", "num_layers", "epochs", "batch_size", "patience",
                    "r3d_clip_len", "r3d_clips_per_video", "r3d_head_epochs", "r3d_head_patience")  # fmt: skip
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"Config.{name} must be at least 1, got {getattr(self, name)}.")
        if self.augment_copies < 0:
            raise ValueError(f"Config.augment_copies cannot be negative, got {self.augment_copies}.")
        if not 0.0 <= self.augment_brightness < 1.0:
            raise ValueError(f"Config.augment_brightness must be in [0, 1), got {self.augment_brightness}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"Config.dropout must be in [0, 1), got {self.dropout}.")
        if self.learning_rate <= 0:
            raise ValueError(f"Config.learning_rate must be positive, got {self.learning_rate}.")

    # --- derived locations -------------------------------------------------
    @property
    def feature_key(self) -> str:
        """Identifies a feature cache; changes whenever the features would."""
        return f"{CNN_NAME}-{CNN_WEIGHTS}-T{self.num_frames}"

    @property
    def features_dir(self) -> Path:
        return self.artifacts_dir / "features" / self.feature_key

    @property
    def augmentation_key(self) -> str:
        """Identifies one set of augmented training features (they depend on the seed)."""
        flip = "flip" if self.augment_hflip else "noflip"
        return f"aug{self.augment_copies}-b{self.augment_brightness:g}-{flip}-seed{self.seed}"

    @property
    def augmented_features_dir(self) -> Path:
        return self.artifacts_dir / "features" / f"{self.feature_key}-{self.augmentation_key}"

    @property
    def r3d_feature_key(self) -> str:
        return f"r3d_18-KINETICS400_V1-L{self.r3d_clip_len}-fps{self.r3d_target_fps:g}-K{self.r3d_clips_per_video}"

    @property
    def r3d_features_dir(self) -> Path:
        return self.artifacts_dir / "features" / self.r3d_feature_key

    @property
    def r3d_augmented_features_dir(self) -> Path:
        key = f"r3d_18-KINETICS400_V1-L{self.r3d_clip_len}-fps{self.r3d_target_fps:g}-{self.augmentation_key}"
        return self.artifacts_dir / "features" / key

    @property
    def r3d_checkpoint_path(self) -> Path:
        return self.models_dir / "r3d18_head_best.pt"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def splits_dir(self) -> Path:
        return self.artifacts_dir / "splits"

    @property
    def manifest_path(self) -> Path:
        return self.splits_dir / "manifest.csv"

    @property
    def split_config_path(self) -> Path:
        return self.splits_dir / "split_config.json"

    @property
    def lstm_checkpoint_path(self) -> Path:
        return self.models_dir / "cnn_lstm_best.pt"

    @property
    def baseline_model_path(self) -> Path:
        return self.models_dir / "cnn_svm_baseline.joblib"

    # --- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = list(value)
        return out

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with the given fields replaced (``None`` values are ignored)."""
        known = {f.name for f in fields(self)}
        clean = {}
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in known:
                raise KeyError(f"Unknown config field {key!r}. Known fields: {sorted(known)}")
            clean[key] = _coerce(self, key, value)
        return replace(self, **clean)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        config = cls()
        file_path = Path(path) if path else Path(os.environ.get("VISIONACT_CONFIG", PROJECT_ROOT / "config.json"))
        if file_path.is_file():
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Config file {file_path} is not valid JSON: {exc}") from exc
            config = config.with_overrides(**data)
        elif path:
            raise FileNotFoundError(f"Config file not found: {file_path}")
        env = {field: os.environ[var] for field, var in _ENV_PATHS.items() if os.environ.get(var)}
        return config.with_overrides(**env).with_overrides(**overrides)


def _coerce(config: Config, key: str, value: Any) -> Any:
    """Convert JSON/CLI values to the type of the field's default."""
    current = getattr(config, key)
    if isinstance(current, Path):
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if isinstance(current, tuple):
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        element_type = type(current[0]) if current else str
        return tuple(element_type(item) for item in value)
    if isinstance(current, bool):
        return bool(value)
    return type(current)(value)
