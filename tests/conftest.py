"""Shared fixtures. All test videos are generated on the fly; no dataset is needed."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def make_video(
    path: Path,
    num_frames: int = 30,
    size: tuple[int, int] = (160, 120),
    velocity: tuple[int, int] = (3, 0),
    seed: int = 0,
    colour: tuple[int, int, int] = (255, 255, 255),
    fourcc: str = "MJPG",
) -> Path:
    """Write a video (MJPG .avi by default) of a bright square moving over a fixed textured background."""
    width, height = size
    rng = np.random.default_rng(seed)
    background = cv2.GaussianBlur(rng.integers(0, 90, (height, width, 3), dtype=np.uint8), (9, 9), 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 25.0, (width, height))
    assert writer.isOpened(), "OpenCV could not create the test video"
    for index in range(num_frames):
        frame = background.copy()
        x = (20 + index * velocity[0]) % (width - 30)
        y = (40 + index * velocity[1]) % (height - 30)
        cv2.rectangle(frame, (x, y), (x + 24, y + 24), colour, -1)
        writer.write(frame)
    writer.release()
    return path


# Two synthetic "activities" that differ in both appearance and motion direction.
TINY_CLASSES = {"Clap": dict(colour=(0, 0, 255), velocity=(0, 3)), "Wave": dict(colour=(255, 255, 255), velocity=(3, 0))}


def make_ucf_tree(root: Path, groups: range = range(1, 13), clips: int = 2, num_frames: int = 8) -> Path:
    """Create ``root/<Class>/v_<Class>_gXX_cYY.avi`` for the two tiny classes."""
    for label, style in TINY_CLASSES.items():
        for group in groups:
            for clip in range(1, clips + 1):
                make_video(
                    root / label / f"v_{label}_g{group:02d}_c{clip:02d}.avi",
                    num_frames=num_frames,
                    seed=group * 10 + clip,
                    **style,
                )
    return root


def build_tiny_config(root: Path):
    """A config pointing at a small synthetic dataset created under ``root``."""
    from visionact.config import Config

    make_ucf_tree(root / "data")
    return Config(
        classes=tuple(TINY_CLASSES),
        dataset_dir=root / "data",
        artifacts_dir=root / "artifacts",
        results_dir=root / "results",
        num_frames=4,
        num_val_groups=2,
        hidden_dim=16,
        epochs=8,
        patience=4,
        batch_size=8,
        search_hidden_dims=(8, 16),
        search_dropouts=(0.1,),
        search_learning_rates=(1e-2,),
        svm_c_grid=(0.01, 1.0),
        augment_copies=1,
        r3d_clips_per_video=2,
        r3d_head_epochs=30,
        r3d_head_patience=5,
        r3d_head_dropouts=(0.0,),
        r3d_head_weight_decays=(1e-4,),
    )


@pytest.fixture
def tiny_config(tmp_path: Path):
    return build_tiny_config(tmp_path)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Run the whole pipeline once through the CLI, configured by a JSON file.

    Returns the config and the path of its JSON file. Shared by the pipeline
    and app tests; needs the pretrained CNN weights.
    """
    import json

    from visionact.cli import main

    root = tmp_path_factory.mktemp("pipeline")
    config = build_tiny_config(root)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    assert main(["all", "--config", str(config_path), "--timing-videos", "2"]) == 0
    return config, config_path


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    return make_video(tmp_path / "moving_square.avi")


@pytest.fixture
def textured_image() -> np.ndarray:
    """A smooth random grey texture; optical flow needs texture to lock on to."""
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 2.0)
