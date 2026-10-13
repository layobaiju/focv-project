"""Helpers shared by the Streamlit pages: video selection, caching and project status."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from ..config import Config
from ..dataset import load_manifest
from ..utils import read_json_if_exists
from ..video import (
    SUPPORTED_EXTENSIONS,
    VideoError,
    count_decodable_frames,
    read_consecutive_frames,
    read_frames_at,
    read_metadata,
    sample_frames,
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "visionact_uploads"
BROWSER_PLAYABLE = (".mp4", ".webm", ".m4v")
PIPELINE_COMMANDS = {
    "dataset": "python -m visionact download",
    "split": "python -m visionact prepare",
    "features": "python -m visionact extract",
    "baseline": "python -m visionact train-baseline",
    "lstm": "python -m visionact train-lstm --search",
    "r3d_features": "python -m visionact extract-r3d",
    "r3d": "python -m visionact train-r3d",
    "results": "python -m visionact evaluate",
}


@dataclass(frozen=True)
class SelectedVideo:
    path: Path
    name: str
    source: str  # "upload" or "dataset"
    label: str | None = None  # true class, known only for dataset clips
    split: str | None = None  # train / val / test, known only for dataset clips

    @property
    def cache_key(self) -> tuple[str, float]:
        return (str(self.path), self.path.stat().st_mtime)


@dataclass(frozen=True)
class AppContext:
    config: Config
    video: SelectedVideo | None


# --- project status ---------------------------------------------------------------


def project_status(config: Config) -> dict[str, bool]:
    """Which pipeline stages have produced their outputs."""
    return {
        "dataset": config.dataset_dir.is_dir() and any(config.dataset_dir.iterdir()),
        "split": config.manifest_path.is_file(),
        "features": (config.features_dir / "meta.json").is_file(),
        "baseline": config.baseline_model_path.is_file(),
        "lstm": config.lstm_checkpoint_path.is_file(),
        "r3d_features": (config.r3d_features_dir / "meta.json").is_file(),
        "r3d": config.r3d_checkpoint_path.is_file(),
        "results": (config.results_dir / "comparison.json").is_file(),
    }


def load_results(config: Config) -> dict | None:
    return read_json_if_exists(config.results_dir / "comparison.json")


def missing_steps_message(status: dict[str, bool], needed: list[str]) -> str:
    """Shell commands, in order, for the stages in ``needed`` that have not been run."""
    return "\n".join(PIPELINE_COMMANDS[step] for step in needed if not status[step])


# --- video selection ------------------------------------------------------------------


def _save_upload(uploaded) -> Path:
    """Write the upload to a temp file named by its content hash (re-runs reuse the file)."""
    data = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{hashlib.sha256(data).hexdigest()[:16]}{suffix}"
    if not path.is_file():
        path.write_bytes(data)
    return path


@st.cache_data(show_spinner=False)
def _dataset_clips(manifest_path: str, manifest_mtime: float, _config: Config) -> list[dict]:
    order = {"test": 0, "val": 1, "train": 2}  # held-out clips first: they are the fair ones to demo
    records = sorted(load_manifest(_config), key=lambda r: (order[r.split], r.video_id))
    return [{"path": str(r.path), "name": r.path.name, "label": r.label, "split": r.split} for r in records]


def select_video(config: Config) -> SelectedVideo | None:
    """Sidebar controls for choosing the input video. Returns ``None`` until one is chosen."""
    st.subheader("Input video")
    sources = ["Upload a video"]
    clips: list[dict] = []
    if config.manifest_path.is_file():
        try:
            clips = _dataset_clips(str(config.manifest_path), config.manifest_path.stat().st_mtime, config)
        except (ValueError, FileNotFoundError) as exc:
            st.caption(f"Dataset clips unavailable: {exc}")
        if clips:
            sources.append("Dataset clip")
    source = st.radio("Source", sources, key="video_source", label_visibility="collapsed")

    if source == "Upload a video":
        uploaded = st.file_uploader(
            "Video file", type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS], key="video_upload"
        )
        if uploaded is None:
            return None
        return SelectedVideo(_save_upload(uploaded), uploaded.name, "upload")

    labels = sorted({clip["label"] for clip in clips})
    label = st.selectbox("Class", labels, key="clip_class")
    options = [clip for clip in clips if clip["label"] == label]
    clip = st.selectbox("Clip", options, format_func=lambda c: f"{c['name']}  ({c['split']})", key="clip_name")
    path = Path(clip["path"])
    if not path.is_file():
        st.error(f"{clip['name']} is listed in the split but missing from the dataset folder.")
        return None
    return SelectedVideo(path, clip["name"], "dataset", clip["label"], clip["split"])


def require_video(ctx: AppContext) -> bool:
    """Show a hint and return False when no video has been selected yet."""
    if ctx.video is None:
        st.info("Choose an input video in the sidebar to use this section.")
        return False
    try:
        cached_metadata(*ctx.video.cache_key)
    except VideoError as exc:
        st.error(f"This video cannot be read: {exc}")
        return False
    return True


# --- cached video access ---------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=8)
def cached_metadata(path: str, mtime: float) -> dict:
    return read_metadata(path).to_dict()


@st.cache_data(show_spinner=False, max_entries=8)
def cached_frame_count(path: str, mtime: float) -> int:
    """Frame count from the header, or by decoding when the header does not give one."""
    return read_metadata(path).frame_count or count_decodable_frames(path)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_sample_frames(path: str, mtime: float, num_frames: int) -> tuple[list[np.ndarray], list[int]]:
    frames, indices, _ = sample_frames(path, num_frames)
    return frames, [int(i) for i in indices]


@st.cache_data(show_spinner=False, max_entries=16)
def cached_consecutive_frames(path: str, mtime: float, start: int, count: int) -> list[np.ndarray]:
    return read_consecutive_frames(path, start, count)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_frames_at(path: str, mtime: float, indices: tuple[int, ...]) -> list[np.ndarray]:
    """Frames at the given indices (which may repeat), in the order given."""
    unique = sorted(set(indices))
    by_index = dict(zip(unique, read_frames_at(path, unique)))
    return [by_index[i] for i in indices if i in by_index]


# --- display ---------------------------------------------------------------------------


def show_image(image: np.ndarray, caption: str | None = None) -> None:
    """Display an OpenCV image (BGR colour or single-channel) at the width of its container."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    st.image(rgb, caption=caption, width="stretch")


def odd_slider(label: str, minimum: int, maximum: int, value: int, key: str, help: str | None = None) -> int:
    """Slider restricted to odd values (kernel sizes)."""
    return st.slider(label, min_value=minimum, max_value=maximum, value=value, step=2, key=key, help=help)
