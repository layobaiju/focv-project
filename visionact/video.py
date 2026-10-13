"""Video loading, frame sampling and CNN preprocessing with OpenCV.

Frames are BGR ``uint8`` arrays (OpenCV's native layout) until
:func:`preprocess_frame` converts them to the normalised RGB tensor layout the
pretrained CNN expects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm")

# Statistics of the ImageNet training set, used by the torchvision weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoError(RuntimeError):
    """The file is missing, is not a video OpenCV can open, or has no decodable frames."""


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int  # as reported by the container; may be approximate
    duration_seconds: float
    codec: str

    def to_dict(self) -> dict:
        return asdict(self)


def _open(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Video file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError(
            f"OpenCV could not open {path.name}. Supported containers: {', '.join(SUPPORTED_EXTENSIONS)}. "
            "The file may be corrupt or use an unsupported codec."
        )
    return capture


def read_metadata(path: str | Path) -> VideoMetadata:
    capture = _open(path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(str(path), width, height, fps, frame_count, duration, codec or "unknown")


def count_decodable_frames(path: str | Path) -> int:
    """Count frames by decoding, for containers whose header count is missing or wrong."""
    capture = _open(path)
    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    return count


def sample_frame_indices(frame_count: int, num_frames: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Pick ``num_frames`` indices spread evenly over the video, in temporal order.

    The video is cut into ``num_frames`` equal segments and the middle frame of
    each is taken. A video shorter than ``num_frames`` repeats frames, so the
    result always has the requested length and never decreases.

    With ``rng`` a random frame of each segment is taken instead of the middle
    one (temporal jitter, used only to augment training data).
    """
    if frame_count < 1:
        raise VideoError("Cannot sample frames from a video with no frames.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be at least 1, got {num_frames}.")
    offsets = rng.random(num_frames) if rng is not None else 0.5
    positions = (np.arange(num_frames) + offsets) * frame_count / num_frames
    return np.clip(np.floor(positions).astype(np.int64), 0, frame_count - 1)


def temporal_stride(fps: float, target_fps: float = 15.0) -> int:
    """Frame step that brings a video closest to ``target_fps`` (an unknown rate is treated as 30 fps)."""
    if fps <= 0:
        fps = 30.0
    return max(1, round(fps / target_fps))


def clip_frame_indices(
    frame_count: int, clip_len: int, stride: int, num_clips: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Indices of ``num_clips`` clips of ``clip_len`` frames taken every ``stride`` frames.

    Returns an array of shape (num_clips, clip_len). Clips are spread evenly
    from the start to the end of the video (a single clip is centred); with
    ``rng`` their positions are random. A video shorter than one clip repeats
    its last frame.
    """
    if frame_count < 1:
        raise VideoError("Cannot sample clips from a video with no frames.")
    span = (clip_len - 1) * stride + 1
    last_start = max(frame_count - span, 0)
    if rng is not None:
        starts = np.sort(rng.integers(0, last_start + 1, size=num_clips))
    elif num_clips == 1:
        starts = np.array([last_start // 2])
    else:
        starts = np.rint(np.linspace(0, last_start, num_clips)).astype(np.int64)
    indices = starts[:, None] + np.arange(clip_len)[None, :] * stride
    return np.minimum(indices, frame_count - 1).astype(np.int64)


def read_frames_at(path: str | Path, indices: Sequence[int]) -> list[np.ndarray]:
    """Decode the frames at the given non-decreasing indices.

    Decoding is sequential because seeking is not frame-accurate for many
    codecs. If the video ends early the returned list is shorter than ``indices``.
    """
    wanted = [int(i) for i in indices]
    if any(b < a for a, b in zip(wanted, wanted[1:])):
        raise ValueError("Frame indices must be in non-decreasing (temporal) order.")
    capture = _open(path)
    frames: list[np.ndarray] = []
    position, cursor = 0, 0
    try:
        while cursor < len(wanted) and capture.grab():
            if position == wanted[cursor]:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    break
                while cursor < len(wanted) and wanted[cursor] == position:
                    frames.append(frame)
                    cursor += 1
            position += 1
    finally:
        capture.release()
    return frames


def sample_frames(
    path: str | Path, num_frames: int, rng: np.random.Generator | None = None
) -> tuple[list[np.ndarray], np.ndarray, VideoMetadata]:
    """Return ``num_frames`` evenly spaced BGR frames, their indices and the metadata.

    ``rng`` adds temporal jitter (see :func:`sample_frame_indices`); training augmentation only.
    """
    metadata = read_metadata(path)
    frame_count = metadata.frame_count or count_decodable_frames(path)
    if frame_count < 1:
        raise VideoError(f"{Path(path).name} contains no decodable frames.")
    indices = sample_frame_indices(frame_count, num_frames, rng)
    frames = read_frames_at(path, indices)
    if len(frames) < num_frames:
        # The header over-reported the length: recount by decoding and resample.
        frame_count = count_decodable_frames(path)
        if frame_count < 1:
            raise VideoError(f"{Path(path).name} contains no decodable frames.")
        indices = sample_frame_indices(frame_count, num_frames, rng)
        frames = read_frames_at(path, indices)
        if not frames:
            raise VideoError(f"{Path(path).name} contains no decodable frames.")
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames, indices, metadata


def read_consecutive_frames(path: str | Path, start: int, count: int) -> list[np.ndarray]:
    """Decode ``count`` consecutive frames from ``start`` (fewer if the video ends)."""
    return read_frames_at(path, range(max(start, 0), max(start, 0) + count))


def read_clips(
    path: str | Path,
    clip_len: int,
    target_fps: float,
    num_clips: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[list[np.ndarray]], np.ndarray, VideoMetadata]:
    """Decode ``num_clips`` clips of ``clip_len`` frames at roughly ``target_fps``.

    Returns the clips (lists of BGR frames in temporal order), their frame
    indices with shape (num_clips, clip_len), and the metadata.
    """
    metadata = read_metadata(path)
    stride = temporal_stride(metadata.fps, target_fps)
    frame_count = metadata.frame_count or count_decodable_frames(path)
    for attempt in range(2):
        if frame_count < 1:
            raise VideoError(f"{Path(path).name} contains no decodable frames.")
        indices = clip_frame_indices(frame_count, clip_len, stride, num_clips, rng)
        wanted = sorted({int(i) for i in indices.ravel()})
        frames = dict(zip(wanted, read_frames_at(path, wanted)))
        if len(frames) == len(wanted):
            return [[frames[int(i)] for i in clip] for clip in indices], indices, metadata
        if attempt == 0:
            # The header over-reported the length: recount by decoding and sample again.
            frame_count = count_decodable_frames(path)
    raise VideoError(f"{Path(path).name} could not be decoded consistently.")


# --- image transformations ---------------------------------------------------


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_shorter_side(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize so the shorter side equals ``size``, keeping the aspect ratio."""
    height, width = frame.shape[:2]
    scale = size / min(height, width)
    new_size = (max(round(width * scale), size), max(round(height * scale), size))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, new_size, interpolation=interpolation)


def center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height < size or width < size:
        raise ValueError(f"Cannot take a {size}x{size} crop from a {width}x{height} frame.")
    # Same rounding as torchvision's centre crop (round half to even), so that a
    # frame is cropped exactly as the pretrained weights' own transform would.
    top, left = round((height - size) / 2), round((width - size) / 2)
    return frame[top : top + size, left : left + size]


def resize_and_crop(frame: np.ndarray, resize_size: int = 256, crop_size: int = 224) -> np.ndarray:
    """Geometric part of the CNN preprocessing; returns a BGR ``uint8`` square."""
    return center_crop(resize_shorter_side(frame, resize_size), crop_size)


def normalize(
    rgb: np.ndarray, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD
) -> np.ndarray:
    """Scale ``uint8`` RGB to [0, 1], then standardise each channel."""
    scaled = rgb.astype(np.float32) / 255.0
    return (scaled - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)


def preprocess_frame(
    frame_bgr: np.ndarray,
    resize_size: int = 256,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    augmentation=None,
) -> np.ndarray:
    """BGR ``uint8`` frame -> normalised ``float32`` array of shape (3, crop, crop).

    ``augmentation`` (a :class:`visionact.augment.ClipAugmentation`) replaces the
    centre crop with its own crop, flip and brightness change. It is passed
    only when preparing training data.
    """
    if augmentation is None:
        cropped = resize_and_crop(frame_bgr, resize_size, crop_size)
    else:
        cropped = augmentation.apply(frame_bgr, resize_size, crop_size)
    return np.ascontiguousarray(normalize(bgr_to_rgb(cropped), mean, std).transpose(2, 0, 1))


def preprocess_frames(frames: Sequence[np.ndarray], **kwargs) -> np.ndarray:
    """Stack preprocessed frames into (T, 3, H, W), keeping their order."""
    if not frames:
        raise ValueError("No frames to preprocess.")
    return np.stack([preprocess_frame(frame, **kwargs) for frame in frames])
