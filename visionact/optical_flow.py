"""Optical flow between consecutive frames.

Optical flow estimates, for each pixel (dense, Farneback) or each tracked corner
(sparse, Lucas-Kanade), the displacement ``(dx, dy)`` between two frames. It is
an explicit, hand-designed measurement of motion. The LSTM in this project
instead *learns* which changes between frame features matter for the label.

Convention: ``flow[y, x] = (dx, dy)`` in pixels, x to the right and y downwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .image_processing import to_gray


@dataclass(frozen=True)
class FarnebackParams:
    pyr_scale: float = 0.5  # image scale between pyramid levels
    levels: int = 3  # pyramid levels (more levels capture larger motion)
    winsize: int = 15  # averaging window (larger = smoother, less detailed)
    iterations: int = 3
    poly_n: int = 5  # neighbourhood for the polynomial expansion
    poly_sigma: float = 1.2


def dense_flow(previous: np.ndarray, current: np.ndarray, params: FarnebackParams = FarnebackParams()) -> np.ndarray:
    """Farneback dense optical flow. Returns a ``float32`` array of shape (H, W, 2)."""
    prev_gray, curr_gray = to_gray(previous), to_gray(current)
    if prev_gray.shape != curr_gray.shape:
        raise ValueError(f"Frames must have the same size, got {prev_gray.shape} and {curr_gray.shape}.")
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        params.pyr_scale,
        params.levels,
        params.winsize,
        params.iterations,
        params.poly_n,
        params.poly_sigma,
        0,
    )


def flow_to_color(flow: np.ndarray, max_magnitude: float | None = None) -> np.ndarray:
    """Colour-code a flow field: hue = direction, brightness = speed. Returns BGR ``uint8``."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    scale = max_magnitude if max_magnitude else float(magnitude.max())
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV hue range is 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / max(scale, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(
    frame: np.ndarray, flow: np.ndarray, step: int = 16, min_magnitude: float = 1.0, scale: float = 2.0
) -> np.ndarray:
    """Draw motion vectors on a grid of points; vectors shorter than ``min_magnitude`` are skipped."""
    canvas = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    height, width = flow.shape[:2]
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            dx, dy = flow[y, x]
            if dx * dx + dy * dy < min_magnitude * min_magnitude:
                continue
            tip = (int(round(x + dx * scale)), int(round(y + dy * scale)))
            cv2.arrowedLine(canvas, (x, y), tip, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.35)
    return canvas


def flow_statistics(flow: np.ndarray, moving_threshold: float = 1.0) -> dict[str, float]:
    """Summarise a flow field: speed, share of moving pixels and their mean displacement."""
    magnitude = np.linalg.norm(flow, axis=2)
    moving = magnitude >= moving_threshold
    mean_dx, mean_dy = (flow[moving].mean(axis=0) if moving.any() else (0.0, 0.0))
    return {
        "mean_magnitude": float(magnitude.mean()),
        "max_magnitude": float(magnitude.max()),
        "moving_fraction": float(moving.mean()),
        "mean_dx": float(mean_dx),
        "mean_dy": float(mean_dy),
    }


# --- sparse flow (Lucas-Kanade) ---------------------------------------------------


def sparse_flow(
    previous: np.ndarray, current: np.ndarray, max_corners: int = 150, quality: float = 0.01, min_distance: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Track Shi-Tomasi corners with pyramidal Lucas-Kanade.

    Returns matching ``(N, 2)`` arrays of point positions in the two frames
    (only points that were tracked successfully).
    """
    prev_gray, curr_gray = to_gray(previous), to_gray(current)
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=quality, minDistance=min_distance)
    if corners is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    return corners.reshape(-1, 2)[ok], tracked.reshape(-1, 2)[ok]


def draw_tracks(frame: np.ndarray, start: np.ndarray, end: np.ndarray, min_length: float = 0.5) -> np.ndarray:
    """Draw each tracked corner (red dot) and its displacement (yellow line)."""
    canvas = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for (x0, y0), (x1, y1) in zip(start, end):
        if np.hypot(x1 - x0, y1 - y0) >= min_length:
            cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(x1), int(y1)), 2, (0, 0, 255), -1, cv2.LINE_AA)
    return canvas


# --- motion descriptor for the optional experiment --------------------------------


def flow_histogram(flow: np.ndarray, bins: int = 8, grid: tuple[int, int] = (2, 2)) -> np.ndarray:
    """Histogram of optical flow (HOF) for one frame pair.

    The frame is divided into ``grid`` cells. Each cell contributes a
    magnitude-weighted histogram of flow directions (``bins`` values, L1
    normalised) plus its mean speed. Length: ``rows * cols * (bins + 1)``.
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    bin_index = np.minimum((angle / (2 * np.pi) * bins).astype(np.int64), bins - 1)
    rows, cols = grid
    height, width = flow.shape[:2]
    feature = []
    for row in range(rows):
        for col in range(cols):
            ys = slice(row * height // rows, (row + 1) * height // rows)
            xs = slice(col * width // cols, (col + 1) * width // cols)
            cell_magnitude = magnitude[ys, xs].ravel()
            histogram = np.bincount(bin_index[ys, xs].ravel(), weights=cell_magnitude, minlength=bins)
            total = histogram.sum()
            feature.extend(histogram / total if total > 1e-6 else histogram)
            feature.append(cell_magnitude.mean())
    return np.asarray(feature, dtype=np.float32)


def video_flow_features(
    frames: Sequence[np.ndarray], bins: int = 8, grid: tuple[int, int] = (2, 2), width: int = 160
) -> np.ndarray:
    """HOF descriptor for each consecutive frame pair: shape ``(len(frames) - 1, D)``.

    Frames are downscaled to ``width`` pixels first; flow at low resolution is
    much faster on CPU and is enough for a coarse motion descriptor.
    """
    if len(frames) < 2:
        raise ValueError("At least two frames are needed to compute optical flow.")
    small = []
    for frame in frames:
        gray = to_gray(frame)
        scale = width / gray.shape[1]
        small.append(cv2.resize(gray, (width, max(int(round(gray.shape[0] * scale)), 8)), interpolation=cv2.INTER_AREA))
    return np.stack([flow_histogram(dense_flow(a, b), bins, grid) for a, b in zip(small, small[1:])])
