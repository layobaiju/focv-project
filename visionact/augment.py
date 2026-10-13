"""Training-only augmentation of video clips.

One :class:`ClipAugmentation` is drawn per clip and applied to every frame of
it, so the clip stays temporally consistent. It replaces the centre crop used
at inference with a randomly positioned crop, and optionally mirrors the clip
and changes its brightness. Validation and test videos are never augmented.

Horizontal flipping is only valid when a mirrored video still shows the same
activity. That holds for the default classes (squats, jumping jacks, pull-ups
and push-ups are left-right symmetric). Disable it (``augment_hflip: false``)
for classes defined by direction, such as "turning left".
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from .video import resize_shorter_side


@dataclass(frozen=True)
class ClipAugmentation:
    """Parameters shared by all frames of one clip. The defaults reproduce the inference crop."""

    crop_u: float = 0.5  # horizontal crop position: 0 = left edge, 0.5 = centre, 1 = right edge
    crop_v: float = 0.5  # vertical crop position: 0 = top, 1 = bottom
    flip: bool = False
    brightness: float = 1.0  # multiplier applied to pixel values

    @classmethod
    def sample(cls, rng: np.random.Generator, hflip: bool = True, brightness: float = 0.2) -> "ClipAugmentation":
        """Draw a random crop position, a flip (if allowed) and a brightness factor in [1 - b, 1 + b]."""
        return cls(
            crop_u=float(rng.random()),
            crop_v=float(rng.random()),
            flip=bool(hflip and rng.random() < 0.5),
            brightness=float(rng.uniform(1.0 - brightness, 1.0 + brightness)),
        )

    def apply(self, frame_bgr: np.ndarray, resize_size: int, crop_size: int) -> np.ndarray:
        """Resize the shorter side to ``resize_size``, then crop, flip and adjust brightness."""
        resized = resize_shorter_side(frame_bgr, resize_size)
        height, width = resized.shape[:2]
        # round() matches video.center_crop, so the default position is exactly the inference crop.
        top, left = round(self.crop_v * (height - crop_size)), round(self.crop_u * (width - crop_size))
        crop = resized[top : top + crop_size, left : left + crop_size]
        if self.flip:
            crop = crop[:, ::-1]
        if self.brightness != 1.0:
            crop = np.clip(np.rint(crop.astype(np.float32) * self.brightness), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(crop)


def video_rng(seed: int, video_id: str, copy: int = 0) -> np.random.Generator:
    """Random generator tied to one video and copy number.

    The augmentation of a video then depends only on the seed and the video,
    not on the order in which videos happen to be processed.
    """
    return np.random.default_rng([seed, zlib.crc32(video_id.encode("utf-8")), copy])
