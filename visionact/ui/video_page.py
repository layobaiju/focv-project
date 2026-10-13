"""Video Analysis: metadata, frame sampling, and the preprocessing applied before the CNN."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from ..video import IMAGENET_MEAN, IMAGENET_STD, bgr_to_rgb, normalize, resize_and_crop
from .common import BROWSER_PLAYABLE, AppContext, cached_metadata, cached_sample_frames, require_video, show_image

_FRAMES_PER_ROW = 8


def render(ctx: AppContext) -> None:
    st.header("Video Analysis")
    st.caption("Demonstration: frame extraction and sampling, resizing and normalisation. No model is involved.")
    if not require_video(ctx):
        return
    video = ctx.video
    meta = cached_metadata(*video.cache_key)

    st.subheader(video.name)
    columns = st.columns(5)
    columns[0].metric("Resolution", f"{meta['width']} x {meta['height']}")
    columns[1].metric("Frame rate", f"{meta['fps']:.2f} fps" if meta["fps"] else "unknown")
    columns[2].metric("Frames", meta["frame_count"] or "unknown")
    columns[3].metric("Duration", f"{meta['duration_seconds']:.2f} s" if meta["duration_seconds"] else "unknown")
    columns[4].metric("Codec", meta["codec"])
    if video.label:
        st.caption(f"Dataset clip. True class: {video.label}. Split: {video.split}.")

    if video.path.suffix.lower() in BROWSER_PLAYABLE:
        st.video(str(video.path), width=480)
    else:
        st.caption(
            f"Browsers cannot play {video.path.suffix} files directly, so no player is shown. "
            "The frames below are decoded with OpenCV."
        )

    st.subheader("Frame sampling")
    num_frames = st.slider(
        "Number of frames to sample", 4, 32, ctx.config.num_frames, key="video_num_frames",
        help=f"The classifier uses {ctx.config.num_frames} frames per video.",
    )  # fmt: skip
    frames, indices = cached_sample_frames(*video.cache_key, num_frames)
    st.markdown(
        "The video is divided into equal segments and the middle frame of each segment is taken, so the samples "
        "cover the whole clip in temporal order. A video with fewer frames than requested repeats frames."
    )
    if len(set(indices)) < len(indices):
        st.warning(f"This video has fewer than {num_frames} frames, so some frames are repeated.")
    for start in range(0, len(frames), _FRAMES_PER_ROW):
        row = st.columns(_FRAMES_PER_ROW)
        for column, frame, index in zip(row, frames[start:], indices[start:]):
            with column:
                show_image(frame, f"frame {index}")

    st.subheader("Preprocessing for the CNN")
    position = st.slider("Sampled frame to inspect", 1, len(frames), 1, key="video_inspect") - 1
    frame = frames[position]
    cropped = resize_and_crop(frame)
    normalised = normalize(bgr_to_rgb(cropped))
    # The normalised image has negative values; rescale it to 0..255 only so it can be displayed.
    display = cv2.normalize(normalised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    left, middle, right = st.columns(3)
    with left:
        show_image(frame, f"Original: {frame.shape[1]} x {frame.shape[0]}")
    with middle:
        show_image(cropped, "Shorter side resized to 256, centre crop 224 x 224")
    with right:
        st.image(display, caption="Normalised (values rescaled for display)", width="stretch")

    rgb = bgr_to_rgb(cropped).astype(np.float32) / 255.0
    st.dataframe(
        [
            {
                "Channel": name,
                "Mean before": round(float(rgb[..., i].mean()), 3),
                "Std before": round(float(rgb[..., i].std()), 3),
                "ImageNet mean": IMAGENET_MEAN[i],
                "ImageNet std": IMAGENET_STD[i],
                "Mean after": round(float(normalised[..., i].mean()), 3),
                "Std after": round(float(normalised[..., i].std()), 3),
            }
            for i, name in enumerate(("Red", "Green", "Blue"))
        ],
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        "Each pixel is scaled to the range 0 to 1, then the ImageNet channel mean is subtracted and the result is "
        "divided by the ImageNet channel standard deviation. MobileNetV3-Small was trained on images prepared this "
        "way, so new frames must be prepared identically for its features to be meaningful. OpenCV loads images as "
        "BGR, so the channels are also reordered to RGB."
    )
