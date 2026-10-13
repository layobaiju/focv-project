"""Motion Analysis: dense and sparse optical flow between frames of the selected video."""

from __future__ import annotations

import numpy as np
import streamlit as st

from .. import optical_flow as of
from ..video import sample_frame_indices
from .common import AppContext, cached_consecutive_frames, cached_frame_count, require_video, show_image

_PROFILE_PAIRS = 24


@st.cache_data(show_spinner=False)
def _direction_legend(size: int = 121) -> np.ndarray:
    """Colour wheel: the colour used for motion in each direction, brighter towards the rim."""
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    dx, dy = np.meshgrid(axis, axis)
    flow = np.dstack([dx, dy])
    flow[np.hypot(dx, dy) > 1] = 0
    return of.flow_to_color(flow, max_magnitude=1.0)


@st.cache_data(show_spinner="Measuring motion along the video...", max_entries=8)
def _motion_profile(path: str, mtime: float, frame_count: int) -> tuple[list[int], list[float]]:
    """Mean flow magnitude for consecutive-frame pairs spread over the video."""
    starts = sorted({int(i) for i in sample_frame_indices(frame_count - 1, _PROFILE_PAIRS)})
    magnitudes = []
    for start in starts:
        pair = cached_consecutive_frames(path, mtime, start, 2)
        if len(pair) < 2:
            break
        magnitudes.append(of.flow_statistics(of.dense_flow(*pair))["mean_magnitude"])
    return starts[: len(magnitudes)], magnitudes


def render(ctx: AppContext) -> None:
    st.header("Motion Analysis")
    st.caption("Demonstration: optical flow measures how pixels move between two frames. No model is involved.")
    if not require_video(ctx):
        return
    video = ctx.video
    frame_count = cached_frame_count(*video.cache_key)
    if frame_count < 2:
        st.warning("Optical flow needs at least two frames, and this video has fewer.")
        return

    controls = st.columns(2)
    max_gap = min(5, frame_count - 1)
    gap = 1
    if max_gap > 1:
        gap = controls[1].slider("Frames between the pair", 1, max_gap, 1, key="flow_gap",
                                 help="A larger gap shows larger displacements.")  # fmt: skip
    last_start = frame_count - 1 - gap
    start = controls[0].slider("First frame of the pair", 0, last_start, last_start // 2, key="flow_start") if last_start > 0 else 0
    pair = cached_consecutive_frames(*video.cache_key, start, gap + 1)
    if len(pair) < gap + 1:
        st.warning("The video ended before the second frame of the pair. Choose an earlier frame.")
        return
    previous, current = pair[0], pair[-1]

    dense, sparse, profile = st.tabs(["Dense flow (Farneback)", "Sparse flow (Lucas-Kanade)", "Motion over time"])
    with dense:
        _dense_tab(previous, current, start, gap)
    with sparse:
        _sparse_tab(previous, current)
    with profile:
        _profile_tab(video, frame_count)

    st.subheader("Optical flow and the LSTM")
    st.markdown(
        """
Both deal with change over time, in different ways.

| | Optical flow | LSTM on CNN features |
|---|---|---|
| What it measures | Displacement of pixels between two frames | Whatever changes in the frame features help predict the label |
| How it is obtained | A fixed algorithm with no training | Weights learned from labelled videos |
| Time span | Two neighbouring frames | The whole sampled sequence |
| Output | A motion field that can be inspected directly | A hidden state that is not directly interpretable |

Optical flow is therefore a complementary signal: it describes *how things move*, while CNN features describe
*what is visible*. The optional motion experiment (see Model Comparison) tests whether adding flow features helps
on this dataset.
"""
    )


def _dense_tab(previous: np.ndarray, current: np.ndarray, start: int, gap: int) -> None:
    with st.expander("Farneback parameters"):
        controls = st.columns(3)
        winsize = controls[0].slider("Window size", 5, 35, 15, step=2, key="flow_winsize")
        levels = controls[1].slider("Pyramid levels", 1, 5, 3, key="flow_levels")
        step = controls[2].slider("Arrow spacing (px)", 8, 32, 16, step=4, key="flow_step")
    flow = of.dense_flow(previous, current, of.FarnebackParams(winsize=winsize, levels=levels))
    stats = of.flow_statistics(flow)

    metrics = st.columns(4)
    metrics[0].metric("Mean speed", f"{stats['mean_magnitude']:.2f} px")
    metrics[1].metric("Fastest pixel", f"{stats['max_magnitude']:.1f} px")
    metrics[2].metric("Pixels moving more than 1 px", f"{stats['moving_fraction']:.1%}")
    metrics[3].metric("Mean motion of moving pixels", f"({stats['mean_dx']:+.1f}, {stats['mean_dy']:+.1f}) px")

    images = st.columns(4)
    with images[0]:
        show_image(previous, f"Frame {start}")
    with images[1]:
        show_image(current, f"Frame {start + gap}")
    with images[2]:
        show_image(of.flow_to_color(flow), "Flow field: colour = direction, brightness = speed")
    with images[3]:
        show_image(of.draw_flow_arrows(current, flow, step=step), "Motion vectors (2x length)")
    legend, text = st.columns([1, 5])
    with legend:
        show_image(_direction_legend(), "Direction key")
    with text:
        st.markdown(
            "**Dense optical flow** estimates a displacement vector for every pixel, assuming a pixel keeps its "
            "brightness as it moves and that neighbouring pixels move together. In the colour image, the hue "
            "gives the direction of motion (see the key: a point's colour is the colour used for motion from the "
            "centre towards that point) and brightness gives speed, so a still background is black. Displacements "
            "are in pixels, with x to the right and y downwards. Camera motion appears as flow over the whole frame."
        )


def _sparse_tab(previous: np.ndarray, current: np.ndarray) -> None:
    max_corners = st.slider("Maximum corners to track", 20, 300, 150, step=10, key="flow_corners")
    start, end = of.sparse_flow(previous, current, max_corners=max_corners)
    left, right = st.columns(2)
    with left:
        show_image(of.draw_tracks(current, start, end), f"{len(start)} corners tracked")
    with right:
        if len(start):
            displacement = np.linalg.norm(end - start, axis=1)
            st.metric("Median displacement", f"{np.median(displacement):.2f} px")
            st.metric("Corners moving more than 1 px", f"{(displacement > 1).mean():.0%}")
        else:
            st.info("No trackable corners were found in this frame.")
        st.markdown(
            "**Sparse optical flow** first finds corners (Shi-Tomasi), points whose surroundings change in every "
            "direction and are therefore easy to recognise again, then follows each one into the next frame with "
            "the pyramidal Lucas-Kanade method. It is much cheaper than dense flow and is the basis of feature "
            "tracking. Red dots are the tracked positions; yellow lines show how far each corner moved."
        )


def _profile_tab(video, frame_count: int) -> None:
    starts, magnitudes = _motion_profile(*video.cache_key, frame_count)
    if len(magnitudes) < 2:
        st.info("The video is too short for a motion profile.")
        return
    st.line_chart(
        {"frame": starts, "mean flow magnitude (px)": magnitudes},
        x="frame", y="mean flow magnitude (px)", height=260,
    )  # fmt: skip
    st.markdown(
        f"Mean dense-flow speed between consecutive frames at {len(starts)} positions along the video. Repetitive "
        "activities such as jumping jacks or push-ups tend to produce a rhythmic curve; a static shot stays near "
        "zero. This curve is a hand-made temporal description of the video, the kind of information a temporal "
        "model has to discover from frame features by itself."
    )
