"""Image Processing: classical operators applied to one frame of the selected video."""

from __future__ import annotations

import numpy as np
import streamlit as st

from .. import image_processing as ip
from .common import AppContext, cached_sample_frames, odd_slider, require_video, show_image

_NUM_CHOICES = 16  # frames offered for inspection


def render(ctx: AppContext) -> None:
    st.header("Image Processing")
    st.caption(
        "Demonstration of classical operators on a single frame. These are not used by the activity classifier; "
        "they show the kind of filtering a CNN learns to do on its own."
    )
    if not require_video(ctx):
        return
    frames, indices = cached_sample_frames(*ctx.video.cache_key, _NUM_CHOICES)
    position = st.slider("Frame", 1, len(frames), len(frames) // 2, key="ip_frame") - 1
    frame = frames[position]
    gray = ip.to_gray(frame)
    st.caption(f"Working on frame {indices[position]} of {ctx.video.name}.")

    filtering, thresholding, morphology, edges, shapes = st.tabs(
        ["Filtering", "Thresholding", "Morphology", "Edge detection", "Shape analysis"]
    )
    with filtering:
        _filtering_tab(frame)
    with thresholding:
        mask = _thresholding_tab(gray)
    with morphology:
        cleaned = _morphology_tab(mask)
    with edges:
        _edges_tab(frame)
    with shapes:
        _shapes_tab(frame, cleaned)


def _filtering_tab(frame: np.ndarray) -> None:
    controls = st.columns(4)
    amount = controls[0].slider("Salt-and-pepper noise (%)", 0, 20, 5, key="ip_noise") / 100
    with controls[1]:
        gaussian_k = odd_slider("Gaussian kernel size", 3, 21, 5, "ip_gauss_k")
    with controls[2]:
        sigma = st.slider("Gaussian sigma (0 = automatic)", 0.0, 5.0, 0.0, 0.5, key="ip_sigma")
    with controls[3]:
        median_k = odd_slider("Median kernel size", 3, 21, 5, "ip_median_k")
    noisy = ip.add_salt_pepper_noise(frame, amount) if amount > 0 else frame

    images = st.columns(4)
    with images[0]:
        show_image(frame, "Original")
    with images[1]:
        show_image(noisy, f"Input to the filters ({amount:.0%} noise)")
    with images[2]:
        show_image(ip.gaussian_blur(noisy, gaussian_k, sigma), f"Gaussian filter {gaussian_k} x {gaussian_k}")
    with images[3]:
        show_image(ip.median_blur(noisy, median_k), f"Median filter {median_k} x {median_k}")
    st.markdown(
        "**Gaussian filtering** replaces each pixel by a weighted average of its neighbourhood. It reduces "
        "sensor-like noise but also blurs edges, and it only smears salt-and-pepper noise. **Median filtering** "
        "replaces each pixel by the neighbourhood median, so isolated black or white pixels are discarded while "
        "edges stay sharp. Raise the noise level to see the difference."
    )


def _thresholding_tab(gray: np.ndarray) -> np.ndarray:
    controls = st.columns(4)
    method = controls[0].selectbox("Method", ip.THRESHOLD_METHODS, index=1, key="ip_thresh_method")
    value = controls[1].slider("Threshold (binary method)", 0, 255, 127, key="ip_thresh_value", disabled=method != "binary")
    adaptive = method.startswith("adaptive")
    with controls[2]:
        block = odd_slider("Block size (adaptive methods)", 3, 51, 11, "ip_thresh_block")
    invert = controls[3].checkbox("Invert", key="ip_thresh_invert")
    mask, used = ip.threshold(gray, method, value, block, 2.0, invert)

    left, right = st.columns(2)
    with left:
        show_image(gray, "Greyscale frame")
    with right:
        caption = "Binary mask" + (f" (threshold = {used:.0f})" if used is not None else " (local thresholds)")
        show_image(mask, caption)
    histogram = np.bincount(gray.ravel(), minlength=256)
    st.bar_chart({"pixels": histogram}, height=160, x_label="grey level", y_label="pixels")
    if adaptive:
        st.markdown(
            "**Adaptive thresholding** compares each pixel with the mean (or Gaussian-weighted mean) of its own "
            "neighbourhood, so it copes with uneven lighting but responds to local texture."
        )
    elif method == "otsu":
        st.markdown(
            f"**Otsu's method** chose the threshold {used:.0f} automatically: it is the grey level that best "
            "separates the histogram above into two groups (smallest variance within each group)."
        )
    else:
        st.markdown("**Global thresholding**: pixels brighter than the chosen value become white, the rest black.")
    return mask


def _morphology_tab(mask: np.ndarray) -> np.ndarray:
    controls = st.columns(4)
    operation = controls[0].selectbox("Operation", list(ip.MORPH_OPERATIONS), index=2, key="ip_morph_op")
    with controls[1]:
        ksize = odd_slider("Structuring element size", 3, 21, 5, "ip_morph_k")
    shape = controls[2].selectbox("Shape", list(ip.KERNEL_SHAPES), index=1, key="ip_morph_shape")
    iterations = controls[3].slider("Iterations", 1, 5, 1, key="ip_morph_iter")
    result = ip.morphology(mask, operation, ksize, iterations, shape)

    left, right = st.columns(2)
    with left:
        show_image(mask, "Mask from the Thresholding tab")
    with right:
        show_image(result, f"After {operation}")
    changed = float((result != mask).mean())
    st.caption(f"{changed:.1%} of pixels changed.")
    st.markdown(
        "Morphological operations probe a binary image with a small structuring element. **Erosion** shrinks "
        "white regions and **dilation** grows them. **Opening** (erosion then dilation) removes small specks; "
        "**closing** (dilation then erosion) fills small holes; the **gradient** (dilation minus erosion) keeps "
        "only region boundaries."
    )
    return result


def _edges_tab(frame: np.ndarray) -> None:
    controls = st.columns(3)
    low, high = controls[0].slider("Hysteresis thresholds (low, high)", 0, 400, (50, 150), key="ip_canny")
    with controls[1]:
        blur = odd_slider("Gaussian pre-blur (1 = none)", 1, 15, 5, "ip_canny_blur")
    edge_map = ip.canny_edges(frame, low, high, blur)

    left, right = st.columns(2)
    with left:
        show_image(frame, "Original")
    with right:
        show_image(edge_map, f"Canny edges ({(edge_map > 0).mean():.1%} of pixels)")
    st.markdown(
        "**Canny edge detection** smooths the image, computes the intensity gradient, thins the response to "
        "one-pixel-wide ridges, and keeps weak edges only where they connect to strong ones (the two thresholds). "
        "Edges mark where brightness changes abruptly, which is where object outlines and body contours are, so an "
        "edge map keeps the *structure* of the scene while discarding colour and shading. The first layers of a CNN "
        "learn filters that respond to edges like these; deeper layers combine them into parts and objects."
    )


def _shapes_tab(frame: np.ndarray, mask: np.ndarray) -> None:
    min_area = st.slider("Ignore regions smaller than (pixels)", 10, 2000, 150, step=10, key="ip_min_area")
    contours, contour_regions = ip.find_contours(mask, min_area)
    labelled, regions = ip.connected_components(mask, min_area)

    columns = st.columns(3)
    with columns[0]:
        show_image(mask, "Mask from the Morphology tab")
    with columns[1]:
        show_image(labelled, f"Connected components: {len(regions)}")
    with columns[2]:
        show_image(ip.draw_contours(frame, contours, contour_regions), f"Outer contours and bounding boxes: {len(contours)}")
    if regions:
        st.dataframe(
            [
                {
                    "Label": region.label,
                    "Area (px)": int(region.area),
                    "Bounding box (x, y, w, h)": str(region.bbox),
                    "Centroid (x, y)": f"({region.centroid[0]:.0f}, {region.centroid[1]:.0f})",
                }
                for region in regions[:15]
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No region is larger than the minimum area. Lower the minimum or change the threshold settings.")
    st.markdown(
        "**Connected-component labelling** gives every group of touching white pixels its own label, from which "
        "area, bounding box and centroid follow. **Contour tracing** follows the boundary of each region. Both "
        "depend entirely on the mask, so on natural video they find bright regions rather than people. They are "
        "shown here as shape-analysis tools and are deliberately not fed to the classifier."
    )
