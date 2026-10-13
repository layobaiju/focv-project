"""Classical image processing: filtering, thresholding, morphology, edges, shapes.

These operations are demonstrated in the app's Image Processing section. They
are not part of the activity classifier: the CNN learns its own filters from
data, and these hand-designed operators show what such filters compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

THRESHOLD_METHODS = ("binary", "otsu", "adaptive_mean", "adaptive_gaussian")
MORPH_OPERATIONS = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
}
KERNEL_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


def _require_odd(ksize: int, name: str = "ksize", minimum: int = 1) -> int:
    if ksize < minimum or ksize % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= {minimum}, got {ksize}.")
    return int(ksize)


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --- noise and smoothing -------------------------------------------------------


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05, seed: int = 0) -> np.ndarray:
    """Set a fraction ``amount`` of pixels to black or white (to demonstrate median filtering)."""
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1], got {amount}.")
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    draw = rng.random(image.shape[:2])
    noisy[draw < amount / 2] = 0
    noisy[draw > 1 - amount / 2] = 255
    return noisy


def gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Weighted average with a Gaussian kernel. ``sigma=0`` derives sigma from ``ksize``."""
    ksize = _require_odd(ksize)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Replace each pixel by the median of its neighbourhood (removes impulse noise, keeps edges)."""
    return cv2.medianBlur(image, _require_odd(ksize))


# --- thresholding and morphology ----------------------------------------------


def threshold(
    image: np.ndarray,
    method: str = "otsu",
    value: int = 127,
    block_size: int = 11,
    c: float = 2.0,
    invert: bool = False,
) -> tuple[np.ndarray, float | None]:
    """Binarise an image. Returns the mask and the global threshold used (``None`` if adaptive)."""
    gray = to_gray(image)
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "binary":
        used, mask = cv2.threshold(gray, value, 255, kind)
        return mask, float(used)
    if method == "otsu":
        used, mask = cv2.threshold(gray, 0, 255, kind | cv2.THRESH_OTSU)
        return mask, float(used)
    if method in ("adaptive_mean", "adaptive_gaussian"):
        adaptive = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        mask = cv2.adaptiveThreshold(gray, 255, adaptive, kind, _require_odd(block_size, "block_size", 3), c)
        return mask, None
    raise ValueError(f"Unknown threshold method {method!r}. Choose one of {THRESHOLD_METHODS}.")


def morphology(
    mask: np.ndarray, operation: str = "open", ksize: int = 3, iterations: int = 1, shape: str = "ellipse"
) -> np.ndarray:
    """Apply erosion, dilation, opening, closing or the morphological gradient."""
    if operation not in MORPH_OPERATIONS:
        raise ValueError(f"Unknown morphology operation {operation!r}. Choose one of {tuple(MORPH_OPERATIONS)}.")
    if shape not in KERNEL_SHAPES:
        raise ValueError(f"Unknown kernel shape {shape!r}. Choose one of {tuple(KERNEL_SHAPES)}.")
    ksize = _require_odd(ksize)
    kernel = cv2.getStructuringElement(KERNEL_SHAPES[shape], (ksize, ksize))
    return cv2.morphologyEx(mask, MORPH_OPERATIONS[operation], kernel, iterations=max(int(iterations), 1))


# --- edges -----------------------------------------------------------------------


def canny_edges(image: np.ndarray, low: int = 50, high: int = 150, blur_ksize: int = 5) -> np.ndarray:
    """Canny edge map. A Gaussian pre-blur (``blur_ksize`` > 1) suppresses noise edges."""
    if low > high:
        raise ValueError(f"Canny low threshold ({low}) must not exceed the high threshold ({high}).")
    gray = to_gray(image)
    if blur_ksize > 1:
        gray = gaussian_blur(gray, blur_ksize)
    return cv2.Canny(gray, low, high)


# --- object and shape analysis ---------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One connected foreground region of a binary mask."""

    label: int
    area: float
    bbox: tuple[int, int, int, int]  # x, y, width, height
    centroid: tuple[float, float]
    perimeter: float | None = None


def find_contours(mask: np.ndarray, min_area: float = 50.0) -> tuple[list[np.ndarray], list[Region]]:
    """Trace the outer boundary of each foreground region, largest first."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = sorted((c for c in contours if cv2.contourArea(c) >= min_area), key=cv2.contourArea, reverse=True)
    regions = []
    for index, contour in enumerate(kept, start=1):
        moments = cv2.moments(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if moments["m00"] > 0:
            centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        else:
            centroid = (x + w / 2, y + h / 2)
        regions.append(Region(index, float(cv2.contourArea(contour)), (x, y, w, h), centroid, cv2.arcLength(contour, True)))
    return kept, regions


def draw_contours(image: np.ndarray, contours: list[np.ndarray], regions: list[Region]) -> np.ndarray:
    """Overlay contours (green) and bounding boxes (orange) on a copy of the image."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canvas, contours, -1, (0, 220, 0), 2)
    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 140, 255), 1)
    return canvas


def connected_components(mask: np.ndarray, min_area: int = 50) -> tuple[np.ndarray, list[Region]]:
    """Label 8-connected foreground regions. Returns a colour-coded label image and region statistics."""
    binary = (to_gray(mask) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    colours = np.zeros((count, 3), dtype=np.uint8)
    regions: list[Region] = []
    for label in range(1, count):  # label 0 is the background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # Spread hues with the golden angle so neighbouring labels get distinct colours.
        hue = int((label * 137.5) % 180)
        colours[label] = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        bbox = tuple(int(stats[label, key]) for key in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        regions.append(Region(label, float(area), bbox, (float(centroids[label][0]), float(centroids[label][1]))))
    regions.sort(key=lambda region: region.area, reverse=True)
    return colours[labels], regions
