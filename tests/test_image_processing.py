import cv2
import numpy as np
import pytest

from visionact import image_processing as ip


@pytest.fixture
def two_squares() -> np.ndarray:
    """Black mask with a 40x40 and a 20x20 white square."""
    mask = np.zeros((120, 160), np.uint8)
    mask[20:60, 20:60] = 255
    mask[80:100, 110:130] = 255
    return mask


def test_gaussian_blur_reduces_noise_variance():
    noise = np.random.default_rng(0).integers(0, 256, (100, 100), dtype=np.uint8)
    assert ip.gaussian_blur(noise, 7).std() < noise.std() / 2


@pytest.mark.parametrize("ksize", [0, 4, -3])
def test_filters_reject_invalid_kernel_sizes(ksize):
    image = np.zeros((20, 20), np.uint8)
    with pytest.raises(ValueError):
        ip.gaussian_blur(image, ksize)
    with pytest.raises(ValueError):
        ip.median_blur(image, ksize)


def test_median_filter_removes_salt_and_pepper_noise():
    clean = np.full((100, 100), 120, np.uint8)
    noisy = ip.add_salt_pepper_noise(clean, amount=0.1, seed=0)
    assert 0.05 < (noisy != clean).mean() < 0.15
    restored = ip.median_blur(noisy, 3)
    assert (restored != clean).mean() < 0.01
    # A Gaussian blur only smears impulse noise, it does not remove it.
    assert np.abs(ip.gaussian_blur(noisy, 3).astype(int) - 120).mean() > np.abs(restored.astype(int) - 120).mean()


def test_otsu_threshold_separates_a_bimodal_image():
    image = np.full((50, 100), 40, np.uint8)
    image[:, 50:] = 200
    mask, used = ip.threshold(image, "otsu")
    assert 40 <= used < 200
    assert mask[:, :50].max() == 0 and mask[:, 50:].min() == 255


def test_threshold_methods_return_binary_masks():
    gradient = np.tile(np.arange(100, dtype=np.uint8) * 2, (60, 1))
    for method in ip.THRESHOLD_METHODS:
        mask, used = ip.threshold(gradient, method, value=100)
        assert set(np.unique(mask)) <= {0, 255}
        assert (used is None) == method.startswith("adaptive")
    inverted, _ = ip.threshold(gradient, "binary", value=100, invert=True)
    normal, _ = ip.threshold(gradient, "binary", value=100)
    assert np.array_equal(inverted, 255 - normal)
    with pytest.raises(ValueError):
        ip.threshold(gradient, "magic")


def test_opening_removes_specks_and_closing_fills_holes(two_squares):
    speckled = two_squares.copy()
    speckled[5, 150] = 255  # isolated foreground pixel
    assert ip.morphology(speckled, "open", 3)[5, 150] == 0
    holed = two_squares.copy()
    holed[40, 40] = 0  # one-pixel hole inside the big square
    assert ip.morphology(holed, "close", 3)[40, 40] == 255
    assert ip.morphology(two_squares, "dilate", 3).sum() > two_squares.sum() > ip.morphology(two_squares, "erode", 3).sum()
    with pytest.raises(ValueError):
        ip.morphology(two_squares, "shrink")


def test_canny_finds_the_boundary_but_not_the_interior(two_squares):
    edges = ip.canny_edges(two_squares, 50, 150, blur_ksize=3)
    assert edges[18:23, 30:50].any(), "top edge of the large square"
    assert not edges[30:50, 30:50].any(), "uniform interior has no edges"
    assert not edges[0:10, 0:10].any(), "uniform background has no edges"
    with pytest.raises(ValueError):
        ip.canny_edges(two_squares, 200, 100)


def test_find_contours_reports_regions_largest_first(two_squares):
    contours, regions = ip.find_contours(two_squares, min_area=10)
    assert len(contours) == len(regions) == 2
    assert regions[0].area > regions[1].area
    assert regions[0].bbox == (20, 20, 40, 40)
    assert regions[0].centroid == pytest.approx((39.5, 39.5), abs=0.6)
    assert len(ip.find_contours(two_squares, min_area=1000)[1]) == 1
    overlay = ip.draw_contours(cv2.cvtColor(two_squares, cv2.COLOR_GRAY2BGR), contours, regions)
    assert overlay.shape == (120, 160, 3)


def test_connected_components_labels_each_region(two_squares):
    coloured, regions = ip.connected_components(two_squares, min_area=10)
    assert coloured.shape == (120, 160, 3)
    assert [region.area for region in regions] == [1600.0, 400.0]
    assert not coloured[0, 0].any(), "background stays black"
    assert not np.array_equal(coloured[30, 30], coloured[90, 120]), "regions get different colours"
    assert len(ip.connected_components(two_squares, min_area=1000)[1]) == 1
