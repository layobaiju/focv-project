import numpy as np
import pytest

from visionact import optical_flow as of
from visionact import video


def shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(image, (dy, dx), axis=(0, 1))


@pytest.mark.parametrize("dx,dy", [(3, 0), (0, -2), (-2, 2)])
def test_dense_flow_recovers_a_known_translation(textured_image, dx, dy):
    flow = of.dense_flow(textured_image, shift(textured_image, dx, dy))
    assert flow.shape == (*textured_image.shape, 2) and flow.dtype == np.float32
    centre = flow[30:-30, 30:-30]  # ignore the wrapped-around borders
    assert np.median(centre[..., 0]) == pytest.approx(dx, abs=0.3)
    assert np.median(centre[..., 1]) == pytest.approx(dy, abs=0.3)


def test_dense_flow_is_zero_for_identical_frames(textured_image):
    stats = of.flow_statistics(of.dense_flow(textured_image, textured_image))
    assert stats["mean_magnitude"] < 0.05 and stats["moving_fraction"] == 0.0


def test_dense_flow_rejects_mismatched_sizes(textured_image):
    with pytest.raises(ValueError):
        of.dense_flow(textured_image, textured_image[:60])


def test_flow_statistics_report_direction(textured_image):
    stats = of.flow_statistics(of.dense_flow(textured_image, shift(textured_image, 3, 0)))
    assert stats["moving_fraction"] > 0.8
    assert stats["mean_dx"] == pytest.approx(3, abs=0.4) and abs(stats["mean_dy"]) < 0.4


def test_flow_visualisations(textured_image):
    flow = of.dense_flow(textured_image, shift(textured_image, 3, 0))
    colour = of.flow_to_color(flow)
    assert colour.shape == (*textured_image.shape, 3) and colour.dtype == np.uint8
    arrows = of.draw_flow_arrows(textured_image, flow)
    assert arrows.shape == (*textured_image.shape, 3)
    still = of.draw_flow_arrows(textured_image, np.zeros_like(flow))
    assert (arrows != still).any(), "arrows are drawn only where there is motion"
    assert not of.flow_to_color(np.zeros_like(flow)).any()


def test_sparse_flow_tracks_corners(textured_image):
    start, end = of.sparse_flow(textured_image, shift(textured_image, 3, 1))
    assert len(start) == len(end) > 10
    assert np.median(end - start, axis=0) == pytest.approx([3, 1], abs=0.3)
    assert of.draw_tracks(textured_image, start, end).shape == (*textured_image.shape, 3)


def test_sparse_flow_handles_a_featureless_frame():
    blank = np.zeros((60, 80), np.uint8)
    start, end = of.sparse_flow(blank, blank)
    assert start.shape == end.shape == (0, 2)


def test_flow_histogram_votes_for_the_motion_direction(textured_image):
    flow = of.dense_flow(textured_image, shift(textured_image, 3, 0))
    feature = of.flow_histogram(flow, bins=8, grid=(2, 2))
    assert feature.shape == (2 * 2 * 9,)
    first_cell = feature[:8]
    assert first_cell.sum() == pytest.approx(1.0, abs=1e-4)
    assert first_cell[[0, 7]].sum() > 0.9, "rightward motion falls in the bins around 0 degrees"
    assert not of.flow_histogram(np.zeros_like(flow)).any()


def test_video_flow_features_shape(video_path):
    frames = video.read_consecutive_frames(video_path, 0, 6)
    features = of.video_flow_features(frames)
    assert features.shape == (5, 36)
    assert np.isfinite(features).all()
    with pytest.raises(ValueError):
        of.video_flow_features(frames[:1])
