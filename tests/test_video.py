import numpy as np
import pytest

from tests.conftest import make_video
from visionact import video
from visionact.video import VideoError


@pytest.mark.parametrize("frame_count,num_frames", [(100, 16), (16, 16), (5, 16), (1, 8), (1000, 1)])
def test_sample_frame_indices_are_ordered_and_in_range(frame_count, num_frames):
    indices = video.sample_frame_indices(frame_count, num_frames)
    assert len(indices) == num_frames
    assert indices.min() >= 0 and indices.max() <= frame_count - 1
    assert np.all(np.diff(indices) >= 0), "temporal order must be preserved"


def test_sample_frame_indices_cover_the_whole_video():
    indices = video.sample_frame_indices(160, 16)
    assert np.all(np.diff(indices) == 10)
    assert indices[0] < 10 and indices[-1] >= 150


def test_sample_frame_indices_rejects_empty_video():
    with pytest.raises(VideoError):
        video.sample_frame_indices(0, 16)
    with pytest.raises(ValueError):
        video.sample_frame_indices(10, 0)


def test_read_metadata(video_path):
    meta = video.read_metadata(video_path)
    assert (meta.width, meta.height) == (160, 120)
    assert meta.frame_count == 30
    assert meta.fps == pytest.approx(25.0)
    assert meta.duration_seconds == pytest.approx(30 / 25.0)


def test_sample_frames_returns_requested_number(video_path):
    frames, indices, meta = video.sample_frames(video_path, 8)
    assert len(frames) == 8 and len(indices) == 8
    assert all(frame.shape == (120, 160, 3) and frame.dtype == np.uint8 for frame in frames)


def test_sample_frames_follow_the_motion_in_order(video_path):
    """The square moves right, so its column must increase across sampled frames."""
    frames, _, _ = video.sample_frames(video_path, 6)
    columns = [np.argwhere(frame[..., 0] > 200)[:, 1].mean() for frame in frames]
    assert all(b > a for a, b in zip(columns, columns[1:]))


def test_sample_frames_pads_a_video_with_too_few_frames(tmp_path):
    short = make_video(tmp_path / "short.avi", num_frames=5)
    frames, indices, _ = video.sample_frames(short, 16)
    assert len(frames) == 16
    assert indices.max() == 4 and np.all(np.diff(indices) >= 0)


def test_read_consecutive_frames_stops_at_end_of_video(video_path):
    assert len(video.read_consecutive_frames(video_path, 0, 4)) == 4
    assert len(video.read_consecutive_frames(video_path, 28, 10)) == 2


def test_read_frames_at_rejects_unordered_indices(video_path):
    with pytest.raises(ValueError):
        video.read_frames_at(video_path, [5, 2])


def test_missing_and_corrupt_files_raise_video_error(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        video.read_metadata(tmp_path / "missing.avi")
    corrupt = tmp_path / "corrupt.avi"
    corrupt.write_bytes(b"this is not a video" * 50)
    with pytest.raises(VideoError):
        video.sample_frames(corrupt, 8)


def test_resize_and_crop_handles_landscape_and_portrait():
    landscape = np.zeros((240, 320, 3), np.uint8)
    portrait = np.zeros((640, 360, 3), np.uint8)
    assert video.resize_shorter_side(landscape, 256).shape[:2] == (256, 341)
    assert video.resize_and_crop(landscape).shape == (224, 224, 3)
    assert video.resize_and_crop(portrait).shape == (224, 224, 3)


def test_preprocess_frame_normalises_and_converts_bgr_to_rgb():
    blue_bgr = np.zeros((240, 320, 3), np.uint8)
    blue_bgr[..., 0] = 255  # OpenCV stores blue first
    tensor = video.preprocess_frame(blue_bgr)
    assert tensor.shape == (3, 224, 224) and tensor.dtype == np.float32
    mean, std = np.array(video.IMAGENET_MEAN), np.array(video.IMAGENET_STD)
    expected = (np.array([0.0, 0.0, 1.0]) - mean) / std  # RGB order after conversion
    np.testing.assert_allclose(tensor.mean(axis=(1, 2)), expected, atol=1e-5)


def test_preprocess_frames_keeps_order():
    frames = [np.full((240, 320, 3), value, np.uint8) for value in (0, 100, 200)]
    batch = video.preprocess_frames(frames)
    assert batch.shape == (3, 3, 224, 224)
    means = batch.mean(axis=(1, 2, 3))
    assert means[0] < means[1] < means[2]
    with pytest.raises(ValueError):
        video.preprocess_frames([])
