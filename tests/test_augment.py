import numpy as np
import pytest

from visionact import video
from visionact.augment import ClipAugmentation, video_rng


@pytest.fixture
def frame() -> np.ndarray:
    """240 x 320 BGR frame with a horizontal brightness ramp, so mirrored and shifted crops differ."""
    ramp = np.tile(np.linspace(20, 200, 320, dtype=np.uint8), (240, 1))
    return np.dstack([ramp, ramp // 2, np.full_like(ramp, 90)])


@pytest.mark.parametrize("resize_size,crop_size", [(256, 224), (128, 112)])
def test_identity_augmentation_equals_the_inference_crop(frame, resize_size, crop_size):
    crop = ClipAugmentation().apply(frame, resize_size, crop_size)
    np.testing.assert_array_equal(crop, video.resize_and_crop(frame, resize_size, crop_size))


def test_centre_crop_uses_the_same_offsets_as_torchvision():
    """171 - 112 = 59 is odd; torchvision rounds the 29.5 offset to 30, and so must we."""
    columns = np.tile(np.arange(171, dtype=np.uint8), (128, 1))
    crop = video.center_crop(np.dstack([columns] * 3), 112)
    assert crop.shape == (112, 112, 3) and crop[0, 0, 0] == 30 and crop[0, -1, 0] == 141


def test_crop_position_moves_the_window_but_keeps_the_size(frame):
    left = ClipAugmentation(crop_u=0.0).apply(frame, 256, 224)
    right = ClipAugmentation(crop_u=1.0).apply(frame, 256, 224)
    assert left.shape == right.shape == (224, 224, 3)
    assert left.mean() < right.mean(), "the ramp is brighter on the right"
    resized = video.resize_shorter_side(frame, 256)
    np.testing.assert_array_equal(left, resized[16:240, :224])
    np.testing.assert_array_equal(right, resized[16:240, -224:])


def test_flip_mirrors_the_crop(frame):
    plain = ClipAugmentation().apply(frame, 256, 224)
    flipped = ClipAugmentation(flip=True).apply(frame, 256, 224)
    np.testing.assert_array_equal(flipped, plain[:, ::-1])


def test_brightness_scales_pixels_and_clips_at_white(frame):
    plain = ClipAugmentation().apply(frame, 256, 224).astype(float)
    darker = ClipAugmentation(brightness=0.8).apply(frame, 256, 224)
    brighter = ClipAugmentation(brightness=3.0).apply(frame, 256, 224)
    np.testing.assert_allclose(darker, plain * 0.8, atol=1.0)
    assert brighter.dtype == np.uint8 and brighter.max() == 255


def test_one_augmentation_treats_every_frame_of_a_clip_alike(frame):
    augmentation = ClipAugmentation.sample(np.random.default_rng(0))
    first, second = (augmentation.apply(frame, 128, 112) for _ in range(2))
    np.testing.assert_array_equal(first, second)


def test_sampled_parameters_stay_in_their_ranges():
    rng = np.random.default_rng(1)
    samples = [ClipAugmentation.sample(rng, brightness=0.2) for _ in range(300)]
    assert all(0.0 <= s.crop_u <= 1.0 and 0.0 <= s.crop_v <= 1.0 for s in samples)
    assert all(0.8 <= s.brightness <= 1.2 for s in samples)
    assert 0.3 < np.mean([s.flip for s in samples]) < 0.7
    assert len({s.crop_u for s in samples}) > 250, "crops should vary"


def test_horizontal_flip_can_be_switched_off():
    rng = np.random.default_rng(2)
    assert not any(ClipAugmentation.sample(rng, hflip=False).flip for _ in range(200))


def test_video_rng_is_reproducible_and_specific_to_the_video():
    def draw(*args):
        return video_rng(*args).random(4).tolist()

    assert draw(42, "PushUps/v_PushUps_g08_c01") == draw(42, "PushUps/v_PushUps_g08_c01")
    assert draw(42, "PushUps/v_PushUps_g08_c01") != draw(42, "PushUps/v_PushUps_g08_c02")
    assert draw(42, "PushUps/v_PushUps_g08_c01") != draw(43, "PushUps/v_PushUps_g08_c01")


# --- temporal jitter and clip sampling ---------------------------------------------------


def test_frame_indices_without_rng_are_unchanged_segment_centres():
    np.testing.assert_array_equal(video.sample_frame_indices(160, 16), np.arange(5, 160, 10))


def test_jittered_frame_indices_stay_in_their_segments_and_in_order():
    rng = np.random.default_rng(0)
    for _ in range(50):
        indices = video.sample_frame_indices(160, 16, rng)
        assert np.all(indices // 10 == np.arange(16)), "one frame from each of the 16 segments"
    assert len({tuple(video.sample_frame_indices(160, 16, rng)) for _ in range(20)}) > 15
    short = video.sample_frame_indices(5, 16, rng)
    assert len(short) == 16 and short.max() <= 4 and np.all(np.diff(short) >= 0)


@pytest.mark.parametrize("fps,expected", [(25.0, 2), (29.97, 2), (15.0, 1), (60.0, 4), (10.0, 1), (0.0, 2)])
def test_temporal_stride_targets_fifteen_frames_per_second(fps, expected):
    assert video.temporal_stride(fps, target_fps=15.0) == expected


def test_clip_indices_are_evenly_spread_consecutive_runs():
    clips = video.clip_frame_indices(frame_count=100, clip_len=16, stride=2, num_clips=3)
    assert clips.shape == (3, 16)
    assert np.all(np.diff(clips, axis=1) == 2)
    assert clips[0, 0] == 0 and clips[-1, -1] == 99, "clips span the whole video"
    assert clips[1, 0] == round((100 - 31) / 2)
    single = video.clip_frame_indices(100, 16, 2, 1)
    assert single[0, 0] == (100 - 31) // 2, "a single clip is centred"


def test_clip_indices_for_a_video_shorter_than_one_clip_repeat_the_last_frame():
    clips = video.clip_frame_indices(frame_count=10, clip_len=16, stride=2, num_clips=2)
    assert clips.shape == (2, 16) and clips.max() == 9 and clips.min() == 0
    assert np.all(np.diff(clips, axis=1) >= 0)
    with pytest.raises(video.VideoError):
        video.clip_frame_indices(0, 16, 2, 1)


def test_random_clip_positions_are_seeded_and_valid():
    first = video.clip_frame_indices(200, 16, 2, 4, np.random.default_rng(3))
    again = video.clip_frame_indices(200, 16, 2, 4, np.random.default_rng(3))
    np.testing.assert_array_equal(first, again)
    assert first.min() >= 0 and first.max() <= 199 and np.all(np.diff(first, axis=1) == 2)
    assert len({int(row[0]) for row in first}) > 1


def test_read_clips_returns_frames_for_each_clip(video_path, tmp_path):
    from tests.conftest import make_video

    clips, indices, meta = video.read_clips(video_path, clip_len=8, target_fps=15.0, num_clips=2)
    assert len(clips) == 2 and all(len(clip) == 8 for clip in clips)
    assert indices.shape == (2, 8) and meta.frame_count == 30
    assert clips[0][0].shape == (120, 160, 3)
    # The square moves right, so its position must advance within a clip.
    columns = [np.argwhere(f[..., 0] > 200)[:, 1].mean() for f in clips[0]]
    assert all(b > a for a, b in zip(columns, columns[1:]))

    short = make_video(tmp_path / "short.avi", num_frames=3)
    clips, indices, _ = video.read_clips(short, clip_len=8, target_fps=15.0, num_clips=2)
    assert all(len(clip) == 8 for clip in clips) and indices.max() == 2

    corrupt = tmp_path / "corrupt.avi"
    corrupt.write_bytes(b"junk" * 200)
    with pytest.raises(video.VideoError):
        video.read_clips(corrupt, 8, 15.0, 2)
