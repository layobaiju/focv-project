import numpy as np
import pytest

from visionact import motion_experiment as me

VISUAL = {"val_accuracy": 69 / 75, "val_macro_f1": 0.920}


def fused(correct: int, macro_f1: float) -> dict:
    return {"val_accuracy": correct / 75, "val_macro_f1": macro_f1}


def test_flow_is_not_retained_without_a_validation_gain():
    same_accuracy = me.fusion_decision(VISUAL, fused(69, 0.923), val_videos=75, flow_ms_per_video=40.0)
    assert same_accuracy["retained"] is False and same_accuracy["validation_videos_gained"] == 0
    worse = me.fusion_decision(VISUAL, fused(66, 0.88), val_videos=75, flow_ms_per_video=40.0)
    assert worse["retained"] is False and worse["validation_videos_gained"] == -3


def test_a_single_extra_validation_video_is_treated_as_noise():
    decision = me.fusion_decision(VISUAL, fused(70, 0.934), val_videos=75, flow_ms_per_video=40.0)
    assert decision["retained"] is False and decision["validation_videos_gained"] == 1
    assert str(me.MIN_VIDEOS_GAINED) in decision["reason"]


def test_flow_is_retained_for_a_clear_and_affordable_gain():
    decision = me.fusion_decision(VISUAL, fused(72, 0.96), val_videos=75, flow_ms_per_video=40.0)
    assert decision["retained"] is True and decision["validation_videos_gained"] == 3


def test_flow_is_not_retained_when_it_is_too_slow_or_lowers_macro_f1():
    slow = me.fusion_decision(VISUAL, fused(72, 0.96), val_videos=75, flow_ms_per_video=5000.0)
    assert slow["retained"] is False and "slow" in slow["reason"]
    lower_f1 = me.fusion_decision(VISUAL, fused(72, 0.91), val_videos=75, flow_ms_per_video=40.0)
    assert lower_f1["retained"] is False


def test_motion_pooling_concatenates_mean_and_spread_over_time():
    descriptors = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    pooled = me.pool_motion(descriptors)
    assert pooled.shape == (2, 6)
    np.testing.assert_allclose(pooled[:, :3], descriptors.mean(axis=1))
    np.testing.assert_allclose(pooled[:, 3:], descriptors.std(axis=1))
    assert pytest.approx(pooled[0, 0]) == 4.5
