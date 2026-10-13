from dataclasses import replace

import pytest

from tests.conftest import make_video
from visionact import dataset
from visionact.dataset import VideoRecord


def test_parse_ucf101_name():
    assert dataset.parse_ucf101_name("v_PushUps_g08_c02") == ("PushUps", 8, 2)
    assert dataset.parse_ucf101_name("holiday_clip") is None


def test_scan_finds_all_videos(tiny_config):
    records, problems = dataset.scan_dataset(tiny_config.dataset_dir, tiny_config.classes)
    assert len(records) == 2 * 12 * 2 and not problems
    assert {record.label for record in records} == set(tiny_config.classes)
    assert records[0].video_id == "Clap/v_Clap_g01_c01" and records[0].group == "g01"


def test_scan_reports_missing_folders_with_guidance(tiny_config, tmp_path):
    with pytest.raises(FileNotFoundError, match="visionact download"):
        dataset.scan_dataset(tmp_path / "nowhere", tiny_config.classes)
    with pytest.raises(ValueError, match="Available class folders"):
        dataset.scan_dataset(tiny_config.dataset_dir, ("Clap", "Falling"))
    (tiny_config.dataset_dir / "Empty").mkdir()
    with pytest.raises(ValueError, match="no video files"):
        dataset.scan_dataset(tiny_config.dataset_dir, ("Clap", "Empty"))


def test_scan_excludes_files_whose_name_contradicts_the_folder(tiny_config):
    make_video(tiny_config.dataset_dir / "Clap" / "v_Wave_g01_c09.avi", num_frames=4)
    records, problems = dataset.scan_dataset(tiny_config.dataset_dir, tiny_config.classes)
    assert len(records) == 48 and len(problems) == 1
    assert "Wave" in problems[0]["reason"]


def test_validation_rejects_corrupt_videos(tiny_config):
    corrupt = tiny_config.dataset_dir / "Wave" / "v_Wave_g03_c09.avi"
    corrupt.write_bytes(b"\x00" * 2048)
    records, _ = dataset.scan_dataset(tiny_config.dataset_dir, tiny_config.classes)
    valid, problems = dataset.validate_records(records)
    assert len(valid) == 48 and [p["path"] for p in problems] == [str(corrupt)]
    assert all(record.num_frames == 8 for record in valid)


def test_ucf101_split_keeps_groups_together(tiny_config):
    records, _ = dataset.scan_dataset(tiny_config.dataset_dir, tiny_config.classes)
    assigned, description = dataset.split_records(records, tiny_config)
    for record in assigned:
        group = int(record.group[1:])
        assert (record.split == "test") == (group in tiny_config.test_groups)
    assert len(description["val_groups"]) == tiny_config.num_val_groups
    assert not set(description["val_groups"]) & set(description["test_groups"])
    by_group = {}
    for record in assigned:
        assert by_group.setdefault(record.group_key, record.split) == record.split
    counts = dataset.split_counts(assigned, tiny_config.classes)
    assert counts["test"] == {"Clap": 14, "Wave": 14}
    assert counts["val"] == {"Clap": 4, "Wave": 4}
    assert counts["train"] == {"Clap": 6, "Wave": 6}


def test_split_is_deterministic_and_seed_dependent(tiny_config):
    records, _ = dataset.scan_dataset(tiny_config.dataset_dir, tiny_config.classes)
    first = dataset.split_records(records, tiny_config)[1]["val_groups"]
    assert dataset.split_records(records, tiny_config)[1]["val_groups"] == first
    others = {tuple(dataset.split_records(records, replace(tiny_config, seed=s))[1]["val_groups"]) for s in range(1, 8)}
    assert len(others) > 1


def test_ucf101_split_needs_training_groups_left(tiny_config):
    records, _ = dataset.scan_dataset(tiny_config.dataset_dir, tiny_config.classes)
    with pytest.raises(ValueError, match="num_val_groups"):
        dataset.split_records(records, replace(tiny_config, num_val_groups=5))


def test_random_group_split_for_custom_names(tmp_path, tiny_config):
    records = [
        VideoRecord(f"{label}/{group}__{clip}", label, group, tmp_path / f"{group}__{clip}.mp4")
        for label in ("Clap", "Wave")
        for group in (f"person{i}" for i in range(10))
        for clip in range(3)
    ]
    assigned, _ = dataset.split_records(records, replace(tiny_config, split_mode="random_group"))
    dataset.assert_no_group_leakage(assigned)
    counts = dataset.split_counts(assigned, ("Clap", "Wave"))
    assert all(counts[split][label] > 0 for split in dataset.SPLITS for label in ("Clap", "Wave"))
    assert counts["test"]["Clap"] == 3 * 2 and counts["val"]["Clap"] == 3 * 2  # 25% and 15% of 10 groups, rounded
    with pytest.raises(ValueError, match="at least 3"):
        dataset.split_records(records[:6], replace(tiny_config, split_mode="random_group"))


def test_ucf101_mode_rejects_custom_file_names(tmp_path, tiny_config):
    records = [VideoRecord("Clap/holiday", "Clap", "holiday", tmp_path / "holiday.mp4")]
    with pytest.raises(ValueError, match="random_group"):
        dataset.split_records(records, tiny_config)


def test_leakage_check_detects_a_group_in_two_splits(tmp_path):
    records = [
        VideoRecord("A/v_A_g01_c01", "A", "g01", tmp_path / "a.avi", split="train"),
        VideoRecord("A/v_A_g01_c02", "A", "g01", tmp_path / "b.avi", split="test"),
    ]
    with pytest.raises(ValueError, match="leakage"):
        dataset.assert_no_group_leakage(records)
    # The same group number in another class is a different source video.
    dataset.assert_no_group_leakage([records[0], VideoRecord("B/v_B_g01_c01", "B", "g01", tmp_path / "c.avi", split="test")])


def test_prepare_writes_a_manifest_that_loads_back(tiny_config):
    summary = dataset.prepare_dataset(tiny_config)
    assert summary["totals"] == {"train": 12, "val": 8, "test": 28}
    assert summary["excluded_files"] == [] and summary["seed"] == tiny_config.seed
    records = dataset.load_manifest(tiny_config)
    assert len(records) == 48 and all(record.path.is_file() for record in records)
    assert len(dataset.records_for_split(records, "test")) == 28
    assert dataset.load_split_summary(tiny_config)["split"]["test_groups"] == [1, 2, 3, 4, 5, 6, 7]
    with pytest.raises(ValueError):
        dataset.records_for_split(records, "holdout")


def test_load_manifest_errors_are_actionable(tiny_config):
    with pytest.raises(FileNotFoundError, match="visionact prepare"):
        dataset.load_manifest(tiny_config)
    dataset.prepare_dataset(tiny_config)
    with pytest.raises(ValueError, match="prepare"):
        dataset.load_manifest(replace(tiny_config, classes=("Clap", "Jump")))


def test_experiment_record_pins_down_the_split_and_settings(tiny_config):
    dataset.prepare_dataset(tiny_config)
    record = dataset.experiment_record(tiny_config, features="some-key", preprocessing={"crop": 224})
    assert record["seed"] == tiny_config.seed and record["classes"] == list(tiny_config.classes)
    assert record["split"]["test_groups"] == [1, 2, 3, 4, 5, 6, 7]
    assert record["split_totals"] == {"train": 12, "val": 8, "test": 28}
    assert record["features"] == "some-key" and record["preprocessing"] == {"crop": 224}
    assert len(record["manifest_sha256"]) == 64
    assert dataset.experiment_record(tiny_config)["manifest_sha256"] == record["manifest_sha256"]
    # A different split is a different experiment.
    dataset.prepare_dataset(replace(tiny_config, num_val_groups=1))
    assert dataset.experiment_record(tiny_config)["manifest_sha256"] != record["manifest_sha256"]


def test_prepare_fails_when_a_class_has_no_test_videos(tiny_config):
    for path in (tiny_config.dataset_dir / "Wave").glob("v_Wave_g0[1-7]_*.avi"):
        path.unlink()
    with pytest.raises(ValueError, match="test split has no videos"):
        dataset.prepare_dataset(tiny_config)
