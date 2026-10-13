"""Dataset scanning, validation and leakage-free splitting.

Expected layout (the official UCF101 layout, also used for custom data)::

    <dataset_dir>/
        JumpingJack/v_JumpingJack_g01_c01.avi
        PushUps/v_PushUps_g01_c01.avi
        ...

The unit of splitting is the **group**, not the clip. In UCF101 the clips
``v_<Class>_gXX_cYY`` with the same ``gXX`` were cut from the same source video
and share actor, background and viewpoint, so the dataset authors require that
a group never appears in both training and testing. For custom data, name files
``<group>__<anything>.mp4`` to mark clips that belong together; otherwise each
file is its own group.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .utils import get_logger, read_json, write_json
from .video import SUPPORTED_EXTENSIONS, VideoError, read_frames_at, read_metadata

log = get_logger(__name__)

SPLITS = ("train", "val", "test")
_UCF_NAME = re.compile(r"^v_(?P<label>[A-Za-z]+)_g(?P<group>\d+)_c(?P<clip>\d+)$")
_MANIFEST_FIELDS = ("video_id", "label", "group", "split", "num_frames", "path")


@dataclass(frozen=True)
class VideoRecord:
    video_id: str  # "<label>/<file stem>", unique within the dataset
    label: str
    group: str  # clips sharing (label, group) must stay in one split
    path: Path
    split: str = ""
    num_frames: int = 0

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.label, self.group)


def parse_ucf101_name(stem: str) -> tuple[str, int, int] | None:
    """``v_PushUps_g08_c02`` -> ``("PushUps", 8, 2)``; ``None`` for other names."""
    match = _UCF_NAME.match(stem)
    return (match["label"], int(match["group"]), int(match["clip"])) if match else None


def _group_of(stem: str) -> str:
    parsed = parse_ucf101_name(stem)
    if parsed:
        return f"g{parsed[1]:02d}"
    return stem.split("__", 1)[0] if "__" in stem else stem


def scan_dataset(dataset_dir: Path, classes: tuple[str, ...]) -> tuple[list[VideoRecord], list[dict]]:
    """Find the videos of the requested classes. Returns records and labelling problems."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Download the default classes with `python -m visionact download`, or point "
            "--dataset-dir (or VISIONACT_DATASET_DIR) at a folder with one sub-folder of videos per class."
        )
    available = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    records: list[VideoRecord] = []
    problems: list[dict] = []
    for label in classes:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            raise ValueError(
                f"Class folder {label!r} does not exist in {dataset_dir}. "
                f"Available class folders: {available or 'none'}. Class names are case-sensitive."
            )
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            raise ValueError(f"Class folder {class_dir} contains no video files ({', '.join(SUPPORTED_EXTENSIONS)}).")
        for path in files:
            parsed = parse_ucf101_name(path.stem)
            if parsed and parsed[0] != label:
                problems.append({"path": str(path), "reason": f"file name says class {parsed[0]!r} but folder is {label!r}"})
                continue
            records.append(VideoRecord(f"{label}/{path.stem}", label, _group_of(path.stem), path))
    return records, problems


def validate_records(records: list[VideoRecord]) -> tuple[list[VideoRecord], list[dict]]:
    """Open every video and decode its first frame. Returns usable records and the rejected files."""
    valid: list[VideoRecord] = []
    problems: list[dict] = []
    for record in records:
        try:
            metadata = read_metadata(record.path)
            if not read_frames_at(record.path, [0]):
                raise VideoError("no decodable frames")
        except VideoError as exc:
            problems.append({"path": str(record.path), "reason": str(exc)})
            continue
        valid.append(replace(record, num_frames=metadata.frame_count))
    return valid, problems


def split_records(records: list[VideoRecord], config: Config) -> tuple[list[VideoRecord], dict]:
    """Assign each record to train/val/test at group level. Returns records and the split description."""
    if config.split_mode == "ucf101_split1":
        assignment, description = _split_ucf101(records, config)
    else:
        assignment, description = _split_random_groups(records, config)
    assigned = [replace(record, split=assignment[record.group_key]) for record in records]
    assert_no_group_leakage(assigned)
    return assigned, description


def _split_ucf101(records: list[VideoRecord], config: Config) -> tuple[dict, dict]:
    groups: set[int] = set()
    for record in records:
        parsed = parse_ucf101_name(record.path.stem)
        if not parsed:
            raise ValueError(
                f"{record.path.name} is not named like a UCF101 clip (v_<Class>_gXX_cYY). "
                'Set "split_mode": "random_group" in config.json for a custom dataset.'
            )
        groups.add(parsed[1])
    test_groups = sorted(groups & set(config.test_groups))
    remaining = sorted(groups - set(config.test_groups))
    if not test_groups:
        raise ValueError(f"None of the test groups {config.test_groups} are present in the dataset.")
    if len(remaining) <= config.num_val_groups:
        raise ValueError(
            f"Only {len(remaining)} non-test groups are available; cannot reserve {config.num_val_groups} for "
            "validation and still have training data. Lower num_val_groups or add data."
        )
    rng = np.random.default_rng(config.seed)
    val_groups = sorted(int(g) for g in rng.choice(remaining, size=config.num_val_groups, replace=False))
    train_groups = [g for g in remaining if g not in val_groups]

    def split_of(group: int) -> str:
        return "test" if group in test_groups else "val" if group in val_groups else "train"

    assignment = {r.group_key: split_of(parse_ucf101_name(r.path.stem)[1]) for r in records}
    description = {
        "protocol": "UCF101 official split 1 for the test set; validation groups drawn from the training groups",
        "test_groups": test_groups,
        "val_groups": val_groups,
        "train_groups": train_groups,
    }
    return assignment, description


def _split_random_groups(records: list[VideoRecord], config: Config) -> tuple[dict, dict]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.group not in by_label[record.label]:
            by_label[record.label].append(record.group)
    rng = np.random.default_rng(config.seed)
    assignment: dict[tuple[str, str], str] = {}
    for label in sorted(by_label):
        groups = sorted(by_label[label])
        if len(groups) < 3:
            raise ValueError(
                f"Class {label!r} has only {len(groups)} group(s); at least 3 are needed so that train, "
                "validation and test each get one. Add videos, or check the <group>__<name> file naming."
            )
        order = [groups[i] for i in rng.permutation(len(groups))]
        n_test = min(max(1, round(config.test_fraction * len(groups))), len(groups) - 2)
        n_val = min(max(1, round(config.val_fraction * len(groups))), len(groups) - n_test - 1)
        for index, group in enumerate(order):
            split = "test" if index < n_test else "val" if index < n_test + n_val else "train"
            assignment[(label, group)] = split
    description = {
        "protocol": "seeded random split of groups within each class",
        "test_fraction": config.test_fraction,
        "val_fraction": config.val_fraction,
    }
    return assignment, description


def assert_no_group_leakage(records: list[VideoRecord]) -> None:
    """Raise if clips of one (class, group) are spread over more than one split."""
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        previous = seen.setdefault(record.group_key, record.split)
        if previous != record.split:
            raise ValueError(
                f"Data leakage: group {record.group} of class {record.label} appears in both "
                f"{previous!r} and {record.split!r}."
            )


def split_counts(records: list[VideoRecord], classes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """``{split: {label: number of videos}}`` with explicit zeros."""
    counter = Counter((record.split, record.label) for record in records)
    return {split: {label: counter[(split, label)] for label in classes} for split in SPLITS}


def records_for_split(records: list[VideoRecord], split: str) -> list[VideoRecord]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    return [record for record in records if record.split == split]


def prepare_dataset(config: Config) -> dict:
    """Scan, validate and split the dataset, then write the manifest and split description."""
    scanned, label_problems = scan_dataset(config.dataset_dir, config.classes)
    log.info("Found %d video files in %d classes; checking that each one decodes...", len(scanned), len(config.classes))
    valid, decode_problems = validate_records(scanned)
    problems = label_problems + decode_problems
    for problem in problems:
        log.warning("Excluded %s: %s", problem["path"], problem["reason"])
    records, description = split_records(valid, config)

    counts = split_counts(records, config.classes)
    for split in SPLITS:
        empty = [label for label, n in counts[split].items() if n == 0]
        if empty:
            raise ValueError(
                f"The {split} split has no videos for class(es) {empty}. There is not enough usable data "
                "for these classes; add videos or choose other classes."
            )
    train_counts = counts["train"]
    if max(train_counts.values()) > 1.5 * min(train_counts.values()):
        log.warning("Training classes are imbalanced %s; class-balanced weights are used in training.", train_counts)

    config.splits_dir.mkdir(parents=True, exist_ok=True)
    with config.manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda r: r.video_id):
            writer.writerow(
                {
                    "video_id": record.video_id,
                    "label": record.label,
                    "group": record.group,
                    "split": record.split,
                    "num_frames": record.num_frames,
                    "path": record.path.relative_to(config.dataset_dir).as_posix(),
                }
            )
    frame_counts = [record.num_frames for record in records]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(config.dataset_dir),
        "classes": list(config.classes),
        "seed": config.seed,
        "split_mode": config.split_mode,
        "split": description,
        "counts": counts,
        "totals": {split: sum(counts[split].values()) for split in SPLITS},
        "num_videos": len(records),
        "frames_per_video": {"min": min(frame_counts), "max": max(frame_counts), "mean": float(np.mean(frame_counts))},
        "excluded_files": problems,
    }
    write_json(config.split_config_path, summary)

    log.info("%-18s %6s %6s %6s", "class", *SPLITS)
    for label in config.classes:
        log.info("%-18s %6d %6d %6d", label, *(counts[split][label] for split in SPLITS))
    log.info("%-18s %6d %6d %6d", "total", *(summary["totals"][split] for split in SPLITS))
    log.info("Excluded files: %d. Manifest written to %s", len(problems), config.manifest_path)
    return summary


def load_manifest(config: Config) -> list[VideoRecord]:
    """Read the split written by :func:`prepare_dataset` and re-check it for leakage."""
    if not config.manifest_path.is_file():
        raise FileNotFoundError(
            f"No dataset split found at {config.manifest_path}. Run `python -m visionact prepare` first."
        )
    with config.manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = [
        VideoRecord(
            video_id=row["video_id"],
            label=row["label"],
            group=row["group"],
            path=config.dataset_dir / row["path"],
            split=row["split"],
            num_frames=int(row["num_frames"]),
        )
        for row in rows
    ]
    manifest_classes = {record.label for record in records}
    if manifest_classes != set(config.classes):
        raise ValueError(
            f"The saved split covers classes {sorted(manifest_classes)} but the config asks for "
            f"{sorted(config.classes)}. Run `python -m visionact prepare` again."
        )
    assert_no_group_leakage(records)
    return records


def load_split_summary(config: Config) -> dict | None:
    return read_json(config.split_config_path) if config.split_config_path.is_file() else None


def experiment_record(config: Config, **details) -> dict:
    """What an experiment was run on: seed, classes and the exact split.

    The SHA-256 of the manifest identifies the split unambiguously, so two
    results can be compared only if their fingerprints match. ``details`` adds
    model-specific settings such as preprocessing and augmentation.
    """
    summary = load_split_summary(config) or {}
    return {
        "seed": config.seed,
        "classes": list(config.classes),
        "split_mode": config.split_mode,
        "manifest_sha256": hashlib.sha256(config.manifest_path.read_bytes()).hexdigest(),
        "split": summary.get("split"),
        "split_totals": summary.get("totals"),
        **details,
    }
