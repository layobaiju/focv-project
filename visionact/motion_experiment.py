"""Optional experiment: does explicit optical-flow information add to the visual features?

Separate from the main pipeline. For each video, dense optical flow is computed
for a few pairs of consecutive frames and summarised as histograms of flow
direction (HOF). Linear SVMs are then trained, on the same splits as the main
models, for each visual feature set with and without the flow descriptor:

* ``flow_svm``         flow descriptor only,
* ``cnn_pooled_svm``   mean-pooled MobileNetV3-Small features (identical to the baseline),
* ``cnn_flow_svm``     the same plus the flow descriptor,
* ``r3d_pooled_svm``   mean-pooled R3D-18 clip features,
* ``r3d_flow_svm``     the same plus the flow descriptor.

Whether flow fusion belongs in the prediction pipeline is decided from the
*validation* results by :func:`fusion_decision`; test accuracy is reported but
plays no part. Results go to ``results/motion_experiment.json``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import baseline
from .config import Config
from .dataset import VideoRecord, load_manifest, records_for_split
from .evaluation import compute_metrics, group_bootstrap_interval
from .features import load_features
from .optical_flow import video_flow_features
from .utils import get_logger, write_json
from .video import VideoError, count_decodable_frames, read_frames_at, read_metadata, sample_frame_indices

log = get_logger(__name__)

NUM_PAIRS = 8  # frame pairs per video, spread evenly over its length
HOF_BINS, HOF_GRID = 8, (2, 2)
MIN_VIDEOS_GAINED = 2  # one extra correct validation video is within noise
MAX_FLOW_MS = 1000.0  # flow must cost less than this per video to be practical on a CPU
DISPLAY_NAMES = {
    "flow_svm": "Optical-flow histograms + SVM",
    "cnn_pooled_svm": "MobileNetV3 mean pooling + SVM (baseline)",
    "cnn_flow_svm": "MobileNetV3 mean pooling + optical flow + SVM",
    "r3d_pooled_svm": "R3D-18 mean pooling + SVM",
    "r3d_flow_svm": "R3D-18 mean pooling + optical flow + SVM",
}


def _read_pairs(path: Path, frame_count: int, num_pairs: int) -> list[tuple[np.ndarray, np.ndarray]] | None:
    starts = sample_frame_indices(frame_count - 1, num_pairs)
    wanted = sorted({int(i) for start in starts for i in (start, start + 1)})
    frames = read_frames_at(path, wanted)
    if len(frames) < len(wanted):
        return None  # the video is shorter than its header claims
    by_index = dict(zip(wanted, frames))
    return [(by_index[int(start)], by_index[int(start) + 1]) for start in starts]


def video_motion_descriptor(path: str | Path, num_pairs: int = NUM_PAIRS) -> np.ndarray:
    """HOF for ``num_pairs`` consecutive-frame pairs: shape ``(num_pairs, 36)``, in temporal order."""
    path = Path(path)
    frame_count = read_metadata(path).frame_count or count_decodable_frames(path)
    pairs = _read_pairs(path, frame_count, num_pairs) if frame_count >= 2 else None
    if pairs is None:
        frame_count = count_decodable_frames(path)
        if frame_count < 2:
            raise VideoError(f"{path.name} has fewer than two frames; optical flow needs a pair of frames.")
        pairs = _read_pairs(path, frame_count, num_pairs)
    return np.concatenate([video_flow_features(pair, HOF_BINS, HOF_GRID) for pair in pairs])


def pool_motion(descriptors: np.ndarray) -> np.ndarray:
    """(N, pairs, D) -> (N, 2D): mean and standard deviation over time."""
    return np.concatenate([descriptors.mean(axis=1), descriptors.std(axis=1)], axis=1)


def _load_motion(config: Config, records: list[VideoRecord]) -> np.ndarray:
    cache_dir = config.artifacts_dir / "features" / f"hof-P{NUM_PAIRS}"
    descriptors = []
    for record in records:
        cached = cache_dir / f"{record.video_id}.npy"
        if cached.is_file():
            descriptors.append(np.load(cached))
            continue
        descriptor = video_motion_descriptor(record.path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, descriptor)
        descriptors.append(descriptor)
    return np.stack(descriptors)


def _time_flow(records: list[VideoRecord], sample: int = 8) -> float:
    """Milliseconds to compute the flow descriptor of one video (mean over a few videos, cache bypassed)."""
    chosen = records[:: max(len(records) // sample, 1)][:sample]
    started = time.perf_counter()
    for record in chosen:
        video_motion_descriptor(record.path)
    return 1000 * (time.perf_counter() - started) / len(chosen)


def fusion_decision(visual: dict, fused: dict, val_videos: int, flow_ms_per_video: float) -> dict:
    """Decide from validation results whether flow fusion earns a place in the prediction pipeline.

    It must get at least ``MIN_VIDEOS_GAINED`` more validation videos right
    than the visual features alone, improve validation macro-F1, and be
    affordable on a CPU. The thresholds were fixed before looking at results.
    """
    gained = round((fused["val_accuracy"] - visual["val_accuracy"]) * val_videos)
    f1_gain = fused["val_macro_f1"] - visual["val_macro_f1"]
    change = f"Adding flow changed validation accuracy by {gained:+d} video(s) and macro-F1 by {f1_gain:+.3f}"
    if gained < MIN_VIDEOS_GAINED or f1_gain <= 0:
        retained = False
        reason = (
            f"{change}; at least {MIN_VIDEOS_GAINED} more correct validation videos and a higher macro-F1 "
            "are required, so flow fusion is not used."
        )
    elif flow_ms_per_video > MAX_FLOW_MS:
        retained = False
        reason = f"{change}, but computing flow takes {flow_ms_per_video:.0f} ms per video, which is too slow."
    else:
        retained = True
        reason = f"{change} at a cost of {flow_ms_per_video:.0f} ms per video, so flow fusion is worth using."
    return {
        "retained": retained,
        "validation_videos_gained": int(gained),
        "validation_macro_f1_gain": float(f1_gain),
        "reason": reason,
    }


def run_motion_experiment(config: Config) -> dict:
    records = load_manifest(config)
    class_names = list(config.classes)
    splits = {split: records_for_split(records, split) for split in ("train", "val", "test")}
    test_groups = [record.group_key for record in splits["test"]]

    log.info("Computing optical-flow descriptors for %d videos (cached after the first run)...", len(records))
    motion = {split: pool_motion(_load_motion(config, items)) for split, items in splits.items()}
    flow_ms = _time_flow(splits["test"])

    labels, visual = {}, {"cnn": {}}
    for split, items in splits.items():
        features, labels[split] = load_features(config, items)
        visual["cnn"][split] = baseline.temporal_mean_pool(features)
    try:
        from .r3d import load_r3d_features

        visual["r3d"] = {split: load_r3d_features(config, items)[0].mean(axis=1) for split, items in splits.items()}
    except FileNotFoundError:
        log.info("R3D-18 features are not cached; the R3D rows are skipped (run `python -m visionact extract-r3d`).")

    inputs = {"flow_svm": motion}
    for name, data in visual.items():
        inputs[f"{name}_pooled_svm"] = data
        inputs[f"{name}_flow_svm"] = {split: np.concatenate([data[split], motion[split]], axis=1) for split in splits}

    models = {}
    for key, data in inputs.items():
        log.info("--- %s ---", DISPLAY_NAMES[key])
        model, selected, candidates = baseline.fit_and_select_svm(data["train"], labels["train"], data["val"], labels["val"], config)
        predicted = baseline.class_scores(model, data["test"], len(class_names)).argmax(axis=1)
        metrics = compute_metrics(labels["test"], predicted, class_names)
        metrics["accuracy_ci95_group_bootstrap"] = list(
            group_bootstrap_interval(predicted == labels["test"], test_groups, config.seed)
        )
        models[key] = {
            "display_name": DISPLAY_NAMES[key],
            "input_dim": int(data["train"].shape[1]),
            "validation": {"val_accuracy": selected["val_accuracy"], "val_macro_f1": selected["val_macro_f1"]},
            "selected_hyperparameters": selected,
            "candidates": candidates,
            **metrics,
        }
        log.info("%s: validation accuracy %.3f, test accuracy %.3f", DISPLAY_NAMES[key],
                 selected["val_accuracy"], metrics["accuracy"])  # fmt: skip

    val_videos = int(len(labels["val"]))
    decisions = {
        name: fusion_decision(
            models[f"{name}_pooled_svm"]["validation"], models[f"{name}_flow_svm"]["validation"], val_videos, flow_ms
        )
        for name in visual
    }
    for name, decision in decisions.items():
        log.info("Flow fusion with %s features: %s", name.upper(), decision["reason"])

    results = {
        "description": (
            f"Farneback dense optical flow on {NUM_PAIRS} consecutive-frame pairs per video (frames downscaled to "
            f"160 px wide), summarised as {HOF_BINS}-bin direction histograms on a {HOF_GRID[0]}x{HOF_GRID[1]} grid, "
            "then mean and standard deviation over time."
        ),
        "decision_rule": (
            f"Flow fusion is used only if it gets at least {MIN_VIDEOS_GAINED} more validation videos right than "
            f"the visual features alone, improves validation macro-F1, and costs under {MAX_FLOW_MS:.0f} ms per video."
        ),
        "class_names": class_names,
        "val_videos": val_videos,
        "test_videos": int(len(labels["test"])),
        "flow_ms_per_video": flow_ms,
        "models": models,
        "decisions": decisions,
    }
    write_json(config.results_dir / "motion_experiment.json", results)
    return results
