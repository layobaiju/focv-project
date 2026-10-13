"""Command-line entry point: ``python -m visionact <command>``.

Commands mirror the stages of the experiment, in the order they are run:

    download -> prepare -> extract -> train-baseline -> train-lstm
             -> extract-augmented -> train-baseline --augment -> train-lstm --augment
             -> extract-r3d -> train-r3d -> evaluate

``all`` runs everything after the download. Heavy modules are imported inside
each command so ``--help`` and ``download`` start instantly.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .registry import MODELS
from .utils import get_logger

log = get_logger(__name__)

# CLI flag -> Config field. Only the settings worth changing per run are exposed;
# everything else can be set in config.json.
_OVERRIDES = {
    "dataset_dir": dict(help="folder containing one sub-folder of videos per class"),
    "classes": dict(help="comma-separated class names, e.g. JumpingJack,PushUps"),
    "seed": dict(type=int),
    "num_frames": dict(type=int, help="frames sampled per video (LSTM sequence length)"),
    "hidden_dim": dict(type=int, help="LSTM hidden size"),
    "num_layers": dict(type=int, help="number of stacked LSTM layers"),
    "dropout": dict(type=float),
    "epochs": dict(type=int, help="maximum training epochs"),
    "batch_size": dict(type=int),
    "learning_rate": dict(type=float),
    "patience": dict(type=int, help="early-stopping patience in epochs"),
}

_WEIGHTS_HELP = "allow the one-time 127 MB R3D-18 weight download without asking"


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="JSON file with config overrides (default: config.json if present)")
    for field, options in _OVERRIDES.items():
        common.add_argument(f"--{field.replace('_', '-')}", dest=field, default=None, **options)

    parser = argparse.ArgumentParser(prog="python -m visionact", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, parents=[common], help=help)

    download = add("download", "download the configured UCF101 classes")
    download.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    download.add_argument("--workers", type=int, default=8, help="parallel downloads")

    add("prepare", "validate videos and write the train/val/test split")
    add("diagnose", "audit the data, the split and the baseline models")
    add("extract", "extract and cache MobileNetV3-Small features")
    add("extract-augmented", "cache augmented MobileNetV3-Small features for the training split")
    add("extract-r3d", "extract and cache R3D-18 clip features").add_argument("--yes", action="store_true", help=_WEIGHTS_HELP)

    train_baseline = add("train-baseline", "train the CNN + mean-pooling + linear SVM baseline")
    train_baseline.add_argument("--augment", action="store_true", help="also train on augmented training videos (saved as a separate model)")
    train_lstm = add("train-lstm", "train the CNN + LSTM classifier")
    train_lstm.add_argument("--search", action="store_true", help="pick hyperparameters on the validation set")
    train_lstm.add_argument("--augment", action="store_true", help="also train on augmented training videos (saved as a separate model)")
    add("train-r3d", "train the classification head on frozen R3D-18 features")

    evaluate = add("evaluate", "evaluate every trained model on the held-out test set")
    evaluate.add_argument("--timing-videos", type=int, default=20, help="test videos used for the CPU timing benchmark")
    add("motion-experiment", "optional: optical-flow histogram features with a linear SVM")

    predict = add("predict", "classify one video with a trained model")
    predict.add_argument("video", help="path to a video file")
    predict.add_argument("--model", default=None, help=f"one of: {', '.join(MODELS)} (default: best on validation)")

    run_all = add("all", "prepare, extract features, train every model and evaluate")
    run_all.add_argument("--timing-videos", type=int, default=20)
    run_all.add_argument("--yes", action="store_true", help=_WEIGHTS_HELP)
    return parser


def _config_from(args: argparse.Namespace) -> Config:
    overrides = {field: getattr(args, field) for field in _OVERRIDES}
    return Config.load(args.config, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = _config_from(args)
        return _dispatch(args, config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # These carry an actionable message written for the user; no traceback needed.
        log.error("%s", exc)
        return 1


def _dispatch(args: argparse.Namespace, config: Config) -> int:
    command = args.command
    if command == "download":
        from .download import download_subset

        counts = download_subset(config.classes, config.dataset_dir, assume_yes=args.yes, workers=args.workers)
        return 1 if counts["failed"] else 0
    if command == "predict":
        return _predict(args, config)

    if command == "all":
        # (step, augment). The plain baselines come first and are never overwritten by the variants.
        steps = [("prepare", False), ("extract", False), ("train-baseline", False), ("train-lstm", False)]
        if config.augment_copies > 0:
            steps += [("extract-augmented", False), ("train-baseline", True), ("train-lstm", True)]
        steps += [("extract-r3d", False), ("train-r3d", False), ("evaluate", False)]
    else:
        steps = [(command, getattr(args, "augment", False))]
    for step, augment in steps:
        log.info("=== %s%s ===", step, " (augmented training)" if augment else "")
        _run_step(step, augment, args, config)
    return 0


def _predict(args: argparse.Namespace, config: Config) -> int:
    from .inference import predict_video

    prediction = predict_video(args.video, config, args.model)
    confidence = f" ({prediction.confidence:.1%})" if prediction.confidence is not None else ""
    print(f"Model: {prediction.model_name}")
    print(f"Predicted activity: {prediction.label}{confidence}")
    for name, score in sorted(prediction.scores.items(), key=lambda item: -item[1]):
        shown = f"{score:6.1%}" if prediction.score_kind == "softmax" else f"{score:+.3f}"
        print(f"  {name:<20s} {shown}")
    if prediction.score_kind != "softmax":
        print("  (SVM decision values: larger means more confident; they are not probabilities)")
    print(f"CPU time: {prediction.total_seconds * 1000:.0f} ms")
    return 0


def _run_step(step: str, augment: bool, args: argparse.Namespace, config: Config) -> None:
    if step == "prepare":
        from .dataset import prepare_dataset

        prepare_dataset(config)
    elif step == "diagnose":
        from .diagnostics import run_diagnostics

        run_diagnostics(config)
    elif step == "extract":
        from .features import extract_dataset_features

        extract_dataset_features(config)
    elif step == "extract-augmented":
        from .features import extract_augmented_features

        extract_augmented_features(config)
    elif step == "extract-r3d":
        from .r3d import extract_r3d_features

        extract_r3d_features(config, assume_yes=getattr(args, "yes", False))
    elif step == "train-baseline":
        from .baseline import train_baseline

        train_baseline(config, augment=augment)
    elif step == "train-lstm":
        from .lstm import train_lstm

        # ``all`` always searches, so the reported model is the validation-selected one.
        train_lstm(config, search=args.command == "all" or getattr(args, "search", False), augment=augment)
    elif step == "train-r3d":
        from .r3d import train_r3d

        train_r3d(config)
    elif step == "evaluate":
        from .evaluation import evaluate_models

        evaluate_models(config, timing_videos=args.timing_videos)
    elif step == "motion-experiment":
        from .motion_experiment import run_motion_experiment

        run_motion_experiment(config)


if __name__ == "__main__":
    sys.exit(main())
