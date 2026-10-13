"""End-to-end inference on one video with any trained model.

``load_predictor(config, key)`` returns an object whose ``predict(path)`` decodes
the video, runs the model's backbone and classifier, and reports the predicted
class, the class scores and how long each stage took. Without a key it uses the
model with the best *validation* score (see :func:`registry.best_validated_model`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import baseline, lstm, r3d
from .config import Config
from .features import FeatureExtractor
from .registry import ModelSpec, best_validated_model, get_spec
from .utils import ModelNotTrainedError, configure_torch_threads


@dataclass(frozen=True)
class Prediction:
    model_key: str
    model_name: str
    label: str
    confidence: float | None  # softmax score of the predicted class; None for the SVM
    scores: dict[str, float]  # one score per class
    score_kind: str  # "softmax" (sums to 1, not calibrated) or "svm_decision" (signed margins)
    frame_indices: list[list[int]]  # frames used, one list per clip, in temporal order
    timings: dict[str, float]  # seconds: decode, backbone, classifier

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())


def get_backbone(backbones: dict, name: str):
    """Load a backbone once and reuse it for every model that shares it."""
    if name not in backbones:
        backbones[name] = FeatureExtractor() if name == "mobilenet_v3_small" else r3d.R3DFeatureExtractor()
    return backbones[name]


class Predictor:
    """One trained model, ready to classify videos."""

    def __init__(self, config: Config, spec: ModelSpec, backbones: dict) -> None:
        self.spec = spec
        self.classes = list(config.classes)
        # Load the classifier first: if it is missing we fail before loading a backbone.
        if spec.family == "svm":
            bundle = baseline.load_baseline(config, spec.key)
            self._classifier, self.num_frames = bundle["pipeline"], bundle["num_frames"]
            self.description = f"{self.num_frames} frames, MobileNetV3-Small features averaged over time"
        elif spec.family == "lstm":
            self._classifier, checkpoint = lstm.load_lstm(config, spec.key)
            self.num_frames = checkpoint["num_frames"]  # the sequence length the model was trained with
            self.description = f"{self.num_frames} frames, MobileNetV3-Small features read in order"
        else:
            self._classifier, checkpoint = r3d.load_r3d_head(config)
            self.clip_len, self.target_fps = checkpoint["clip_len"], checkpoint["target_fps"]
            self.num_clips = checkpoint["clips_per_video"]
            self.description = (
                f"{self.num_clips} clips of {self.clip_len} frames at about {self.target_fps:g} fps, R3D-18 features"
            )
        self._backbone = get_backbone(backbones, spec.backbone)

    def predict(self, video_path: str | Path) -> Prediction:
        if self.spec.family == "r3d_head":
            features, timing = self._backbone.extract_video(video_path, self.clip_len, self.target_fps, self.num_clips)
            indices = [[int(i) for i in clip] for clip in timing["indices"]]
        else:
            features, timing = self._backbone.extract_video(video_path, self.num_frames)
            indices = [[int(i) for i in timing["indices"]]]

        started = time.perf_counter()
        if self.spec.family == "svm":
            pooled = baseline.temporal_mean_pool(features[None])
            scores = baseline.class_scores(self._classifier, pooled, len(self.classes))[0]
            kind = "svm_decision"
        else:
            if self.spec.family == "lstm":
                logits = lstm.predict_logits(self._classifier, features[None])[0]
            else:
                logits = r3d.video_logits(self._classifier, features[None])[0]
            scores = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
            kind = "softmax"
        classifier_seconds = time.perf_counter() - started

        best = int(np.argmax(scores))
        return Prediction(
            model_key=self.spec.key,
            model_name=self.spec.display_name,
            label=self.classes[best],
            confidence=float(scores[best]) if kind == "softmax" else None,
            scores={name: float(score) for name, score in zip(self.classes, scores)},
            score_kind=kind,
            frame_indices=indices,
            timings={
                "decode": timing["decode_seconds"],
                "backbone": timing["backbone_seconds"],
                "classifier": classifier_seconds,
            },
        )


def load_predictor(config: Config, key: str | None = None, backbones: dict | None = None) -> Predictor:
    """Load a trained model for inference.

    ``key`` defaults to the model with the best validation score. ``backbones``
    is an optional dict that caches the pretrained networks, so several
    predictors can share them.
    """
    configure_torch_threads(config.torch_threads)
    if key is None:
        key = best_validated_model(config)
        if key is None:
            raise ModelNotTrainedError(
                "No trained model was found. Run `python -m visionact all`, or train one model "
                "(for example `python -m visionact train-lstm`)."
            )
    return Predictor(config, get_spec(key), backbones if backbones is not None else {})


def predict_video(video_path: str | Path, config: Config, key: str | None = None) -> Prediction:
    return load_predictor(config, key).predict(video_path)
