"""About: where each course topic is demonstrated, what feeds the classifier, and the project's limits."""

from __future__ import annotations

from importlib import metadata

import streamlit as st

from ..config import DEVICE
from .common import AppContext, load_results

_CLASSIFIER = "Part of the classifier pipeline"
_DEMO = "Independent demonstration"
_EXPERIMENT = "Separate experiment"

# (topic, what is implemented, role, app section, module)
_TOPICS = [
    ("Image transformations", "Frame and clip sampling, resizing, cropping, normalisation", _CLASSIFIER, "Video Analysis", "video.py"),
    ("Image transformations", "Random crops, brightness changes and mirroring of training clips", _CLASSIFIER + " (training only)", "none", "augment.py"),
    ("Image preprocessing", "Gaussian and median filtering", _DEMO, "Image Processing", "image_processing.py"),
    ("Image preprocessing", "Global, Otsu and adaptive thresholding; erosion, dilation, opening, closing", _DEMO, "Image Processing", "image_processing.py"),
    ("Edge detection and features", "Canny edge maps; edges as image structure", _DEMO, "Image Processing", "image_processing.py"),
    ("Object and shape analysis", "Connected-component labelling, contours, bounding boxes", _DEMO, "Image Processing", "image_processing.py"),
    ("Motion analysis", "Dense (Farneback) and sparse (Lucas-Kanade) optical flow", _DEMO, "Motion Analysis", "optical_flow.py"),
    ("Motion analysis", "Optical-flow histograms combined with CNN features", _EXPERIMENT, "Model Comparison", "motion_experiment.py"),
    ("CNN", "MobileNetV3-Small as a frozen appearance feature extractor", _CLASSIFIER, "Activity Recognition", "features.py"),
    ("CNN", "R3D-18, a 3D CNN pretrained on action videos, with a new classification head", _CLASSIFIER, "Activity Recognition", "r3d.py"),
    ("RNN", "LSTM over the ordered frame features", _CLASSIFIER, "Activity Recognition", "lstm.py"),
    ("Classification and evaluation", "Accuracy, precision, recall, F1, confusion matrix; all models on the same test videos", _CLASSIFIER, "Model Comparison, Evaluation", "baseline.py, evaluation.py"),
]

_PACKAGES = ["torch", "torchvision", "opencv-python-headless", "numpy", "scikit-learn", "streamlit", "matplotlib"]


def render(ctx: AppContext) -> None:
    config = ctx.config
    st.header("About VisionAct")
    st.markdown(
        "VisionAct is an academic project for *Foundations of Computer Vision*. Its contribution is an "
        "implementation and an experimental comparison of standard techniques. It does not propose a new algorithm."
    )

    st.subheader("Course topics and where they appear")
    st.dataframe(
        [{"Topic": topic, "What is implemented": what, "Role": role, "App section": section, "Module (visionact/)": module}
         for topic, what, role, section, module in _TOPICS],
        hide_index=True,
        width="stretch",
    )  # fmt: skip
    st.markdown(
        """
**What feeds the classifiers.** Only frame sampling, resizing, cropping, normalisation and the neural networks
(with augmentation while training) are on the path from a video to a prediction.

**What is demonstrated separately.** Filtering, thresholding, morphology, Canny edges, contours and connected
components are shown on single frames to illustrate the course topics. They are deliberately not fed to the
classifiers: the networks were pretrained on unfiltered images, and a threshold mask of a natural video frame
isolates bright regions rather than people.

**Optical flow** is a demonstration and also a separate experiment that tests whether adding flow features to the
visual features helps. It is used in the prediction pipeline only if it improves validation results (see Model
Comparison).
"""
    )

    st.subheader("Limitations")
    results = load_results(config)
    test_videos = results["test_videos"] if results else None
    st.markdown(
        f"""
- **Closed set of classes.** The classifiers know only {", ".join(config.classes)}. Any other input is still
  assigned to one of these.
- **Small experiment.** {"The test set has " + str(test_videos) + " videos" if test_videos else "The test set is small"}
  cut from a few dozen source videos, so accuracy figures carry wide uncertainty intervals. Results come from a
  single random seed.
- **Few independent training examples.** With the default split each class has 14 source videos for training.
  The frame-based models fit the training clips perfectly yet make many more errors on new source videos.
- **Unreliable validation.** With the default 4 validation source videos per class, validation scores move by
  many points depending on which videos are drawn, so model selection is noisy.
- **Pretraining overlap.** R3D-18 was pretrained on Kinetics-400, which contains push-up, pull-up and squat
  classes. Its strong results partly reflect that prior knowledge, and whether any Kinetics video shares a source
  with a UCF101 test clip has not been checked.
- **Dataset bias.** UCF101 clips come from YouTube, and clips of a class often share settings and camera angles.
  Performance on other footage is unknown and has not been tested.
- **Not for real-world decisions.** This is a teaching project. It has not been validated for safety, health,
  surveillance or any other operational use.
"""
    )

    st.subheader("Data")
    st.markdown(
        "The experiments use a subset of **UCF101** (Soomro, Zamir and Shah, 2012, "
        "[crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)). Clips cut from the same source "
        "video form a *group*; groups are never divided between training, validation and test."
    )

    st.subheader("Environment")
    versions = []
    for package in _PACKAGES:
        try:
            versions.append({"Package": package, "Version": metadata.version(package)})
        except metadata.PackageNotFoundError:
            versions.append({"Package": package, "Version": "not installed"})
    st.dataframe(versions, hide_index=True, width="content")
    st.caption(f"Device: {DEVICE}. Seed: {config.seed}.")
