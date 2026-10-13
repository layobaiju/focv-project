"""Home: what the project does, its architecture and the current state of the pipeline."""

from __future__ import annotations

import streamlit as st

from .common import PIPELINE_COMMANDS, AppContext, project_status

_PIPELINE = """
digraph {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#5b7db1", fontname="Helvetica", fontsize=11];
    edge [color="#5b7db1"];
    video [label="Input video"];
    frames [label="OpenCV\\nsample 16 frames\\nresize, crop, normalise"];
    cnn [label="MobileNetV3-Small\\n(frozen, ImageNet)\\n1024-d per frame"];
    lstm [label="LSTM\\nreads frames in order"];
    pool [label="Mean over time\\n+ linear SVM\\n(baseline)", fillcolor="#f4f4f4", color="#888888"];
    clips [label="OpenCV\\n5 clips of 16 frames\\nresize, crop, normalise", fillcolor="#eef8ee", color="#4f8f4f"];
    r3d [label="R3D-18\\n(frozen, Kinetics-400)\\n512-d per clip", fillcolor="#eef8ee", color="#4f8f4f"];
    head [label="Linear head\\naverage over clips", fillcolor="#eef8ee", color="#4f8f4f"];
    out [label="Activity class\\n+ class scores"];
    eval [label="Evaluation\\naccuracy, F1,\\nconfusion matrix"];
    video -> frames -> cnn -> lstm -> out -> eval;
    cnn -> pool [style=dashed, color="#888888"];
    pool -> out [style=dashed, color="#888888"];
    video -> clips [color="#4f8f4f"];
    clips -> r3d -> head -> out [color="#4f8f4f"];
}
"""

_STAGES = [
    ("dataset", "Dataset downloaded"),
    ("split", "Videos validated and split by source video"),
    ("features", "MobileNetV3-Small features extracted and cached"),
    ("baseline", "CNN + SVM baseline trained"),
    ("lstm", "CNN + LSTM trained"),
    ("r3d_features", "R3D-18 clip features extracted and cached"),
    ("r3d", "R3D-18 classification head trained"),
    ("results", "All trained models evaluated on the test set"),
]


def render(ctx: AppContext) -> None:
    config = ctx.config
    st.title("VisionAct")
    st.subheader("Human activity recognition and motion analysis using computer vision")
    st.markdown(
        f"""
VisionAct takes a video, samples frames from it, and classifies the human activity it shows as one of
**{len(config.classes)} classes**: {", ".join(config.classes)}. It is an academic project for *Foundations of
Computer Vision* that implements the course topics as working features and compares three ways of
recognising an activity from the same videos:

1. **Appearance only.** Frame features from an image CNN, averaged over time, with a linear SVM (the baseline).
2. **Appearance in order.** The same frame features read in sequence by an LSTM.
3. **Appearance and motion together.** A 3D CNN (R3D-18) pretrained on action videos, with a new classification head.
"""
    )

    st.subheader("Architecture")
    st.graphviz_chart(_PIPELINE, width="stretch")
    st.caption(
        "Blue: the CNN + LSTM pipeline. Grey, dashed: the order-blind baseline on the same frame features. "
        "Green: the R3D-18 pipeline. All models are trained and tested on exactly the same videos, and "
        "everything runs on the CPU."
    )

    st.subheader("How to use this app")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
**Demonstrations** (work on any video, no trained model needed)

1. **Video Analysis** shows metadata, sampled frames and the CNN preprocessing.
2. **Image Processing** applies filtering, thresholding, morphology, edge detection and shape analysis to a frame.
3. **Motion Analysis** visualises optical flow.
"""
        )
    with right:
        st.markdown(
            """
**Model results** (need trained models)

4. **Activity Recognition** classifies the selected video with any trained model.
5. **Model Comparison** compares all trained models on the same test videos.
6. **Evaluation** shows per-class metrics and confusion matrices.
"""
        )

    st.subheader("Pipeline status")
    status = project_status(config)
    rows = [
        {"Stage": title, "Done": "yes" if status[key] else "no", "Command": PIPELINE_COMMANDS[key]}
        for key, title in _STAGES
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    if all(status[key] for key, _ in _STAGES):
        st.success("All stages are complete. The model sections show results computed from the saved models.")
    else:
        st.info(
            "Some stages have not been run, so the model sections show only what is available and say what is "
            "missing. Run the commands above in order from the project folder (or `python -m visionact all` "
            "after the download). The demonstration sections work now."
        )
