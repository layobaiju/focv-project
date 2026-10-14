"""VisionAct app. Launch on Windows with Start VisionAct.bat or with Streamlit directly."""

from __future__ import annotations

import streamlit as st

from visionact import registry
from visionact.config import Config
from visionact.ui import (
    about,
    common,
    comparison_page,
    evaluation_page,
    home,
    image_page,
    motion_page,
    recognition_page,
    video_page,
)

SECTIONS = {
    "Home": home.render,
    "Video Analysis": video_page.render,
    "Image Processing": image_page.render,
    "Motion Analysis": motion_page.render,
    "Activity Recognition": recognition_page.render,
    "Model Comparison": comparison_page.render,
    "Evaluation": evaluation_page.render,
    "About": about.render,
}


def main() -> None:
    st.set_page_config(page_title="VisionAct", layout="wide")
    try:
        config = Config.load()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        st.error(f"The configuration could not be loaded: {exc}")
        st.stop()

    with st.sidebar:
        st.title("VisionAct")
        section = st.radio("Section", list(SECTIONS), key="section")
        st.divider()
        video = common.select_video(config)
        st.divider()
        status = common.project_status(config)
        trained = len(registry.trained_models(config))
        st.caption(
            f"Trained models: {trained} of {len(registry.MODELS)}"
            + "  \nEvaluation results: " + ("available" if status["results"] else "not available yet")
        )  # fmt: skip

    SECTIONS[section](common.AppContext(config, video))


main()
