"""Activity Recognition: run a trained model on the selected video."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from ..plots import plot_class_scores
from ..registry import MODELS, best_validated_model, checkpoint_path, trained_models, validation_score
from ..video import VideoError
from .common import AppContext, cached_frames_at, missing_steps_message, project_status, require_video, show_image

_FRAMES_PER_ROW = 8


@st.cache_resource(show_spinner=False)
def _backbones() -> dict:
    """Pretrained networks shared by every model (each is loaded once per app session)."""
    return {}


@st.cache_resource(show_spinner="Loading the model...")
def _load_predictor(_config, key: str, checkpoint: str, checkpoint_mtime: float):
    # Imported here so the demonstration pages start without loading PyTorch.
    from ..inference import load_predictor

    return load_predictor(_config, key, _backbones())


def _predictor(config, key: str):
    path = checkpoint_path(config, key)
    return _load_predictor(config, key, str(path), path.stat().st_mtime)


def render(ctx: AppContext) -> None:
    config = ctx.config
    st.header("Activity Recognition")
    trained = trained_models(config)
    if not trained:
        st.warning(
            "No trained model was found, so no prediction can be shown. "
            "This section never displays a prediction unless it comes from a trained checkpoint."
        )
        st.markdown("To train the models, run these commands from the project folder:")
        steps = ["dataset", "split", "features", "baseline", "lstm", "r3d_features", "r3d"]
        st.code(missing_steps_message(project_status(config), steps), language="bash")
        return

    st.caption("Model prediction: the output below comes from a trained checkpoint.")
    if not require_video(ctx):
        return
    video = ctx.video

    default = best_validated_model(config) or trained[-1]
    key = st.selectbox(
        "Model",
        trained,
        index=trained.index(default),
        format_func=lambda k: MODELS[k].display_name + (" (best on validation)" if k == default else ""),
        key="model_choice",
        help="The default is the model with the best validation score. Test results are not used to choose it.",
    )
    try:
        predictor = _predictor(config, key)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        st.error(f"The model could not be loaded: {exc}")
        return

    validation = validation_score(config, key) or {}
    validated = f" Validation accuracy when it was selected: {validation['val_accuracy']:.1%}." if "val_accuracy" in validation else ""
    st.markdown(
        f"**{MODELS[key].display_name}.** {MODELS[key].summary} Input: {predictor.description}. "
        f"Classes: {', '.join(predictor.classes)}.{validated}"
    )
    st.info(
        "The model can only answer with one of its classes. A video of any other activity, or of no person at "
        "all, will still be assigned to one of them, so a prediction is meaningful only for these activities."
    )

    if st.button("Classify this video", type="primary", key="classify"):
        try:
            with st.spinner("Decoding frames, extracting features and classifying..."):
                st.session_state["prediction"] = (video.cache_key, key, predictor.predict(video.path))
        except VideoError as exc:
            st.error(f"This video cannot be classified: {exc}")
            return
    stored = st.session_state.get("prediction")
    if not stored or stored[:2] != (video.cache_key, key):
        st.markdown(f"Selected video: **{video.name}**. Press the button to run the model on it.")
        return
    prediction = stored[2]

    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted activity", prediction.label)
        if prediction.confidence is not None:
            st.metric("Score of the predicted class", f"{prediction.confidence:.1%}")
        if video.label:
            verdict = "correct" if video.label == prediction.label else "incorrect"
            st.metric("True class (dataset label)", video.label, delta=verdict,
                      delta_color="normal" if verdict == "correct" else "inverse", delta_arrow="off")  # fmt: skip
    with right:
        if prediction.score_kind == "softmax":
            figure = plot_class_scores(prediction.scores, f"{prediction.model_name}: class scores")
            st.pyplot(figure, width="stretch")
            plt.close(figure)
            st.caption(
                "Scores are the softmax outputs of the network. They sum to 100% over the known classes and are "
                "not calibrated probabilities."
            )
        else:
            st.dataframe(
                [{"Class": name, "SVM decision value": round(score, 3)}
                 for name, score in sorted(prediction.scores.items(), key=lambda item: -item[1])],
                hide_index=True, width="stretch",
            )  # fmt: skip
            st.caption(
                "A linear SVM outputs signed distances from its decision boundaries, not probabilities. The class "
                "with the largest value is predicted."
            )

    if video.split == "train":
        st.warning(
            "This clip was part of the training data, so a correct prediction here says nothing about "
            "generalisation. Choose a clip marked (test) for a fair check."
        )
    elif video.split == "val":
        st.warning("This clip was used for model selection (validation). Choose a clip marked (test) for a fair check.")
    elif video.split == "test":
        st.caption("This clip belongs to the held-out test set: the model never saw it during training or selection.")

    timings = prediction.timings
    st.markdown(
        f"**CPU time:** {prediction.total_seconds * 1000:.0f} ms in total "
        f"(decoding {timings['decode'] * 1000:.0f} ms, preprocessing and backbone {timings['backbone'] * 1000:.0f} ms, "
        f"classifier {timings['classifier'] * 1000:.1f} ms)."
    )

    with st.expander("Frames the model used"):
        for number, clip in enumerate(prediction.frame_indices, start=1):
            if len(prediction.frame_indices) > 1:
                st.caption(f"Clip {number}: frames {clip[0]} to {clip[-1]} (every second frame of the clip shown)")
                clip = clip[::2]
            frames = cached_frames_at(*video.cache_key, tuple(clip))
            for start in range(0, len(frames), _FRAMES_PER_ROW):
                for column, frame, index in zip(st.columns(_FRAMES_PER_ROW), frames[start:], clip[start:]):
                    with column:
                        show_image(frame, f"frame {index}")

    if len(trained) > 1:
        st.subheader("All trained models on this video")
        if st.button("Compare all trained models", key="compare_all"):
            rows = []
            with st.spinner("Running every trained model..."):
                for other in trained:
                    result = _predictor(config, other).predict(video.path)
                    rows.append(
                        {
                            "Model": result.model_name,
                            "Prediction": result.label,
                            "Score": f"{result.confidence:.1%}" if result.confidence is not None else "n/a (SVM)",
                            "Correct": ("yes" if result.label == video.label else "no") if video.label else "unknown",
                            "CPU time": f"{result.total_seconds * 1000:.0f} ms",
                        }
                    )
            st.session_state["comparison_rows"] = (video.cache_key, rows)
        compared = st.session_state.get("comparison_rows")
        if compared and compared[0] == video.cache_key:
            st.dataframe(compared[1], hide_index=True, width="stretch")
