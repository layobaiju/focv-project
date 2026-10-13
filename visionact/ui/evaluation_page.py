"""Evaluation: per-class metrics, confusion matrix and misclassified videos for each model."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import streamlit as st

from ..plots import plot_confusion_matrix, plot_training_curves
from ..utils import read_json_if_exists
from .common import AppContext, load_results, missing_steps_message, project_status

_ALL_STEPS = ["dataset", "split", "features", "baseline", "lstm", "r3d_features", "r3d", "results"]


def render(ctx: AppContext) -> None:
    config = ctx.config
    st.header("Evaluation")
    results = load_results(config)
    if results is None:
        st.warning("No evaluation results were found. Metrics are shown only after the models have been evaluated.")
        st.markdown("Run the remaining steps from the project folder:")
        st.code(missing_steps_message(project_status(config), _ALL_STEPS), language="bash")
        return

    class_names = results["class_names"]
    st.caption(
        f"Measured on the {results['test_videos']} held-out test videos. No frame of a test video, and no other "
        "clip from the same source video, was used for training or model selection."
    )
    _split_summary(results.get("split"), class_names)

    models = results["models"]
    key = st.radio(
        "Model", list(models), format_func=lambda k: models[k]["display_name"], horizontal=True, key="eval_model"
    )
    model = models[key]

    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{model['accuracy']:.1%}")
    columns[1].metric("Macro precision", f"{model['macro_precision']:.3f}")
    columns[2].metric("Macro recall", f"{model['macro_recall']:.3f}")
    columns[3].metric("Macro F1", f"{model['macro_f1']:.3f}")
    interval = "95% interval for accuracy, treating videos as independent: {:.1%} to {:.1%}.".format(*model["accuracy_ci95"])
    if "accuracy_ci95_group_bootstrap" in model:
        interval += (
            " Resampling whole source videos instead, which allows for clips of one source video being right or "
            "wrong together: {:.1%} to {:.1%}.".format(*model["accuracy_ci95_group_bootstrap"])
        )
    st.caption(f"{interval} With {model['num_videos']} test videos, one video is worth {1 / model['num_videos']:.1%}.")

    left, right = st.columns(2)
    with left:
        st.subheader("Per-class metrics")
        st.dataframe(
            [
                {
                    "Class": name,
                    "Precision": round(scores["precision"], 3),
                    "Recall": round(scores["recall"], 3),
                    "F1": round(scores["f1"], 3),
                    "Test videos": scores["support"],
                }
                for name, scores in model["per_class"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            "**Precision**: of the videos predicted as a class, the share that truly are that class. "
            "**Recall**: of the videos of a class, the share that were found. **F1** is their harmonic mean, and "
            "**macro** scores average the classes equally."
        )
    with right:
        st.subheader("Confusion matrix")
        figure = plot_confusion_matrix(model["confusion_matrix"], class_names)
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    predictions_path = config.results_dir / key / "predictions.csv"
    if predictions_path.is_file():
        with predictions_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        wrong = [row for row in rows if row["correct"] == "0"]
        st.subheader(f"Misclassified test videos ({len(wrong)} of {len(rows)})")
        st.dataframe(
            [{"Video": r["video_id"], "True class": r["true_label"], "Predicted": r["predicted_label"]} for r in wrong],
            hide_index=True,
            width="stretch",
            height=260,
        )
        download = st.columns(4)
        download[0].download_button("Download predictions (CSV)", predictions_path.read_bytes(),
                                    file_name=f"{key}_predictions.csv", mime="text/csv", key=f"dl_pred_{key}")  # fmt: skip
        metrics_path = config.results_dir / key / "metrics.json"
        if metrics_path.is_file():
            download[1].download_button("Download metrics (JSON)", metrics_path.read_bytes(),
                                        file_name=f"{key}_metrics.json", mime="application/json", key=f"dl_metrics_{key}")  # fmt: skip

    training = read_json_if_exists(config.results_dir / key / "training.json")
    if training and training.get("history"):
        st.subheader("Training curves")
        figure = plot_training_curves(training["history"], training["selected"]["best_epoch"])
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption(
            f"Training stopped after {training['selected']['epochs_run']} epochs (early stopping on "
            f"{training['early_stopping']}). The weights from epoch {training['selected']['best_epoch']} were kept. "
            "These curves use the training and validation splits only."
        )


def _split_summary(split: dict | None, class_names: list[str]) -> None:
    if not split:
        return
    with st.expander("Dataset split"):
        counts = split["counts"]
        st.dataframe(
            [{"Class": name, **{s.capitalize(): counts[s][name] for s in ("train", "val", "test")}} for name in class_names]
            + [{"Class": "Total", **{s.capitalize(): split["totals"][s] for s in ("train", "val", "test")}}],
            hide_index=True,
            width="stretch",
        )
        details = split["split"]
        lines = [f"Protocol: {details['protocol']}. Seed {split['seed']}."]
        if "test_groups" in details:
            lines.append(
                f"Test groups: {details['test_groups']}. Validation groups: {details['val_groups']}. "
                f"Training groups: {details['train_groups']}."
            )
        lines.append(f"Files excluded as unreadable or mislabelled: {len(split['excluded_files'])}.")
        st.markdown(" ".join(lines))
