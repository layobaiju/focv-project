"""Model Comparison: every trained model on the same held-out test videos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from ..plots import plot_metric_comparison
from ..utils import read_json_if_exists, reported_interval
from .common import AppContext, load_results, missing_steps_message, project_status

_ALL_STEPS = ["dataset", "split", "features", "baseline", "lstm", "r3d_features", "r3d", "results"]


def _percent_range(model: dict) -> str:
    """The 95% interval; marked with * when it had to fall back to treating videos as independent."""
    low, high, basis = reported_interval(model)
    return f"{low:.1%} to {high:.1%}" + ("" if basis == "source videos" else " *")


def render(ctx: AppContext) -> None:
    config = ctx.config
    st.header("Model Comparison")
    results = load_results(config)
    if results is None or "paired_comparisons" not in results:
        st.warning("No evaluation results were found, so there is nothing to compare yet.")
        st.markdown("Run the remaining steps from the project folder:")
        st.code(missing_steps_message(project_status(config), _ALL_STEPS), language="bash")
        return

    models = results["models"]
    reference_key, best_key = results["reference_model"], results.get("best_validated_model")
    reference = models[reference_key]
    st.caption(
        f"Measured results on {results['test_videos']} held-out test videos from {results['test_source_videos']} "
        f"source videos ({', '.join(results['class_names'])}). Every model is scored on the same videos. "
        f"Evaluated {results['generated_at'][:10]}."
    )

    st.dataframe(
        [
            {
                "Model": model["display_name"],
                "Test accuracy": f"{model['accuracy']:.1%}",
                "95% interval": _percent_range(model),
                "Macro F1": f"{model['macro_f1']:.3f}",
                "Validation accuracy": f"{model['validation']['val_accuracy']:.1%}",
                "Training time": f"{model['training_seconds']:.2f} s",
                "Inference per video": _milliseconds(model["inference_ms_per_video"]),
                "Size with backbone": f"{model['total_size_mb']:.1f} MB",
            }
            for model in models.values()
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "The 95% interval resamples whole source videos, because clips cut from one source video tend to be right "
        "or wrong together. An interval marked * belongs to a model with no test errors: resampling cannot show "
        "uncertainty then, so it treats the videos as independent and is too narrow. Training time is for the "
        "selected model on cached features."
    )
    if best_key in models:
        best = models[best_key]
        gain = (best["accuracy"] - reference["accuracy"]) * 100
        st.markdown(
            f"**{best['display_name']}** is the best on validation and is the default for predictions. "
            + (
                "It is also the reference baseline."
                if best_key == reference_key
                else f"On the test set it scores {best['accuracy']:.1%} against {reference['accuracy']:.1%} for the "
                f"baseline ({reference['display_name']}), a difference of {gain:+.1f} percentage points."
            )
        )
    figure = plot_metric_comparison(
        {model["display_name"]: model for model in models.values()},
        [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")],
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    if results["paired_comparisons"]:
        st.subheader("Are the differences real?")
        st.dataframe(
            [
                {
                    "Model": models[key]["display_name"],
                    "Both correct": pair["both_correct"],
                    "Only the baseline correct": pair["only_first_correct"],
                    "Only this model correct": pair["only_second_correct"],
                    "Both wrong": pair["both_wrong"],
                    "McNemar p": f"{pair['mcnemar_exact_p']:.3f}",
                    "Below 0.05": "yes" if pair["mcnemar_exact_p"] < 0.05 else "no",
                }
                for key, pair in results["paired_comparisons"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            f"Each model is compared with the baseline ({reference['display_name']}) on the same videos. Only the "
            "videos where exactly one of the two is right say anything about which is better; the exact McNemar "
            "test asks whether that split could be chance. A p-value that is not below 0.05 means the data do not "
            "show a difference. The test treats videos as independent, so with errors clustered in source videos "
            "it is somewhat optimistic; the intervals in the first table are the more cautious guide."
        )

    if results["temporal_order_ablation"]:
        st.subheader("Do the LSTMs use the order of the frames?")
        st.dataframe(
            [
                {
                    "Model": models[key]["display_name"],
                    "Original order": f"{ablation['original_order_accuracy']:.1%}",
                    "Frames reversed": f"{ablation['reversed_order_accuracy']:.1%}",
                    "Frames shuffled": f"{ablation['shuffled_order_accuracy']:.1%}",
                }
                for key, ablation in results["temporal_order_ablation"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        drops = [a["original_order_accuracy"] - a["shuffled_order_accuracy"] for a in results["temporal_order_ablation"].values()]
        st.markdown(
            "Each trained LSTM was re-tested on the same videos with the frame order destroyed. "
            + (
                "Accuracy falls clearly when the frames are shuffled, so temporal order is being used."
                if max(drops) > 0.03
                else "Destroying the order costs little or nothing, so these models recognise the activities mostly "
                "from what the frames look like, not from the order in which they occur."
            )
        )

    timing = results.get("timing")
    if timing:
        st.subheader("CPU cost")
        st.dataframe(
            [
                {
                    "Model": models[key]["display_name"],
                    "Input": stages["input"],
                    "Decoding": f"{stages['decode_ms']:.0f} ms",
                    "Preprocessing and backbone": f"{stages['backbone_ms']:.0f} ms",
                    "Classifier": f"{stages['classifier_ms']:.2f} ms",
                    "Total per video": f"{stages['total_ms']:.0f} ms",
                }
                for key, stages in timing["per_model"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.dataframe(
            [
                {
                    "Backbone": info["name"],
                    "Pretrained on": info["pretrained_on"],
                    "Parameters": f"{info['parameters'] / 1e6:.2f} M",
                    "Size": f"{info['size_mb']:.1f} MB",
                    "Feature size": info["feature_dim"],
                }
                for info in results["feature_extractors"].values()
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Median per video over {timing['videos_timed']} test videos on this machine. The frozen backbone "
            "dominates the cost of every model; the classifiers themselves are negligible."
        )

    st.subheader("How the models were selected")
    st.markdown(
        "Hyperparameters were chosen on the validation split only. The test split was scored after selection and "
        "was not used to choose or tune any model."
    )
    for key, model in models.items():
        training = read_json_if_exists(config.results_dir / key / "training.json")
        if not training:
            continue
        with st.expander(f"{model['display_name']}: validation results for each candidate"):
            st.dataframe(training["candidates"], hide_index=True, width="stretch")
            st.caption(f"Selected by {training['selection_metric']}. Selected settings: {training['selected']}")

    _motion_section(config)


def _motion_section(config) -> None:
    motion = read_json_if_exists(config.results_dir / "motion_experiment.json")
    st.subheader("Optional experiment: adding optical-flow features")
    if motion is None or "decisions" not in motion:
        st.markdown("Not run yet. To run it: `python -m visionact motion-experiment`")
        return
    st.dataframe(
        [
            {
                "Features": entry["display_name"],
                "Input size": entry["input_dim"],
                "Validation accuracy": f"{entry['validation']['val_accuracy']:.1%}",
                "Test accuracy": f"{entry['accuracy']:.1%}",
                "95% interval (test)": _percent_range(entry),
            }
            for entry in motion["models"].values()
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"{motion['description']} Computing the flow descriptor takes about {motion['flow_ms_per_video']:.0f} ms per "
        f"video. Chance level is {1 / len(motion['class_names']):.0%}."
    )
    st.markdown(f"**Decision rule.** {motion['decision_rule']}")
    names = {"cnn": "MobileNetV3 features", "r3d": "R3D-18 features"}
    for name, decision in motion["decisions"].items():
        verdict = "used" if decision["retained"] else "not used"
        st.markdown(f"- With {names.get(name, name)}: flow fusion is **{verdict}**. {decision['reason']}")


def _milliseconds(value: float | None) -> str:
    return f"{value:.0f} ms" if value is not None else "not measured"
