"""Smoke tests of the Streamlit app, run headlessly with Streamlit's AppTest."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import make_video
from visionact import registry
from visionact.config import PROJECT_ROOT

SECTIONS = [
    "Home",
    "Video Analysis",
    "Image Processing",
    "Motion Analysis",
    "Activity Recognition",
    "Model Comparison",
    "Evaluation",
    "About",
]
MODEL_SECTIONS = ["Activity Recognition", "Model Comparison", "Evaluation"]


def start_app(monkeypatch, config_path) -> AppTest:
    monkeypatch.setenv("VISIONACT_CONFIG", str(config_path))
    for variable in ("VISIONACT_DATASET_DIR", "VISIONACT_ARTIFACTS_DIR", "VISIONACT_RESULTS_DIR"):
        monkeypatch.delenv(variable, raising=False)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=180)
    return app.run()


def open_section(app: AppTest, name: str) -> AppTest:
    app.sidebar.radio(key="section").set_value(name)
    return app.run()


def text_of(elements) -> str:
    return " ".join(str(element.value) for element in elements)


@pytest.fixture
def empty_project(tmp_path):
    """Config for a fresh checkout: no dataset, no features, no models, no results."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({name: str(tmp_path / name) for name in ("dataset_dir", "artifacts_dir", "results_dir")})
    )
    return path


def test_section_list_matches_the_app(monkeypatch, empty_project):
    app = start_app(monkeypatch, empty_project)
    assert not app.exception
    assert list(app.sidebar.radio(key="section").options) == SECTIONS


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders_without_dataset_or_model(monkeypatch, empty_project, section):
    app = open_section(start_app(monkeypatch, empty_project), section)
    assert not app.exception, app.exception
    if section in MODEL_SECTIONS:
        # No model or results exist, so these sections must say so and show no numbers.
        assert app.warning, f"{section} should explain that nothing has been trained"
        assert not app.metric, f"{section} must not display metrics or predictions without a trained model"
    if section in ("Video Analysis", "Image Processing", "Motion Analysis"):
        assert "Choose an input video" in text_of(app.info)


def test_demonstration_sections_work_on_an_uploaded_video(monkeypatch, empty_project, tmp_path):
    clip = make_video(tmp_path / "upload.avi", num_frames=20)
    app = start_app(monkeypatch, empty_project)
    app.sidebar.file_uploader(key="video_upload").set_value(("upload.avi", clip.read_bytes(), "video/x-msvideo"))
    app.run()

    open_section(app, "Video Analysis")
    assert not app.exception, app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Resolution"] == "160 x 120" and metrics["Frames"] == "20"

    open_section(app, "Image Processing")
    assert not app.exception, app.exception
    assert [tab.label for tab in app.tabs] == ["Filtering", "Thresholding", "Morphology", "Edge detection", "Shape analysis"]

    open_section(app, "Motion Analysis")
    assert not app.exception, app.exception
    assert "Mean speed" in {metric.label for metric in app.metric}

    # With a video but still no trained model, no prediction may appear.
    open_section(app, "Activity Recognition")
    assert not app.exception and app.warning and not app.metric and not app.button


def test_mp4_upload_is_read_and_offered_in_a_player(monkeypatch, empty_project, tmp_path):
    clip = make_video(tmp_path / "upload.mp4", num_frames=12, fourcc="mp4v")
    app = start_app(monkeypatch, empty_project)
    app.sidebar.file_uploader(key="video_upload").set_value(("upload.mp4", clip.read_bytes(), "video/mp4"))
    app.run()
    open_section(app, "Video Analysis")
    assert not app.exception, app.exception
    assert {metric.label: metric.value for metric in app.metric}["Frames"] == "12"
    assert "cannot play" not in text_of(app.caption), "MP4 files get a player instead of the AVI notice"
    open_section(app, "Motion Analysis")
    assert not app.exception, app.exception


def test_unreadable_upload_is_reported_not_crashed(monkeypatch, empty_project):
    app = start_app(monkeypatch, empty_project)
    app.sidebar.file_uploader(key="video_upload").set_value(("broken.mp4", b"not a video" * 100, "video/mp4"))
    app.run()
    for section in ("Video Analysis", "Image Processing", "Motion Analysis"):
        open_section(app, section)
        assert not app.exception, app.exception
        assert "cannot be read" in text_of(app.error)


@pytest.mark.slow
def test_model_sections_show_results_from_the_trained_pipeline(monkeypatch, trained):
    config, config_path = trained
    app = start_app(monkeypatch, config_path)
    assert not app.exception, app.exception

    # Pick a clip from the dataset; held-out test clips are listed first.
    app.sidebar.radio(key="video_source").set_value("Dataset clip")
    app.run()
    open_section(app, "Activity Recognition")
    assert not app.exception, app.exception
    assert not app.metric, "no prediction before the button is pressed"
    # Every trained model is offered; the default is the best one on validation.
    choice = app.selectbox(key="model_choice")
    assert len(choice.options) == len(registry.trained_models(config)) == 5
    assert choice.value == registry.best_validated_model(config)
    app.button(key="classify").click()
    app.run()
    assert not app.exception, app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Predicted activity"] in config.classes
    assert metrics["True class (dataset label)"] in config.classes
    assert "held-out test set" in text_of(app.caption)

    # Switching model must not show the previous model's prediction under the new name.
    def classify_with(key: str) -> dict:
        app.selectbox(key="model_choice").set_value(key)
        app.run()
        assert not app.exception and not app.metric, "a stale prediction from another model must not be shown"
        app.button(key="classify").click()
        app.run()
        assert not app.exception, app.exception
        return {metric.label: metric.value for metric in app.metric}

    network = "cnn_lstm" if choice.value != "cnn_lstm" else "r3d18_head"
    metrics = classify_with(network)
    assert metrics["Predicted activity"] in config.classes and "Score of the predicted class" in metrics
    metrics = classify_with("cnn_svm_aug")
    assert metrics["Predicted activity"] in config.classes
    assert "Score of the predicted class" not in metrics, "SVM margins are not presented as a probability"

    app.button(key="compare_all").click()
    app.run()
    assert not app.exception, app.exception
    assert any(len(table.value) == 5 for table in app.dataframe), "one row per trained model"

    open_section(app, "Model Comparison")
    assert not app.exception, app.exception
    assert not app.warning and len(app.dataframe) >= 3
    assert len(app.dataframe[0].value) == 5, "the main table has one row per trained model"
    assert "best on validation" in text_of(app.markdown) + text_of(app.caption)

    open_section(app, "Evaluation")
    assert not app.exception, app.exception
    assert {metric.label for metric in app.metric} >= {"Accuracy", "Macro F1"}
    assert len(app.radio(key="eval_model").options) == 5
    for key in ("cnn_lstm", "r3d18_head"):
        app.radio(key="eval_model").set_value(key)
        app.run()
        assert not app.exception, app.exception
        assert "Training curves" in text_of(app.subheader)

    for section in ("Home", "About"):
        open_section(app, section)
        assert not app.exception, app.exception
