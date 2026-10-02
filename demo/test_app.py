# ============================================================
# Minimal smoke tests for the Gradio demo app.
#
# These deliberately avoid any HuggingFace Hub / network access
# (no model download, no inference) so they run fast and offline
# in CI. They cover the two things most likely to silently break
# on a refactor: the image preprocessing pipeline (must keep
# matching what the models were trained on) and the shape of
# MODEL_CONFIG (the UI iterates over it and assumes every entry
# has the same keys).
# ============================================================

import numpy as np
import pytest
import gradio as gr
from PIL import Image

import app


class _DummyModel:
    """Stand-in for a loaded Keras model: reconstructs the input as all-zeros
    so run_inference's guardrail logic can be exercised without any
    HuggingFace Hub access or real inference."""

    def predict(self, batch, verbose=0):
        return np.zeros_like(batch)


def _patch_get_model(monkeypatch, threshold=0.5):
    """Avoid any network/model-download access by making get_model() return
    a dummy model + threshold directly."""
    monkeypatch.setattr(app, "get_model", lambda dataset_key: (_DummyModel(), threshold))


def _make_image_file(tmp_path, name="valid.png", size=(10, 10)):
    path = tmp_path / name
    Image.new("RGB", size, color=(1, 2, 3)).save(path)
    return str(path)


def _make_oversized_file(tmp_path, name="big.png"):
    path = tmp_path / name
    # Write more than MAX_FILE_SIZE_MB of raw bytes. Content doesn't matter —
    # the size guardrail checks os.path.getsize() before ever opening it.
    with open(path, "wb") as f:
        f.write(b"0" * int((app.MAX_FILE_SIZE_MB + 1) * 1024 * 1024))
    return str(path)


def _make_corrupt_file(tmp_path, name="corrupt.png"):
    path = tmp_path / name
    # A plain text file saved with a .png extension: Image.open()/load()
    # will fail to decode it.
    with open(path, "w") as f:
        f.write("this is not an image")
    return str(path)


DATASET_KEY = "MNIST — Digit '1' (Normal) vs Digit '3' (Anomalous)"


def test_preprocess_grayscale_shape_and_range():
    img = Image.new("L", (10, 10), color=128)
    out = app.preprocess(img, grayscale=True)

    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_color_shape_and_range():
    img = Image.new("RGB", (64, 64), color=(10, 20, 30))
    out = app.preprocess(img, grayscale=False)

    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_grayscale_channels_are_equal():
    # A grayscale image repeated to 3 channels should have R == G == B
    # everywhere, since there's no color information to begin with.
    img = Image.new("L", (10, 10), color=200)
    out = app.preprocess(img, grayscale=True)

    assert np.allclose(out[..., 0], out[..., 1])
    assert np.allclose(out[..., 1], out[..., 2])


def test_model_config_entries_have_consistent_keys():
    required_keys = {
        "model_file", "threshold_file", "grayscale", "paper_auc",
        "baseline_auc", "baseline_method", "loss", "arch",
        "description", "example_normal", "example_anomaly",
    }
    assert len(app.MODEL_CONFIG) > 0
    for dataset_key, cfg in app.MODEL_CONFIG.items():
        missing = required_keys - cfg.keys()
        assert not missing, f"{dataset_key} is missing keys: {missing}"


def test_run_inference_with_no_files_returns_warning():
    fig, rows, summary = app.run_inference("MNIST — Digit '1' (Normal) vs Digit '3' (Anomalous)", [])
    assert fig is None
    assert rows == []
    assert "upload at least one image" in summary.lower()


def test_run_inference_too_many_files_raises_error():
    # Fake paths are fine: the MAX_UPLOAD_FILES check happens before any
    # file is opened or the model is loaded.
    fake_files = [f"fake_{i}.png" for i in range(app.MAX_UPLOAD_FILES + 1)]
    with pytest.raises(gr.Error) as excinfo:
        app.run_inference(DATASET_KEY, fake_files)
    assert "too many" in str(excinfo.value).lower()


def test_run_inference_skips_oversized_file(tmp_path, monkeypatch):
    _patch_get_model(monkeypatch)
    valid = _make_image_file(tmp_path, "valid.png")
    oversized = _make_oversized_file(tmp_path, "big.png")

    fig, rows, summary = app.run_inference(DATASET_KEY, [valid, oversized])

    assert fig is not None
    assert len(rows) == 1
    assert rows[0][0] == "valid.png"
    assert "too large" in summary.lower()
    assert "big.png" in summary


def test_run_inference_skips_corrupt_file(tmp_path, monkeypatch):
    _patch_get_model(monkeypatch)
    valid = _make_image_file(tmp_path, "valid.png")
    corrupt = _make_corrupt_file(tmp_path, "corrupt.png")

    fig, rows, summary = app.run_inference(DATASET_KEY, [valid, corrupt])

    assert fig is not None
    assert len(rows) == 1
    assert rows[0][0] == "valid.png"
    assert "corrupt.png" in summary
    assert "skipped" in summary.lower()


def test_run_inference_all_files_unreadable_raises_error(tmp_path, monkeypatch):
    _patch_get_model(monkeypatch)
    corrupt1 = _make_corrupt_file(tmp_path, "corrupt1.png")
    corrupt2 = _make_corrupt_file(tmp_path, "corrupt2.png")

    with pytest.raises(gr.Error) as excinfo:
        app.run_inference(DATASET_KEY, [corrupt1, corrupt2])
    assert "none of the uploaded files" in str(excinfo.value).lower()
