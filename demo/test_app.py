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
from PIL import Image

import app


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
