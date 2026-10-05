# ============================================================
# Autoencoder Anomaly Detection — Gradio Demo App
# IEEE IJCNN 2024 Paper: "Autoencoder Optimization for
# Anomaly Detection: A Comparative Study with Shallow Algorithms"
#
# Authors: Vikas Kumar, Vishesh Srivastava, Sadia Mahjabin,
#          Arindam Pal, Simon Klüttermann, Emmanuel Müller
#
# Deploy on HuggingFace Spaces (Gradio SDK)
# ============================================================

import csv
import os
import inspect
import tempfile
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

# ============================================================
# Configuration
# ============================================================

HF_REPO_ID = os.getenv("HF_REPO_ID", "VisheshSrivastava/autoencoder-anomaly-detection")

# Pin the model checkpoints to a known-good commit of the Hub repo instead of
# always resolving "main", so a compromised/mutated Hub repo can't silently
# swap in a different .h5 file for every Space instance on next cold start.
# Update this SHA (and re-verify the checkpoints) whenever new weights are
# intentionally published.
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "301025cba5eca10e7f3ce82325ce3298ef5eb8e8")

# ── resource-exhaustion guardrails for the shared CPU-only Space ──
MAX_UPLOAD_FILES   = 15          # max images accepted per Analyse click
MAX_FILE_SIZE_MB   = 10          # max size per uploaded file
MAX_PLOT_IMAGES    = 8           # cap on how many rows the result figure renders

MODEL_CONFIG = {
    "MNIST — Digit '1' (Normal) vs Digit '3' (Anomalous)": {
        "model_file":     "mnist_autoencoder.h5",
        "threshold_file": "mnist_autoencoder_threshold.npy",
        "grayscale":      True,
        "paper_auc":      "0.999",
        "baseline_auc":    "0.377",
        "baseline_method": "PCA",
        "loss":           "Binary Cross-Entropy",
        "arch":           "Deep (3-level)",
        "description":    "Trained on handwritten digit '1'. Flags digit '3' as anomalous.",
        "example_normal": "Upload images of digit '1' (white on black background)",
        "example_anomaly":"Upload images of digit '3' to see them flagged",
    },
    "Fashion-MNIST — Trousers (Normal) vs Dresses (Anomalous)": {
        "model_file":     "fashion_mnist_autoencoder.h5",
        "threshold_file": "fashion_mnist_autoencoder_threshold.npy",
        "grayscale":      True,
        "paper_auc":      "0.866",
        "baseline_auc":    "0.560",
        "baseline_method": "PCA",
        "loss":           "MSE",
        "arch":           "Basic (2-level)",
        "description":    "Trained on trousers. Flags dresses as anomalous.",
        "example_normal": "Upload grayscale clothing images (trouser-like)",
        "example_anomaly":"Upload images of dresses",
    },
    "CIFAR-10 — Dogs (Normal) vs Cars (Anomalous)": {
        "model_file":     "cifar10_autoencoder.h5",
        "threshold_file": "cifar10_autoencoder_threshold.npy",
        "grayscale":      False,
        "paper_auc":      "0.829",
        "baseline_auc":    "0.740",
        "baseline_method": "LOF",
        "loss":           "MSE",
        "arch":           "Basic (2-level)",
        "description":    "Trained on dog images. Flags cars as anomalous.",
        "example_normal": "Upload 32x32 colour images of dogs",
        "example_anomaly":"Upload images of cars",
    },
    "SVHN — Digit '1' (Normal) vs Others (Anomalous)": {
        "model_file":     "svhn_autoencoder.h5",
        "threshold_file": "svhn_autoencoder_threshold.npy",
        "grayscale":      False,
        "paper_auc":      "0.631",
        "baseline_auc":    "0.596",
        "baseline_method": "LOF",
        "loss":           "Binary Cross-Entropy",
        "arch":           "Basic (2-level)",
        "description":    "Trained on street-view digit '1'. Flags other digits as anomalous.",
        "example_normal": "Upload colour images of house-number digit '1'",
        "example_anomaly":"Upload other SVHN digit images",
    },
    # MVTec-AD (issue #3): the paper reports this dataset, and it's the only
    # one where the shallow baseline (PCA) *beats* the autoencoder — an
    # interesting, honest negative result worth surfacing in the demo. The
    # code path below is fully wired up (model-selection, preprocessing,
    # inference, UI note), but there are no trained weights to serve it:
    # MVTec-AD requires accepting a license to download
    # (https://www.mvtec.com/company/research/datasets/mvtec-ad) and the
    # repo has no local copy, so this entry is marked unavailable rather
    # than silently failing a HuggingFace Hub lookup. Once someone trains
    # and uploads `mvtec_ad_autoencoder.h5` / `_threshold.npy` to the HF
    # Hub repo, flip `available` to True (no other code changes needed).
    "MVTec-AD — Industrial Defects (requires licensed dataset)": {
        "model_file":     "mvtec_ad_autoencoder.h5",
        "threshold_file": "mvtec_ad_autoencoder_threshold.npy",
        "grayscale":      False,
        "paper_auc":      "0.483",
        "baseline_auc":    "0.653",
        "baseline_method": "PCA",
        "loss":           "MSE",
        "arch":           "Basic (2-level)",
        "description":    (
            "Industrial defect detection (e.g. hazelnut/bottle/carpet categories). "
            "⚠️ Unlike every other dataset here, PCA beats the autoencoder on "
            "MVTec-AD in the paper — likely because resizing to 32×32 throws away "
            "the fine defect texture the model needs."
        ),
        "example_normal": "N/A — weights not trained/uploaded yet (see note above)",
        "example_anomaly":"N/A — weights not trained/uploaded yet (see note above)",
        "available": False,
        "unavailable_reason": (
            "MVTec-AD requires agreeing to a license to download "
            "(mvtec.com/company/research/datasets/mvtec-ad) and is not bundled "
            "in this repo. Training also needs the categories decided (e.g. "
            "hazelnut, bottle, carpet) and, ideally, a higher input resolution "
            "than the 32×32 used elsewhere to preserve defect texture. Once "
            "trained, upload `mvtec_ad_autoencoder.h5` + `_threshold.npy` to "
            "the HF Hub model repo and set `available: True` here."
        ),
    },
}

# ── in-memory model cache so we don't reload on every request ──
_model_cache     = {}
_threshold_cache = {}


# ============================================================
# Model loading
# ============================================================

def get_model(dataset_key: str):
    if dataset_key not in _model_cache:
        cfg = MODEL_CONFIG[dataset_key]

        # Some entries (e.g. MVTec-AD, issue #3) are wired up end-to-end in
        # the UI/preprocessing but have no trained weights to serve yet.
        # Fail fast with the real reason instead of spending a network
        # round-trip on a HuggingFace Hub lookup that will 404 anyway.
        if not cfg.get("available", True):
            raise gr.Error(
                f"⏳ **{dataset_key}** isn't available yet.\n\n"
                + cfg.get("unavailable_reason", "No trained weights for this dataset yet.")
            )

        print(f"Downloading {cfg['model_file']} from HuggingFace Hub …")
        try:
            model_path     = hf_hub_download(repo_id=HF_REPO_ID,
                                             filename=cfg["model_file"],
                                             repo_type="model",
                                             revision=HF_MODEL_REVISION)
            threshold_path = hf_hub_download(repo_id=HF_REPO_ID,
                                             filename=cfg["threshold_file"],
                                             repo_type="model",
                                             revision=HF_MODEL_REVISION)
        except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
            # The checkpoint genuinely isn't present on the Hub yet (or the
            # pinned repo/revision doesn't exist) — this is the one case
            # where "not yet uploaded" is an accurate message.
            raise gr.Error(
                f"⏳ Model for **{dataset_key}** is not yet uploaded. "
                "Please try the **CIFAR-10** model which is available now. "
                "The remaining models will be added shortly!"
            )
        except Exception as e:
            # Anything else (network timeout, HF Hub outage, rate limiting,
            # auth failure, etc.) is a transient/infra problem, not a missing
            # checkpoint — log the real exception and tell the user it's
            # retry-able instead of implying the model doesn't exist.
            print(f"  Error downloading {cfg['model_file']} for {dataset_key}: "
                  f"{e.__class__.__name__}: {e}")
            raise gr.Error(
                f"⚠️ Could not reach the model repository for **{dataset_key}** "
                "right now. Please retry in a moment."
            )
        # Legacy HDF5 (.h5) Keras checkpoints can embed Lambda/custom-object
        # layers that execute arbitrary Python on deserialization. Newer
        # Keras load_model() implementations accept `safe_mode` to refuse
        # that; pass it only if the installed Keras version actually
        # supports it (it is a Keras-3-only argument, and is a no-op for
        # the legacy .h5 format even there — the real protection is the
        # revision pin above, which stops an untrusted/mutated Hub repo
        # from swapping the file out from under us).
        load_kwargs = {"compile": False}
        if "safe_mode" in inspect.signature(load_model).parameters:
            load_kwargs["safe_mode"] = True
        _model_cache[dataset_key]     = load_model(model_path, **load_kwargs)
        _threshold_cache[dataset_key] = float(np.load(threshold_path))
        print(f"  Loaded. Threshold = {_threshold_cache[dataset_key]:.5f}")

    return _model_cache[dataset_key], _threshold_cache[dataset_key]


# ============================================================
# Image preprocessing  (must match training exactly)
# ============================================================

def preprocess(pil_image: Image.Image, grayscale: bool) -> np.ndarray:
    """
    Convert a PIL image to a model-ready (32,32,3) float32 array
    using the same pipeline as training.
    """
    if grayscale:
        img = pil_image.convert("L")      # grayscale
        img = img.convert("RGB")          # repeat to 3 channels (L,L,L)
    else:
        img = pil_image.convert("RGB")

    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.array(img, dtype="float32") / 255.0
    return arr                            # (32, 32, 3)


# ============================================================
# Inference
# ============================================================

def classify_rows(names, mae_scores, threshold):
    """Turn (names, mae_scores) into the result table rows + anomaly count,
    given a decision threshold. Pure function of already-computed scores, so
    it can be re-run cheaply whenever the threshold slider moves without
    re-running model.predict()."""
    table_rows = []
    for name, score in zip(names, mae_scores):
        pred = "🔴  Anomalous" if score > threshold else "🟢  Normal"
        table_rows.append([name, f"{score:.5f}", f"{threshold:.5f}", pred])
    n_anomalous = sum(1 for r in table_rows if "Anomalous" in r[3])
    return table_rows, n_anomalous


def build_summary(dataset_key, names, threshold, n_anomalous, skipped, cfg):
    summary = (
        f"**{len(names)} image(s) analysed** using **{dataset_key}** model  |  "
        f"Threshold: `{threshold:.5f}`  |  "
        f"🔴 Anomalous: **{n_anomalous}**  /  🟢 Normal: **{len(names)-n_anomalous}**  |  "
        f"Paper AUC-ROC: **{cfg['paper_auc']}** (autoencoder) vs **{cfg['baseline_auc']}** "
        f"({cfg['baseline_method']}, best shallow baseline)"
    )
    if skipped:
        summary += "\n\n⚠️ **Skipped unreadable file(s):** " + "; ".join(skipped)
    return summary


def run_inference(dataset_key: str, uploaded_files):
    """
    Core inference function.
    Returns (fig, table_rows, summary_str, cache, slider_update).

    `cache` is a plain dict (JSON-serialisable, suitable for gr.State) holding
    everything needed to re-classify/re-render at a different threshold
    without re-running model.predict() — see reclassify() and export_csv().
    `slider_update` re-centers the threshold slider on this run's default
    threshold/dataset so it stays in sync with whichever model was just used.
    """
    if not uploaded_files:
        return None, [], "⚠️  Please upload at least one image.", None, gr.update()

    # ── resource-exhaustion guardrail: cap number of files per request ──
    if len(uploaded_files) > MAX_UPLOAD_FILES:
        raise gr.Error(
            f"⚠️ Too many files ({len(uploaded_files)}). "
            f"Please upload at most {MAX_UPLOAD_FILES} images per request."
        )

    model, threshold = get_model(dataset_key)
    cfg = MODEL_CONFIG[dataset_key]

    # ── preprocess (skip, don't abort on, bad/oversized/corrupt files) ──
    originals, names, skipped = [], [], []
    for f in uploaded_files:
        path = f.name if hasattr(f, "name") else f
        display_name = os.path.basename(path if isinstance(path, str) else str(f))
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                skipped.append(f"{display_name} (too large: {size_mb:.1f} MB)")
                continue
            pil_img = Image.open(path)
            pil_img.load()   # force decode now so truncated files raise here
            originals.append(preprocess(pil_img, cfg["grayscale"]))
            names.append(display_name)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            skipped.append(f"{display_name} ({e.__class__.__name__})")
            continue

    if not originals:
        raise gr.Error(
            "⚠️ None of the uploaded files could be read as valid images: "
            + "; ".join(skipped)
        )

    batch = np.array(originals)                          # (N, 32, 32, 3)
    reconstructions = model.predict(batch, verbose=0)    # (N, 32, 32, 3)

    mae_scores = np.mean(np.abs(reconstructions - batch), axis=(1, 2, 3))

    table_rows, n_anomalous = classify_rows(names, mae_scores, threshold)
    summary = build_summary(dataset_key, names, threshold, n_anomalous, skipped, cfg)

    # ── visualisation ──
    fig = build_figure(names, originals, reconstructions, mae_scores, threshold, dataset_key)
    # Release pyplot's global reference to the figure now that it has been
    # rendered; the returned Figure object (already drawn on the Agg canvas)
    # is unaffected and still renders fine in gr.Plot. Without this, every
    # Analyse click leaks one more Figure on this long-running process (#18).
    plt.close(fig)

    cache = {
        "dataset_key":  dataset_key,
        "names":        names,
        "originals":    [o.tolist() for o in originals],
        "reconstructions": reconstructions.tolist(),
        "mae_scores":   mae_scores.tolist(),
        "default_threshold": threshold,
        "skipped":      skipped,
        "table_rows":   table_rows,
    }

    slider_update = gr.update(
        minimum=round(threshold * 0.2, 5),
        maximum=round(threshold * 3.0, 5),
        value=threshold,
        visible=True,
    )

    return fig, table_rows, summary, cache, slider_update


def reclassify(threshold, cache):
    """Re-classify the already-computed MAE scores at a new threshold and
    re-render, without calling model.predict() again (issue #13)."""
    if not cache or not cache.get("names"):
        return gr.update(), [], "⚠️ Click **Analyse** first, then drag the slider.", cache

    names           = cache["names"]
    originals       = [np.array(o, dtype="float32") for o in cache["originals"]]
    reconstructions = np.array(cache["reconstructions"], dtype="float32")
    mae_scores      = np.array(cache["mae_scores"], dtype="float32")
    dataset_key     = cache["dataset_key"]
    cfg             = MODEL_CONFIG[dataset_key]

    table_rows, n_anomalous = classify_rows(names, mae_scores, threshold)
    summary = build_summary(dataset_key, names, threshold, n_anomalous, cache.get("skipped", []), cfg)

    fig = build_figure(names, originals, reconstructions, mae_scores, threshold, dataset_key)
    plt.close(fig)

    cache = dict(cache)
    cache["table_rows"] = table_rows

    return fig, table_rows, summary, cache


def export_csv(cache):
    """Write the current result table to a temp CSV and hand its path to a
    gr.DownloadButton, so users can keep a batch's results (issue #14)."""
    if not cache or not cache.get("table_rows"):
        raise gr.Error("⚠️ Click **Analyse** first, then download the results.")

    fd, path = tempfile.mkstemp(suffix=".csv", prefix="anomaly_results_")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "MAE Score", "Threshold", "Prediction"])
        writer.writerows(cache["table_rows"])
    return path


# ============================================================
# Visualisation
# ============================================================

def build_figure(names, originals, reconstructions, mae_scores, threshold, dataset_key):
    n_total = len(originals)
    # Cap how many rows we render — figure height (and matplotlib CPU/memory
    # cost) scales linearly with n, so an unbounded batch could blow both up.
    n = min(n_total, MAX_PLOT_IMAGES)
    fig = plt.figure(figsize=(13, 4.2 * n), facecolor="#0f0f0f")
    title = f"Anomaly Detection — {dataset_key}"
    if n_total > n:
        title += f"  (showing first {n} of {n_total} images)"
    fig.suptitle(
        title,
        fontsize=13, fontweight="bold", color="white", y=1.002
    )

    for i in range(n):
        gs = gridspec.GridSpecFromSubplotSpec(
            1, 4,
            subplot_spec=gridspec.GridSpec(n, 1, figure=fig)[i],
            wspace=0.05
        )

        orig  = originals[i]
        recon = np.clip(reconstructions[i], 0, 1)
        diff  = np.abs(orig - recon)
        diff_norm = diff / (diff.max() + 1e-8)

        is_anomaly = mae_scores[i] > threshold
        colour     = "#ff4b4b" if is_anomaly else "#00cc88"
        label      = "🔴 ANOMALOUS" if is_anomaly else "🟢 NORMAL"

        titles = [
            f"Original\n{names[i]}",
            "Reconstruction",
            "Error Map (hot)",
            f"Score: {mae_scores[i]:.5f}\n{label}"
        ]
        images = [orig, recon, diff_norm, diff_norm]
        cmaps  = [None, None, "hot", "hot"]

        for j in range(4):
            ax = fig.add_subplot(gs[j])
            if j < 3:
                ax.imshow(images[j], cmap=cmaps[j], vmin=0, vmax=1)
            else:
                # Score panel
                ax.set_facecolor(colour + "22")
                ax.text(0.5, 0.6, f"{mae_scores[i]:.5f}",
                        ha="center", va="center",
                        fontsize=16, fontweight="bold", color=colour,
                        transform=ax.transAxes)
                ax.text(0.5, 0.3, label,
                        ha="center", va="center",
                        fontsize=11, color=colour,
                        transform=ax.transAxes)
                ax.text(0.5, 0.12,
                        f"threshold: {threshold:.5f}",
                        ha="center", va="center",
                        fontsize=8, color="#aaaaaa",
                        transform=ax.transAxes)
                for spine in ax.spines.values():
                    spine.set_edgecolor(colour)
                    spine.set_linewidth(2)

            ax.set_title(titles[j], fontsize=8, color="white", pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout(pad=1.5)
    return fig


# ============================================================
# Gradio UI
# ============================================================

PAPER_MD = """
# 🔍 Autoencoder Anomaly Detection Demo

**Paper:** *Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms*
**Venue:** IEEE IJCNN 2024 &nbsp;|&nbsp; **DOI:** [10.1109/IJCNN60899.2024.10650057](https://doi.org/10.1109/IJCNN60899.2024.10650057)
**Authors:** Vikas Kumar · **Vishesh Srivastava** · Sadia Mahjabin · Arindam Pal · Simon Klüttermann · Emmanuel Müller

---
### How it works
These autoencoders are trained **only on normal images** (one-class learning).
At inference, the model tries to reconstruct your uploaded image.
If the **Mean Absolute Error (MAE)** of reconstruction is above a learned threshold → flagged as **Anomalous**.
"""

HOW_TO_MD = """
### 📌 How to use
1. **Select a model** from the dropdown (each is tied to a specific dataset from the paper)
2. **Upload one or more images** — PNG / JPG / JPEG accepted (or click a sample below)
3. Click **Analyse** and view the results
4. Drag the **threshold slider** to see how the normal/anomalous call changes
5. **Download the results as CSV** to keep a record of a batch
> Images are resized to 32×32 internally to match training. Any resolution is accepted.
"""

# ── sample images (issue #4): a couple of normal/anomalous PNGs per
# dataset, extracted ahead of time from each dataset's real test split, so
# visitors can try the demo with one click instead of finding their own
# images first. Keyed by the same dataset_key used in MODEL_CONFIG so the
# examples can be looked up per-model.
_SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

SAMPLE_IMAGES = {
    "MNIST — Digit '1' (Normal) vs Digit '3' (Anomalous)": [
        os.path.join(_SAMPLE_DIR, "mnist", "normal_digit1_1.png"),
        os.path.join(_SAMPLE_DIR, "mnist", "normal_digit1_2.png"),
        os.path.join(_SAMPLE_DIR, "mnist", "anomaly_digit3_1.png"),
        os.path.join(_SAMPLE_DIR, "mnist", "anomaly_digit3_2.png"),
    ],
    "Fashion-MNIST — Trousers (Normal) vs Dresses (Anomalous)": [
        os.path.join(_SAMPLE_DIR, "fashion_mnist", "normal_trouser_1.png"),
        os.path.join(_SAMPLE_DIR, "fashion_mnist", "normal_trouser_2.png"),
        os.path.join(_SAMPLE_DIR, "fashion_mnist", "anomaly_dress_1.png"),
        os.path.join(_SAMPLE_DIR, "fashion_mnist", "anomaly_dress_2.png"),
    ],
    "CIFAR-10 — Dogs (Normal) vs Cars (Anomalous)": [
        os.path.join(_SAMPLE_DIR, "cifar10", "normal_dog_1.png"),
        os.path.join(_SAMPLE_DIR, "cifar10", "normal_dog_2.png"),
        os.path.join(_SAMPLE_DIR, "cifar10", "anomaly_car_1.png"),
        os.path.join(_SAMPLE_DIR, "cifar10", "anomaly_car_2.png"),
    ],
    "SVHN — Digit '1' (Normal) vs Others (Anomalous)": [
        os.path.join(_SAMPLE_DIR, "svhn", "normal_digit1_1.png"),
        os.path.join(_SAMPLE_DIR, "svhn", "normal_digit1_2.png"),
        os.path.join(_SAMPLE_DIR, "svhn", "anomaly_other_1.png"),
        os.path.join(_SAMPLE_DIR, "svhn", "anomaly_other_2.png"),
    ],
}


def load_sample_images(dataset_key):
    """Populate the file-upload component with this model's bundled sample
    images (both normal and anomalous) so a visitor can Analyse immediately."""
    paths = [p for p in SAMPLE_IMAGES.get(dataset_key, []) if os.path.exists(p)]
    return paths

with gr.Blocks(
    title="Autoencoder Anomaly Detection — IJCNN 2024",
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        .gr-button-primary { background: #4f46e5 !important; }
        .result-table tbody tr td { font-family: monospace; font-size: 0.85rem; }
    """
) as demo:

    gr.Markdown(PAPER_MD)

    with gr.Row(equal_height=False):

        # ── Left panel ──
        with gr.Column(scale=1, min_width=340):
            gr.Markdown(HOW_TO_MD)

            dataset_dd = gr.Dropdown(
                choices=list(MODEL_CONFIG.keys()),
                value=list(MODEL_CONFIG.keys())[0],
                label="① Select Dataset Model",
                info="Pre-trained model from the paper"
            )

            dataset_info_md = gr.Markdown()

            def update_info(key):
                cfg = MODEL_CONFIG[key]
                return (
                    f"> **{cfg['description']}**\n\n"
                    f"| | |\n|---|---|\n"
                    f"| Architecture | {cfg['arch']} |\n"
                    f"| Loss function | {cfg['loss']} |\n"
                    f"| Paper AUC-ROC (autoencoder) | **{cfg['paper_auc']}** |\n"
                    f"| Best shallow baseline ({cfg['baseline_method']}) | {cfg['baseline_auc']} |\n"
                    f"| Normal example | {cfg['example_normal']} |\n"
                    f"| Anomaly example | {cfg['example_anomaly']} |"
                )

            dataset_dd.change(update_info, inputs=dataset_dd, outputs=dataset_info_md)

            image_upload = gr.File(
                label="② Upload Test Image(s)",
                file_count="multiple",
                file_types=["image"],
            )

            sample_btn = gr.Button(
                "🖼️  Load sample images for this model",
                size="sm"
            )

            analyse_btn = gr.Button(
                "🔍  Analyse for Anomalies",
                variant="primary",
                size="lg"
            )

            threshold_slider = gr.Slider(
                label="③ Decision threshold (reconstruction MAE)",
                info="Move this to re-classify the already-computed scores live — "
                     "no need to re-run Analyse.",
                minimum=0.0, maximum=1.0, value=0.5, step=0.0005,
                visible=False,
            )

            export_btn = gr.DownloadButton(
                "⬇️  Download results (CSV)",
                visible=False,
            )

            gr.Markdown(
                "<br><sub>⚡ First run may take ~30 s while the model downloads. "
                "Subsequent runs are fast (model is cached).</sub>"
            )

        # ── Right panel ──
        with gr.Column(scale=2):
            summary_md = gr.Markdown(
                value="_Results will appear here after you click Analyse._"
            )
            results_table = gr.Dataframe(
                headers=["Filename", "MAE Score", "Threshold", "Prediction"],
                label="Detection Results",
                wrap=True,
                interactive=False,
                row_count=(1, "dynamic"),
            )
            results_plot = gr.Plot(
                label="Visualisation  (Original | Reconstruction | Error Map | Score)"
            )

    # ── cache of the last Analyse run's scores, shared between the
    # threshold slider (#13) and the CSV export button (#14) so neither
    # has to re-run model.predict() ──
    results_cache = gr.State(value=None)

    # ── Wire up ──
    demo.load(
        fn=update_info,
        inputs=dataset_dd,
        outputs=dataset_info_md
    )

    sample_btn.click(
        fn=load_sample_images,
        inputs=dataset_dd,
        outputs=image_upload,
    )

    analyse_btn.click(
        fn=run_inference,
        inputs=[dataset_dd, image_upload],
        outputs=[results_plot, results_table, summary_md, results_cache, threshold_slider],
    ).then(
        # Once a run has produced a cache, reveal the export button.
        fn=lambda cache: gr.update(visible=bool(cache)),
        inputs=results_cache,
        outputs=export_btn,
    )

    threshold_slider.release(
        fn=reclassify,
        inputs=[threshold_slider, results_cache],
        outputs=[results_plot, results_table, summary_md, results_cache],
    )

    export_btn.click(
        fn=export_csv,
        inputs=results_cache,
        outputs=export_btn,
    )

    gr.Markdown(
        "---\n"
        "<div style='text-align:center; color:#666; font-size:0.8rem'>"
        "Model weights hosted on "
        "<a href='https://huggingface.co/" + HF_REPO_ID + "' target='_blank'>HuggingFace Hub</a> · "
        "<a href='https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection' target='_blank'>GitHub</a> · "
        "MIT License"
        "</div>"
    )

if __name__ == "__main__":
    # Bound how much concurrent inference work the shared CPU-only Space can
    # be pushed into at once (see issue: unbounded uploads / no queueing).
    demo.queue(max_size=20, default_concurrency_limit=2)
    demo.launch()
