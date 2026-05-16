# ============================================================
# Autoencoder Anomaly Detection — Gradio Demo App
# IEEE IJCNN 2024 Paper: "Autoencoder Optimization for
# Anomaly Detection: A Comparative Study with Shallow Algorithms"
#
# Authors: Vikas Kumar, Vishesh Srivastava,
#          Sadia Mahjabin, Emmanuel Muller
#
# Deploy on HuggingFace Spaces (Gradio SDK)
# ============================================================

import os
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# ============================================================
# Configuration
# ============================================================

HF_REPO_ID = os.getenv("HF_REPO_ID", "VisheshSrivastava/autoencoder-anomaly-detection")

MODEL_CONFIG = {
    "MNIST — Digit '1' (Normal) vs Digit '3' (Anomalous)": {
        "model_file":     "mnist_autoencoder.h5",
        "threshold_file": "mnist_autoencoder_threshold.npy",
        "grayscale":      True,
        "paper_auc":      "0.999",
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
        "loss":           "Binary Cross-Entropy",
        "arch":           "Basic (2-level)",
        "description":    "Trained on street-view digit '1'. Flags other digits as anomalous.",
        "example_normal": "Upload colour images of house-number digit '1'",
        "example_anomaly":"Upload other SVHN digit images",
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
        print(f"Downloading {cfg['model_file']} from HuggingFace Hub …")
        try:
            model_path     = hf_hub_download(repo_id=HF_REPO_ID,
                                             filename=cfg["model_file"],
                                             repo_type="model")
            threshold_path = hf_hub_download(repo_id=HF_REPO_ID,
                                             filename=cfg["threshold_file"],
                                             repo_type="model")
        except Exception:
            raise gr.Error(
                f"⏳ Model for **{dataset_key}** is not yet uploaded. "
                "Please try the **CIFAR-10** model which is available now. "
                "The remaining models will be added shortly!"
            )
        _model_cache[dataset_key]     = load_model(model_path, compile=False)
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

def run_inference(dataset_key: str, uploaded_files):
    """
    Core inference function.
    Returns (fig, table_rows, summary_str).
    """
    if not uploaded_files:
        return None, [], "⚠️  Please upload at least one image."

    model, threshold = get_model(dataset_key)
    cfg = MODEL_CONFIG[dataset_key]

    # ── preprocess ──
    originals, names = [], []
    for f in uploaded_files:
        pil_img = Image.open(f.name if hasattr(f, "name") else f)
        originals.append(preprocess(pil_img, cfg["grayscale"]))
        names.append(os.path.basename(f.name if hasattr(f, "name") else str(f)))

    batch = np.array(originals)                          # (N, 32, 32, 3)
    reconstructions = model.predict(batch, verbose=0)    # (N, 32, 32, 3)

    mae_scores = np.mean(np.abs(reconstructions - batch), axis=(1, 2, 3))

    # ── build table ──
    table_rows = []
    for name, score in zip(names, mae_scores):
        pred = "🔴  Anomalous" if score > threshold else "🟢  Normal"
        table_rows.append([name, f"{score:.5f}", f"{threshold:.5f}", pred])

    n_anomalous = sum(1 for r in table_rows if "Anomalous" in r[3])
    summary = (
        f"**{len(names)} image(s) analysed** using **{dataset_key}** model  |  "
        f"Threshold: `{threshold:.5f}`  |  "
        f"🔴 Anomalous: **{n_anomalous}**  /  🟢 Normal: **{len(names)-n_anomalous}**  |  "
        f"Paper AUC-ROC: **{cfg['paper_auc']}**"
    )

    # ── visualisation ──
    fig = build_figure(names, originals, reconstructions, mae_scores, threshold, dataset_key)

    return fig, table_rows, summary


# ============================================================
# Visualisation
# ============================================================

def build_figure(names, originals, reconstructions, mae_scores, threshold, dataset_key):
    n = len(originals)
    fig = plt.figure(figsize=(13, 4.2 * n), facecolor="#0f0f0f")
    fig.suptitle(
        f"Anomaly Detection — {dataset_key}",
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
**Authors:** Vikas Kumar · **Vishesh Srivastava** · Sadia Mahjabin · Emmanuel Muller

---
### How it works
These autoencoders are trained **only on normal images** (one-class learning).
At inference, the model tries to reconstruct your uploaded image.
If the **Mean Absolute Error (MAE)** of reconstruction is above a learned threshold → flagged as **Anomalous**.
"""

HOW_TO_MD = """
### 📌 How to use
1. **Select a model** from the dropdown (each is tied to a specific dataset from the paper)
2. **Upload one or more images** — PNG / JPG / JPEG accepted
3. Click **Analyse** and view the results
> Images are resized to 32×32 internally to match training. Any resolution is accepted.
"""

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
                    f"| Paper AUC-ROC | **{cfg['paper_auc']}** |\n"
                    f"| Normal example | {cfg['example_normal']} |\n"
                    f"| Anomaly example | {cfg['example_anomaly']} |"
                )

            dataset_dd.change(update_info, inputs=dataset_dd, outputs=dataset_info_md)

            image_upload = gr.File(
                label="② Upload Test Image(s)",
                file_count="multiple",
                file_types=["image"],
            )

            analyse_btn = gr.Button(
                "🔍  Analyse for Anomalies",
                variant="primary",
                size="lg"
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

    # ── Wire up ──
    demo.load(
        fn=update_info,
        inputs=dataset_dd,
        outputs=dataset_info_md
    )

    analyse_btn.click(
        fn=run_inference,
        inputs=[dataset_dd, image_upload],
        outputs=[results_plot, results_table, summary_md],
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
    demo.launch()
