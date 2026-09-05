---
title: Autoencoder Anomaly Detection (IJCNN 2024)
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.23.1"
python_version: "3.11"
app_file: app.py
pinned: true
license: mit
---

# Autoencoder Anomaly Detection Demo

**Paper:** *Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms*
**Venue:** IEEE IJCNN 2024
**DOI:** [10.1109/IJCNN60899.2024.10650057](https://doi.org/10.1109/IJCNN60899.2024.10650057)
**Authors:** Vikas Kumar · Vishesh Srivastava · Sadia Mahjabin · Arindam Pal · Simon Klüttermann · Emmanuel Müller

---

## What This Demo Does

Upload an image and the pre-trained convolutional autoencoder tells you whether it's **normal or anomalous**, based on reconstruction error.

Each autoencoder was trained **only on normal images** (one-class learning). At test time it tries to reconstruct whatever you give it — anomalous images come out with a high Mean Absolute Error (MAE) because the model has never seen anything like them, and that's what gets flagged.

## Available Models

| Model | Normal Class | Anomalous Class | Paper AUC-ROC |
|---|---|---|---|
| MNIST | Digit '1' | Digit '3' | **0.999** |
| Fashion-MNIST | Trousers | Dresses | **0.866** |
| CIFAR-10 | Dogs | Cars | **0.829** |
| SVHN | Digit '1' | Other digits | **0.631** |

SVHN is listed for completeness (it's in the paper) but its checkpoint isn't uploaded yet — the training script hits a `protobuf`/`tensorflow_datasets` version conflict (see GitHub issue #7). Selecting it here will show a "not yet uploaded" message rather than results.

## Architecture

All models use convolutional autoencoders:
- **Basic (2-level):** Conv 32→64 encoder + matching decoder
- **Deep (3-level):** Conv 32→64→128 encoder + matching decoder

Anomaly score = Mean Absolute Error between input and reconstruction.
Threshold = 95th percentile of validation-set normal reconstruction errors.

## Citation

```bibtex
@inproceedings{Kumar2024Autoencoder,
  title={Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms},
  author={Kumar, Vikas and Srivastava, Vishesh and Mahjabin, Sadia and Pal, Arindam and Klüttermann, Simon and Müller, Emmanuel},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2024}
}
```
