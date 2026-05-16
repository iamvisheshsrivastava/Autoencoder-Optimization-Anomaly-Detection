---
title: Autoencoder Anomaly Detection (IJCNN 2024)
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
---

# Autoencoder Anomaly Detection Demo

**Paper:** *Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms*
**Venue:** IEEE IJCNN 2024
**DOI:** [10.1109/IJCNN60899.2024.10650057](https://doi.org/10.1109/IJCNN60899.2024.10650057)
**Authors:** Vikas Kumar · Vishesh Srivastava · Sadia Mahjabin · Emmanuel Muller

---

## What This Demo Does

Upload an image and the pre-trained convolutional autoencoder will tell you whether it is **normal or anomalous**, based on reconstruction error.

The autoencoders are trained **only on normal images** (one-class learning paradigm). At test time, anomalous images produce a high Mean Absolute Error (MAE) because the model has never seen them — flagged as anomalous.

## Available Models

| Model | Normal Class | Anomalous Class | Paper AUC-ROC |
|---|---|---|---|
| MNIST | Digit '1' | Digit '3' | **0.999** |
| Fashion-MNIST | Trousers | Dresses | **0.866** |
| CIFAR-10 | Dogs | Cars | **0.829** |
| SVHN | Digit '1' | Other digits | **0.631** |

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
  author={Kumar, Vikas and Srivastava, Vishesh and Mahjabin, Sadia and Muller, Emmanuel},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2024}
}
```
