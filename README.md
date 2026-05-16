# Autoencoder Optimization for Anomaly Detection

[![IEEE IJCNN 2024](https://img.shields.io/badge/IEEE%20IJCNN-2024-blue?style=flat-square)](https://doi.org/10.1109/IJCNN60899.2024.10650057)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-orange?style=flat-square)](https://huggingface.co/spaces/VisheshSrivastava/autoencoder-anomaly-detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

This repository contains the notebook and supporting notes for the paper **"Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms"**, published at IEEE IJCNN 2024.

---

## 🔍 Live Demo

Try the interactive demo — upload your own images and see the anomaly detector in action:

**👉 [huggingface.co/spaces/VisheshSrivastava/autoencoder-anomaly-detection](https://huggingface.co/spaces/VisheshSrivastava/autoencoder-anomaly-detection)**

---

## Overview

The project studies how latent-space size, loss functions, and training choices affect anomaly detection performance, with a focus on autoencoder-based methods compared against classical shallow baselines.

---

## Repository Contents

- `Image Data.ipynb` — notebook for the image-data experiments in this repository
- `demo/app.py` — Gradio web app for the live HuggingFace demo
- `demo/train_and_upload.py` — script to retrain all models and upload weights to HuggingFace Hub
- `demo/requirements.txt` — dependencies for the demo app
- `README.md` — project overview, usage notes, and citation

---

## Publication

- **Venue:** IEEE IJCNN 2024
- **Title:** "Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms"
- **DOI:** [10.1109/IJCNN60899.2024.10650057](https://doi.org/10.1109/IJCNN60899.2024.10650057)
- **Authors:** Vikas Kumar · Vishesh Srivastava · Sadia Mahjabin · Emmanuel Muller

---

## Highlights

- Compares autoencoder-based anomaly detection against shallow baselines (PCA, LOF, CBLOF, KNN)
- Studies the effect of latent dimension and reconstruction loss function choices
- Evaluates on 5 datasets: CIFAR-10, Fashion-MNIST, MNIST, MVTec-AD, SVHN
- One-class learning paradigm — trained only on normal samples

### Results (AUC-ROC)

| Dataset | Autoencoder | Best Baseline | Baseline Method |
|---|---|---|---|
| MNIST | **0.999** | 0.377 | PCA |
| Fashion-MNIST | **0.866** | 0.560 | PCA |
| CIFAR-10 | **0.829** | 0.740 | LOF |
| SVHN | **0.631** | 0.596 | LOF |
| MVTec-AD | 0.483 | 0.653 | PCA |

---

## How to Use

### Run the notebook locally

1. Clone the repository:
   ```bash
   git clone https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection.git
   ```
2. Create a Python environment with the required libraries:
   ```bash
   pip install numpy pandas tensorflow scikit-learn matplotlib jupyter
   ```
3. Open `Image Data.ipynb` in Jupyter Notebook or JupyterLab
4. Update any dataset paths or environment-specific settings before running all cells

### Run the demo app locally

```bash
cd demo
pip install -r requirements.txt
python app.py
```

---

## Citation

```bibtex
@inproceedings{Kumar2024Autoencoder,
  title={Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms},
  author={Kumar, Vikas and Srivastava, Vishesh and Mahjabin, Sadia and Muller, Emmanuel},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
