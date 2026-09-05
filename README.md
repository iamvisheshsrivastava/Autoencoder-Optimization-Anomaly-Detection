# Autoencoder Optimization for Anomaly Detection

[![IEEE IJCNN 2024](https://img.shields.io/badge/IEEE%20IJCNN-2024-blue?style=flat-square)](https://doi.org/10.1109/IJCNN60899.2024.10650057)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-orange?style=flat-square)](https://huggingface.co/spaces/VisheshSrivastava/autoencoder-anomaly-detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

This repository holds the notebook and supporting code behind our paper, **"Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms"**, presented at IEEE IJCNN 2024.

## Live Demo

**[huggingface.co/spaces/VisheshSrivastava/autoencoder-anomaly-detection](https://huggingface.co/spaces/VisheshSrivastava/autoencoder-anomaly-detection)** — upload an image and see the trained autoencoder flag it normal or anomalous, with the reconstruction error map alongside it.

## What's here

We ask how far a plain convolutional autoencoder gets on one-class anomaly detection compared to classical shallow methods (PCA, LOF, CBLOF, KNN), and how much latent-space size and reconstruction loss choice matter along the way. Everything is trained in the one-class setting — only normal samples during training, anomalies show up only at test time as reconstruction failures.

### Results (AUC-ROC)

| Dataset | Autoencoder | Best Baseline | Baseline Method |
|---|---|---|---|
| MNIST | **0.999** | 0.377 | PCA |
| Fashion-MNIST | **0.866** | 0.560 | PCA |
| CIFAR-10 | **0.829** | 0.740 | LOF |
| SVHN | **0.631** | 0.596 | LOF |
| MVTec-AD | 0.483 | 0.653 | PCA |

MVTec-AD is the interesting exception — PCA beats the autoencoder there, most likely because resizing the industrial defect images down to 32×32 throws away exactly the fine texture the model needs to catch small anomalies. We left that result in rather than sweep it under the rug; it's a real finding about where this architecture stops working.

The live demo currently serves the MNIST, Fashion-MNIST, and CIFAR-10 checkpoints. SVHN is trained on paper but not uploaded yet — the training script hits a `protobuf`/`tensorflow_datasets` version conflict that needs a clean environment to sort out (tracked in [#7](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/7)).

## Repository contents

- `Image Data.ipynb` — the experiments notebook: training, evaluation, and baseline comparisons
- `demo/app.py` — the Gradio app behind the HuggingFace Space
- `demo/README_HuggingFace.md` — Space config/README (deployed as the Space's `README.md`)
- `demo/train_and_upload.py`, `demo/train_remaining_models.py` — retrain a model and push its weights + threshold to the HF Hub
- `demo/requirements.txt` — pinned dependencies for the demo app
- `.github/workflows/keep-space-awake.yml` — pings the Space every 20 minutes so it doesn't fall asleep on HF's free tier

## Publication

- **Venue:** IEEE IJCNN 2024
- **Title:** "Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms"
- **DOI:** [10.1109/IJCNN60899.2024.10650057](https://doi.org/10.1109/IJCNN60899.2024.10650057)
- **Authors:** Vikas Kumar · Vishesh Srivastava · Sadia Mahjabin · Arindam Pal · Simon Klüttermann · Emmanuel Müller

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

## Citation

```bibtex
@inproceedings{Kumar2024Autoencoder,
  title={Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms},
  author={Kumar, Vikas and Srivastava, Vishesh and Mahjabin, Sadia and Pal, Arindam and Klüttermann, Simon and Müller, Emmanuel},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
