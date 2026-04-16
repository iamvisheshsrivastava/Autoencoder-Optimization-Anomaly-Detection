# Autoencoder Optimization for Anomaly Detection

This repository contains the notebook and supporting notes for the paper "Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms."

## Overview

The project studies how latent-space size, loss functions, and training choices affect anomaly detection performance, with a focus on autoencoder-based methods.

## Repository contents

- `Image Data.ipynb`: notebook for the image-data experiments in this repository
- `README.md`: project overview, usage notes, and citation

## Publication

- IEEE IJCNN 2024
- "Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms"
- DOI: `10.1109/IJCNN60899.2024.10650057`

## Highlights

- compares autoencoder-based anomaly detection against shallow baselines
- studies the effect of latent dimension and reconstruction loss choices
- keeps the repository lightweight and notebook-centered for reproducible experiments

## How to use

1. Clone the repository.
2. Create a Python environment with the libraries required for the notebook.
3. Open `Image Data.ipynb` in Jupyter Notebook or JupyterLab.
4. Update any dataset paths or environment-specific settings before running all cells.

A typical environment will include:

- `numpy`
- `pandas`
- `tensorflow`
- `scikit-learn`
- `matplotlib`
- `jupyter`

## Citation

```bibtex
@inproceedings{Kumar2024Autoencoder,
  title={Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms},
  author={Kumar, Vikas and Srivastava, Vishesh and Mahjabin, Sadia and Muller, Emmanuel},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
