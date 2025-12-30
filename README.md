# Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms

## Executive Summary
Peer-reviewed research demonstrating how optimized autoencoder architectures outperform traditional shallow algorithms for anomaly detection.
Validated on benchmark datasets and real-world manufacturing data, showing consistent performance gains in robustness and detection accuracy.
Published at IEEE IJCNN 2024 and extended to industrial defect detection (Journal of Composite Materials, 2025).

This repository contains the code accompanying the research paper:

**"Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms"**

Authors: Vikas Kumar, Vishesh Srivastava, Sadia Mahjabin, Emmanuel Müller

## Overview

This project investigates how careful optimization of autoencoder architectures—particularly latent space size, loss functions, and training strategies—can significantly improve anomaly detection performance.
The study benchmarks optimized autoencoders against traditional shallow algorithms using the ADBENCH framework across both image and tabular datasets.

## Key Results
- Optimized convolutional autoencoders consistently outperform shallow anomaly detection methods on multiple image datasets (CIFAR-10, FashionMNIST, MNIST-C, MVTec-AD, SVHN) in terms of AUC-ROC
- Proper latent space selection and loss function choice (MSE vs. Binary Cross-Entropy) are critical drivers of performance gains
- Variational Autoencoders (VAEs) achieve stronger anomaly detection performance than basic autoencoders on most tabular datasets, measured using AUC-ROC and PR-AUC
- Results demonstrate that well-optimized autoencoders can match or exceed state-of-the-art shallow methods highlighted in ADBENCH

## Publications
- **IEEE IJCNN 2024**  
  Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms  
  DOI: 10.1109/IJCNN60899.2024.10650057
  
## Repository Structure

....

## Requirements

- Python 3.x
- Required libraries: numpy, pandas, tensorflow, scikit-learn, matplotlib

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place your datasets in the `data/` directory.
2. **Training**: Use the scripts in the `scripts/` directory to train models.
3. **Evaluation**: Evaluate the trained models using the provided evaluation scripts.

For detailed instructions, refer to the documentation within each script.

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{Kumar2024Autoencoder,
  title={Autoencoder Optimization for Anomaly Detection: A Comparative Study with Shallow Algorithms},
  author={Kumar, Vikas and Srivastava, Vishesh and Mahjabin, Sadia and Müller, Emmanuel},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.
