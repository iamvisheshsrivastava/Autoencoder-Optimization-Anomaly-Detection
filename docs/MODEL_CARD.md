# Model Card — Autoencoder Anomaly Detection

> **Note:** this file is the content for the model card on the HuggingFace
> Hub **model** repository, [`VisheshSrivastava/autoencoder-anomaly-detection`](https://huggingface.co/VisheshSrivastava/autoencoder-anomaly-detection)
> (not the Space). It is kept here in the GitHub repo because pushing it to
> the Hub directly requires a valid HuggingFace write token, which wasn't
> available when this was written (see the note on issue #6). Copy the
> content below into that repo's `README.md` (with the YAML front-matter)
> to close the loop.

```yaml
---
license: mit
library_name: keras
tags:
  - anomaly-detection
  - autoencoder
  - computer-vision
  - one-class-classification
datasets:
  - mnist
  - fashion_mnist
  - cifar10
  - svhn
pipeline_tag: image-classification
---
```

# Autoencoder Anomaly Detection

Convolutional autoencoders trained for **one-class anomaly detection**: each
model is trained only on a single "normal" class and flags anything else as
anomalous based on reconstruction error. From the paper *"Autoencoder
Optimization for Anomaly Detection: A Comparative Study with Shallow
Algorithms"* (IEEE IJCNN 2024).

- **Paper (DOI):** [10.1109/IJCNN60899.2024.10650057](https://doi.org/10.1109/IJCNN60899.2024.10650057)
- **Code:** [github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection)
- **Live demo (Space):** [huggingface.co/spaces/VisheshSrivastava/autoencoder-anomaly-detection](https://huggingface.co/spaces/VisheshSrivastava/autoencoder-anomaly-detection)

## Results (AUC-ROC)

| Model file | Dataset | Normal class | Anomalous class | Architecture | Loss | Autoencoder AUC-ROC | Best shallow baseline |
|---|---|---|---|---|---|---|---|
| `mnist_autoencoder.h5` | MNIST | Digit '1' | Digit '3' | Deep (3-level) | Binary Cross-Entropy | **0.999** | 0.377 (PCA) |
| `fashion_mnist_autoencoder.h5` | Fashion-MNIST | Trousers | Dresses | Basic (2-level) | MSE | **0.866** | 0.560 (PCA) |
| `cifar10_autoencoder.h5` | CIFAR-10 | Dogs | Cars | Basic (2-level) | MSE | **0.829** | 0.740 (LOF) |
| `svhn_autoencoder.h5` | SVHN | Digit '1' | Other digits | Basic (2-level) | Binary Cross-Entropy | **0.631** | 0.596 (LOF) |
| *(not yet trained)* | MVTec-AD | — | industrial defects | Basic (2-level) | MSE | 0.483 | 0.653 (PCA) |

MVTec-AD is the one dataset in the paper where the shallow baseline (PCA)
**beats** the autoencoder — most likely because resizing the industrial
defect images down to 32×32 throws away the fine texture the model needs
to catch small defects. See GitHub issue
[#3](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/3)
for status (no weights yet — dataset requires a license to download).

All models are trained one-class: only normal-class images during training,
with a decision threshold set at the 95th percentile of validation-set
normal reconstruction error. At inference, any image whose reconstruction
MAE exceeds that threshold is flagged anomalous.

## Usage

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

REPO_ID = "VisheshSrivastava/autoencoder-anomaly-detection"

model_path = hf_hub_download(repo_id=REPO_ID, filename="cifar10_autoencoder.h5")
threshold_path = hf_hub_download(repo_id=REPO_ID, filename="cifar10_autoencoder_threshold.npy")

model = load_model(model_path, compile=False)
threshold = float(np.load(threshold_path))

# Preprocess: resize to 32x32, scale to [0, 1], 3 channels (repeat grayscale if needed)
img = Image.open("your_image.png").convert("RGB").resize((32, 32))
image = np.asarray(img, dtype="float32") / 255.0

reconstruction = model.predict(image[np.newaxis])[0]
mae_score = float(np.mean(np.abs(reconstruction - image)))

is_anomalous = mae_score > threshold
print(f"MAE={mae_score:.5f}  threshold={threshold:.5f}  anomalous={is_anomalous}")
```

Each dataset has its own `<dataset>_autoencoder.h5` + `<dataset>_autoencoder_threshold.npy`
pair; see the results table above for the available filenames. The Gradio
Space (`demo/app.py` in the GitHub repo) wraps this exact flow with a UI,
input-size/file-count guardrails, and batch visualisation.

## Limitations

- Each model was trained on a single normal class vs. a single held-out
  anomalous class — these are *not* general-purpose anomaly detectors for
  arbitrary images of the same dataset's other classes.
- All inputs are resized to 32×32, which caps how much fine detail the model
  can use (see the MVTec-AD result above for where this clearly hurts).
- `.h5` is the legacy Keras HDF5 checkpoint format; loading uses
  `compile=False` and (where supported) Keras's `safe_mode=True` to reduce
  (not eliminate) the risk of a malicious checkpoint executing arbitrary
  code via embedded Lambda layers. Only load checkpoints from a repo/revision
  you trust.

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

MIT — see [LICENSE](../LICENSE) in the GitHub repo.
