"""
Latent-dimension ablation sweep (issue #15).

Trains the same convolutional-autoencoder architecture used in
"Image Data.ipynb" across a sweep of bottleneck sizes (the filter count of
the deepest Conv2D layer, which is this architecture's "latent capacity" —
it has no single flattened latent vector, just a 8x8xN bottleneck feature
map) and records AUC-ROC per run.

Kept as a standalone script rather than new notebook cells: "Image Data.ipynb"
is already ~1MB with embedded outputs, and a script is easier to re-run,
diff, and keep reproducible from the command line.

Usage:
    cd docs
    python latent_dim_sweep.py --dataset mnist
    python latent_dim_sweep.py --dataset fashion_mnist

Writes/updates:
    docs/latent_dim_sweep_results.csv   (dataset, latent_dim, auc_roc)
    docs/screenshots/latent_dim_sweep.png
"""
import argparse
import csv
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "latent_dim_sweep_results.csv")
PLOT_PATH = os.path.join(HERE, "screenshots", "latent_dim_sweep.png")

# (normal_class, anomalous_class, loss) per dataset — matches the paper /
# Image Data.ipynb's choices for each dataset.
DATASET_CONFIG = {
    "mnist":         dict(normal=1, anomalous=3, loss="binary_crossentropy"),
    "fashion_mnist": dict(normal=1, anomalous=3, loss="mean_squared_error"),
}

LATENT_DIMS = [8, 16, 32, 64, 128]


def load_dataset(name):
    loader = getattr(tf.keras.datasets, name)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    if x_train.ndim == 3:  # grayscale -> repeat to 3 channels
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)

    x_train = tf.image.resize(x_train, [32, 32]).numpy() / 255.0
    x_test = tf.image.resize(x_test, [32, 32]).numpy() / 255.0

    cfg = DATASET_CONFIG[name]
    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
    x_train_normal = x_train[y_train == cfg["normal"]]
    x_test_normal = x_test[y_test == cfg["normal"]]
    x_test_anomalous = x_test[y_test == cfg["anomalous"]]
    return x_train_normal, x_test_normal, x_test_anomalous, cfg["loss"]


def build_autoencoder(latent_dim):
    """Same 2-level conv autoencoder as Image Data.ipynb, with the
    bottleneck (deepest Conv2D) filter count parameterized."""
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(32, (5, 5), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(latent_dim, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(latent_dim, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)
    return Model(input_img, decoded)


def run_sweep(dataset_name, latent_dims=None, epochs=50, batch_size=256, on_result=None):
    x_train_normal, x_test_normal, x_test_anomalous, loss = load_dataset(dataset_name)
    results = []

    for latent_dim in (latent_dims or LATENT_DIMS):
        tf.keras.utils.set_random_seed(42)
        model = build_autoencoder(latent_dim)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss)

        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(
            x_train_normal, x_train_normal,
            epochs=epochs, batch_size=batch_size,
            validation_data=(x_test_normal, x_test_normal),
            callbacks=[early_stopping],
            verbose=2,
        )

        recon_normal = model.predict(x_test_normal, verbose=0)
        recon_anomalous = model.predict(x_test_anomalous, verbose=0)
        mae_normal = np.mean(np.abs(recon_normal - x_test_normal), axis=(1, 2, 3))
        mae_anomalous = np.mean(np.abs(recon_anomalous - x_test_anomalous), axis=(1, 2, 3))

        labels = np.concatenate([np.zeros(len(mae_normal)), np.ones(len(mae_anomalous))])
        scores = np.concatenate([mae_normal, mae_anomalous])
        auc = roc_auc_score(labels, scores)

        print(f"[{dataset_name}] latent_dim={latent_dim:>3}  AUC-ROC={auc:.4f}")
        row = (dataset_name, latent_dim, auc)
        results.append(row)
        if on_result:
            # Persist immediately so a long multi-config sweep doesn't lose
            # earlier results if a later (slower) config is interrupted.
            on_result(row)

    return results


def append_results_csv(rows):
    file_exists = os.path.exists(CSV_PATH)
    existing = []
    if file_exists:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            existing = list(csv.reader(f))[1:]  # skip header
    # Replace any rows for (dataset, latent_dim) pairs we just re-ran.
    keys = {(r[0], str(r[1])) for r in rows}
    existing = [r for r in existing if (r[0], r[1]) not in keys]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "latent_dim", "auc_roc"])
        writer.writerows(existing)
        writer.writerows(rows)


def plot_results():
    import matplotlib.pyplot as plt

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]

    by_dataset = {}
    for dataset, latent_dim, auc in rows:
        by_dataset.setdefault(dataset, []).append((int(latent_dim), float(auc)))

    plt.figure(figsize=(8, 5))
    for dataset, points in by_dataset.items():
        points.sort()
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", label=dataset)

    plt.xscale("log", base=2)
    plt.xlabel("Bottleneck size (latent capacity, # filters)")
    plt.ylabel("AUC-ROC")
    plt.title("Latent-dimension ablation sweep")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    print("wrote", PLOT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--latent-dims", type=str, default=None,
        help="Comma-separated subset of latent dims to run, e.g. '8,16'. Defaults to all of " + str(LATENT_DIMS),
    )
    args = parser.parse_args()

    latent_dims = [int(d) for d in args.latent_dims.split(",")] if args.latent_dims else None

    rows = run_sweep(
        args.dataset, latent_dims=latent_dims, epochs=args.epochs,
        on_result=lambda row: append_results_csv([row]),
    )
    plot_results()
