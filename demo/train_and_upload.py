# ============================================================
# Autoencoder Anomaly Detection — Training Script
# IEEE IJCNN 2024 Paper: "Autoencoder Optimization for
# Anomaly Detection: A Comparative Study with Shallow Algorithms"
#
# HOW TO USE ON KAGGLE:
#   1. Create a new Kaggle Notebook
#   2. Enable GPU (Settings → Accelerator → GPU T4 x2)
#   3. Add a secret called HF_TOKEN with your HuggingFace token
#      (Kaggle → Settings → Secrets)
#   4. Paste this entire script into a code cell and run
#   5. Models will be saved to /kaggle/working/ and uploaded
#      automatically to your HuggingFace Hub repo
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ── Install HuggingFace Hub (not pre-installed on Kaggle) ──
os.system("pip install -q huggingface_hub")
from huggingface_hub import HfApi, login, create_repo

# ============================================================
# CONFIGURATION — Edit these before running
# ============================================================

import kaggle_secrets  # only available on Kaggle
HF_TOKEN    = kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")
HF_USERNAME = "VisheshSrivastava"
HF_REPO_ID  = f"{HF_USERNAME}/autoencoder-anomaly-detection"
OUTPUT_DIR  = "/kaggle/working"

# ============================================================
# Login to HuggingFace and create repo if it doesn't exist
# ============================================================

login(token=HF_TOKEN)
api = HfApi()

try:
    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    print(f"HuggingFace repo ready: https://huggingface.co/{HF_REPO_ID}")
except Exception as e:
    print(f"Repo creation note: {e}")

# ============================================================
# Model Architectures  (exact match to paper notebook)
# ============================================================

def build_basic_autoencoder(input_shape=(32, 32, 3)):
    """2-level conv encoder/decoder — used for CIFAR-10, Fashion-MNIST, SVHN."""
    inputs = Input(shape=input_shape)
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(inputs, decoded, name="basic_autoencoder")


def build_deep_autoencoder(input_shape=(32, 32, 3)):
    """3-level conv encoder/decoder — used for MNIST (best performer)."""
    inputs = Input(shape=input_shape)
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(inputs, decoded, name="deep_autoencoder")


# ============================================================
# Helper: augment + train + compute threshold + save + upload
# ============================================================

def train_and_save(
    model, model_name, loss_fn,
    x_train_normal, x_val_normal,
    epochs=50, batch_size=256, lr=0.001
):
    """
    Trains the autoencoder on normal data only (one-class paradigm),
    computes a detection threshold (95th percentile of val MAE),
    saves the model + threshold, and uploads both to HuggingFace Hub.
    """
    # ── Data augmentation (matches paper: width/height shift 0.1) ──
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    aug_batch = next(datagen.flow(x_train_normal,
                                  batch_size=len(x_train_normal),
                                  shuffle=False))
    x_aug = np.concatenate([x_train_normal, aug_batch], axis=0)

    # ── Compile & train ──
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=3,
        restore_best_weights=True
    )
    model.fit(
        x_aug, x_aug,
        epochs=epochs, batch_size=batch_size,
        validation_data=(x_val_normal, x_val_normal),
        callbacks=[early_stop],
        verbose=1
    )

    # ── Compute threshold on validation normal images ──
    reconstructions = model.predict(x_val_normal, verbose=0)
    mae_scores = np.mean(np.abs(reconstructions - x_val_normal), axis=(1, 2, 3))
    threshold = float(np.percentile(mae_scores, 95))
    print(f"  [{model_name}] threshold (95th pct of val MAE): {threshold:.5f}")

    # ── Save locally ──
    model_path     = os.path.join(OUTPUT_DIR, f"{model_name}.h5")
    threshold_path = os.path.join(OUTPUT_DIR, f"{model_name}_threshold.npy")
    model.save(model_path)
    np.save(threshold_path, np.array(threshold))
    print(f"  [{model_name}] saved to {model_path}")

    # ── Upload to HuggingFace Hub ──
    for fpath in [model_path, threshold_path]:
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=os.path.basename(fpath),
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
    print(f"  [{model_name}] uploaded to HuggingFace Hub ✓")

    return threshold


# ============================================================
# Preprocessing helpers
# ============================================================

def to_3ch_32(images, grayscale=False):
    """Resize to 32x32, optionally convert grayscale→3ch, normalize to [0,1].
    FIX: channel dim must be added BEFORE tf.image.resize so TF sees
    (N, H, W, C) not a bare 3-D tensor it mis-interprets as (H, W, C).
    """
    if grayscale and images.ndim == 3:        # (N, H, W) → (N, H, W, 1)
        images = images[..., np.newaxis]
    imgs = tf.image.resize(images, [32, 32]).numpy()   # (N, 32, 32, 1 or 3)
    if imgs.shape[-1] == 1:                   # repeat to 3 channels
        imgs = np.repeat(imgs, 3, axis=-1)
    return imgs.astype('float32') / 255.0


# ============================================================
# 1. CIFAR-10  — Dogs (5) vs Cars (1)
#    Architecture: basic autoencoder | Loss: MSE | Paper AUC: 0.829
# ============================================================
print("\n" + "="*60)
print("TRAINING: CIFAR-10")
print("="*60)

(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
y_tr, y_te = y_tr.flatten(), y_te.flatten()

x_tr_normal_cifar = to_3ch_32(x_tr[y_tr == 5])
x_te_normal_cifar = to_3ch_32(x_te[y_te == 5])

train_and_save(
    model       = build_basic_autoencoder(),
    model_name  = "cifar10_autoencoder",
    loss_fn     = "mean_squared_error",
    x_train_normal = x_tr_normal_cifar,
    x_val_normal   = x_te_normal_cifar,
    epochs=50, lr=0.001
)
del x_tr, y_tr, x_te, y_te


# ============================================================
# 2. Fashion-MNIST  — Trousers (1) vs Dresses (3)
#    Architecture: basic autoencoder | Loss: MSE | Paper AUC: 0.866
# ============================================================
print("\n" + "="*60)
print("TRAINING: Fashion-MNIST")
print("="*60)

(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
y_tr, y_te = y_tr.flatten(), y_te.flatten()

x_tr_normal_fmnist = to_3ch_32(x_tr[y_tr == 1], grayscale=True)
x_te_normal_fmnist = to_3ch_32(x_te[y_te == 1], grayscale=True)

train_and_save(
    model       = build_basic_autoencoder(),
    model_name  = "fashion_mnist_autoencoder",
    loss_fn     = "mean_squared_error",        # MSE > BCE for Fashion-MNIST (paper finding)
    x_train_normal = x_tr_normal_fmnist,
    x_val_normal   = x_te_normal_fmnist,
    epochs=50, lr=0.001
)
del x_tr, y_tr, x_te, y_te


# ============================================================
# 3. MNIST  — Digit '1' vs Digit '3'
#    Architecture: deep autoencoder | Loss: BCE | Paper AUC: 0.999
# ============================================================
print("\n" + "="*60)
print("TRAINING: MNIST")
print("="*60)

(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
y_tr, y_te = y_tr.flatten(), y_te.flatten()

x_tr_normal_mnist = to_3ch_32(x_tr[y_tr == 1], grayscale=True)
x_te_normal_mnist = to_3ch_32(x_te[y_te == 1], grayscale=True)

train_and_save(
    model       = build_deep_autoencoder(),
    model_name  = "mnist_autoencoder",
    loss_fn     = "binary_crossentropy",       # BCE > MSE for MNIST (paper finding)
    x_train_normal = x_tr_normal_mnist,
    x_val_normal   = x_te_normal_mnist,
    epochs=50, lr=0.001
)
del x_tr, y_tr, x_te, y_te


# ============================================================
# 4. SVHN (Cropped)  — Digit '1' vs Others
#    Architecture: basic autoencoder | Loss: BCE | Paper AUC: 0.631
# ============================================================
print("\n" + "="*60)
print("TRAINING: SVHN")
print("="*60)

os.system("pip install -q tensorflow-datasets")
import tensorflow_datasets as tfds

ds_train = tfds.load("svhn_cropped", split="train", as_supervised=True)
ds_test  = tfds.load("svhn_cropped", split="test",  as_supervised=True)

def extract_class(ds, label_val, max_samples=10000):
    imgs, labels = [], []
    for img, lbl in ds:
        imgs.append(img.numpy())
        labels.append(lbl.numpy())
        if len(imgs) >= 50000:
            break
    imgs   = np.array(imgs,   dtype='float32') / 255.0
    labels = np.array(labels)
    mask   = (labels == label_val)
    return imgs[mask][:max_samples]

x_tr_normal_svhn = extract_class(ds_train, label_val=1, max_samples=10000)
x_te_normal_svhn = extract_class(ds_test,  label_val=1, max_samples=2000)
x_tr_normal_svhn = tf.image.resize(x_tr_normal_svhn, [32, 32]).numpy()
x_te_normal_svhn = tf.image.resize(x_te_normal_svhn, [32, 32]).numpy()

train_and_save(
    model       = build_basic_autoencoder(),
    model_name  = "svhn_autoencoder",
    loss_fn     = "binary_crossentropy",
    x_train_normal = x_tr_normal_svhn,
    x_val_normal   = x_te_normal_svhn,
    epochs=100, lr=0.001
)

print("\n" + "="*60)
print("ALL MODELS TRAINED AND UPLOADED SUCCESSFULLY!")
print(f"View at: https://huggingface.co/{HF_REPO_ID}")
print("="*60)
