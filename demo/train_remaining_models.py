# ============================================================
# Train remaining 3 models: Fashion-MNIST, MNIST, SVHN
# Paste this into a NEW Kaggle cell and run.
# (CIFAR-10 is already done — this skips it.)
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.system("pip install -q huggingface_hub")
from huggingface_hub import HfApi, login
import kaggle_secrets

HF_TOKEN   = kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")
HF_REPO_ID = "VisheshSrivastava/autoencoder-anomaly-detection"
OUTPUT_DIR = "/kaggle/working"

login(token=HF_TOKEN)
api = HfApi()

# ── Fixed preprocessing (bug fix: add channel BEFORE resize) ──
def to_3ch_32(images, grayscale=False):
    if grayscale and images.ndim == 3:     # (N,H,W) → (N,H,W,1)
        images = images[..., np.newaxis]
    imgs = tf.image.resize(images, [32, 32]).numpy()
    if imgs.shape[-1] == 1:
        imgs = np.repeat(imgs, 3, axis=-1)
    return imgs.astype('float32') / 255.0

# ── Architectures ──
def build_basic_autoencoder():
    inp = Input(shape=(32,32,3))
    x = Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
    enc = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(64,(3,3),activation='relu',padding='same')(enc)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((2,2))(x)
    out = Conv2D(3,(3,3),activation='sigmoid',padding='same')(x)
    return Model(inp, out)

def build_deep_autoencoder():
    inp = Input(shape=(32,32,3))
    x = Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
    enc = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(128,(3,3),activation='relu',padding='same')(enc)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((2,2))(x)
    out = Conv2D(3,(3,3),activation='sigmoid',padding='same')(x)
    return Model(inp, out)

def train_and_save(model, name, loss_fn, x_tr, x_val, epochs=50, lr=0.001):
    dg = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    aug = next(dg.flow(x_tr, batch_size=len(x_tr), shuffle=False))
    x_aug = np.concatenate([x_tr, aug])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss_fn)
    model.fit(x_aug, x_aug, epochs=epochs, batch_size=256,
              validation_data=(x_val, x_val),
              callbacks=[EarlyStopping(monitor='val_loss', patience=3,
                                       restore_best_weights=True)], verbose=1)
    recon = model.predict(x_val, verbose=0)
    mae   = np.mean(np.abs(recon - x_val), axis=(1,2,3))
    thr   = float(np.percentile(mae, 95))
    print(f"[{name}] threshold: {thr:.5f}")
    m_path = f"{OUTPUT_DIR}/{name}.h5"
    t_path = f"{OUTPUT_DIR}/{name}_threshold.npy"
    model.save(m_path)
    np.save(t_path, np.array(thr))
    for p in [m_path, t_path]:
        api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p),
                        repo_id=HF_REPO_ID, repo_type="model")
    print(f"[{name}] uploaded ✓")

# ============================================================
# 1. Fashion-MNIST  — Trousers vs Dresses
# ============================================================
print("\n== Fashion-MNIST ==")
(x_tr,y_tr),(x_te,y_te) = tf.keras.datasets.fashion_mnist.load_data()
y_tr,y_te = y_tr.flatten(),y_te.flatten()
train_and_save(
    build_basic_autoencoder(), "fashion_mnist_autoencoder",
    "mean_squared_error",
    to_3ch_32(x_tr[y_tr==1], grayscale=True),
    to_3ch_32(x_te[y_te==1], grayscale=True),
    epochs=50
)
del x_tr,y_tr,x_te,y_te

# ============================================================
# 2. MNIST  — Digit 1 vs Digit 3
# ============================================================
print("\n== MNIST ==")
(x_tr,y_tr),(x_te,y_te) = tf.keras.datasets.mnist.load_data()
y_tr,y_te = y_tr.flatten(),y_te.flatten()
train_and_save(
    build_deep_autoencoder(), "mnist_autoencoder",
    "binary_crossentropy",
    to_3ch_32(x_tr[y_tr==1], grayscale=True),
    to_3ch_32(x_te[y_te==1], grayscale=True),
    epochs=50
)
del x_tr,y_tr,x_te,y_te

# ============================================================
# 3. SVHN  — Digit 1 vs Others
# ============================================================
print("\n== SVHN ==")
os.system("pip install -q tensorflow-datasets")
import tensorflow_datasets as tfds

def load_svhn_class(split, label_val, max_n=10000):
    ds = tfds.load("svhn_cropped", split=split, as_supervised=True)
    imgs, lbls = [], []
    for img,lbl in ds:
        imgs.append(img.numpy()); lbls.append(lbl.numpy())
        if len(imgs) >= 50000: break
    imgs  = np.array(imgs,  dtype='float32') / 255.0
    lbls  = np.array(lbls)
    return tf.image.resize(imgs[lbls==label_val][:max_n], [32,32]).numpy()

train_and_save(
    build_basic_autoencoder(), "svhn_autoencoder",
    "binary_crossentropy",
    load_svhn_class("train", 1, 10000),
    load_svhn_class("test",  1, 2000),
    epochs=100
)

print("\n ALL 3 MODELS DONE!")
print(f"https://huggingface.co/{HF_REPO_ID}")
