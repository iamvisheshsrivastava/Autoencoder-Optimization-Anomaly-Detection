# Roadmap

This project is a paper artifact (IEEE IJCNN 2024) plus a small Gradio demo
on top of it. The roadmap below is scoped accordingly — it's not a product
backlog, it's "what's realistically worth doing to a research-artifact repo
and demo app."

## Now

Things that are blocked on something outside this repo, or are open bugs
affecting the live demo today:

- **HuggingFace Hub access is currently blocked** (invalid/expired HF
  token). This blocks three open issues and nothing else should be started
  on top of them until it's resolved:
  - [#3](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/3) — MVTec-AD dataset support (code path is already wired into `demo/app.py`'s `MODEL_CONFIG`; only the trained weights are missing, and training also needs the licensed dataset downloaded first)
  - [#6](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/6) — publishing `docs/MODEL_CARD.md` content to the HF Hub model repo's `README.md`
  - [#7](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/7) — uploading trained SVHN weights
- [#22](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/22) — the live HF Space is running a stale `app.py` (missing the sample-image gallery, threshold slider, and CSV export that are already merged to `main`). This is a sync/deploy gap, not a code bug, and doesn't depend on the HF token issue — it just needs the Space's git remote pushed to.

## Next

Small, well-scoped, self-contained fixes and tests that don't need model
training or new infrastructure:

- [#23](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/23) — switching the dataset dropdown doesn't clear the previous Analyse run's results (good first issue)
- [#24](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/24) — missing tests for `classify_rows()`'s threshold boundary and the exact `MAX_UPLOAD_FILES` limit (good first issue)
- [#25](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/25) — add a "download visualization as PNG" button next to the existing CSV export, reusing the same temp-file lifecycle as `export_csv()`
- Auto-sync the HF Space from CI on every push to `main` (a follow-up to #22's root cause, so the Space can't drift from `demo/app.py` again)

## Later

Larger or more open-ended ideas that are real but lower priority — worth
doing if someone has the bandwidth, not blocking anything else:

- [#26](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/26) — extend `docs/latent_dim_sweep.py`'s ablation to CIFAR-10 and SVHN (currently only MNIST/Fashion-MNIST are configured) and plot all datasets' latent-capacity curves together, since CIFAR-10/SVHN are exactly the cases where the paper's MNIST finding ("capacity barely matters") might *not* hold
- Once MVTec-AD weights exist (post-#3), revisit whether a higher input resolution than the shared 32×32 is worth a one-off exception for that dataset, since the README already flags 32×32 resizing as the likely reason PCA beats the autoencoder there
- Broader CI (linting demo/app.py, notebook execution smoke test) if the repo keeps growing — not currently justified by its size

## Explicitly not planned

- No rewrite of the training pipeline or model architecture — the paper's
  results are the point, and re-deriving them isn't on the table.
- No new datasets beyond what the paper already covers (MNIST,
  Fashion-MNIST, CIFAR-10, SVHN, MVTec-AD).
- Paper authorship/citation content is out of scope for this roadmap.
