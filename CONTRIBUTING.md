# Contributing

Thanks for your interest in improving this project. It has two distinct parts, and
contributions to either are welcome:

- **`demo/`** — a Gradio app (`demo/app.py`) that serves the trained autoencoders
  interactively, plus its test suite (`demo/test_app.py`).
- **`Image Data.ipynb`** — the research notebook used to produce the paper's results
  (training, evaluation, and comparisons against shallow baselines like PCA/LOF/CBLOF/KNN).

Check [open issues](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues),
especially those labeled
[`good first issue`](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22),
before starting work, and open an issue to discuss anything non-trivial before sending a PR.

## Dev setup

You'll need Python 3.9–3.11.

```bash
git clone https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection.git
cd Autoencoder-Optimization-Anomaly-Detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Install only what you need for the part you're touching:

- **Demo app** (Gradio + inference only, no training):
  ```bash
  pip install -r demo/requirements.txt
  ```
- **Notebook** (full training/evaluation stack — TensorFlow, keras-tuner, scikit-learn, PyOD):
  ```bash
  pip install -r requirements_notebook.txt
  ```
  See the comments at the top of `requirements_notebook.txt` for a known
  `tensorflow-datasets`/`protobuf` version conflict (tracked in
  [issue #7](https://github.com/iamvisheshsrivastava/Autoencoder-Optimization-Anomaly-Detection/issues/7))
  if you plan to load datasets via `tensorflow_datasets`.

## Running the demo locally

From the `demo/` directory:

```bash
cd demo
python app.py
```

This launches a local Gradio server (prints a `http://127.0.0.1:7860` URL) and will
download the pretrained `.h5` model checkpoints from the
[HuggingFace Hub model repo](https://huggingface.co/VisheshSrivastava/autoencoder-anomaly-detection)
on first use of each model — this needs internet access but no authentication token
for the public models. Checkpoints are cached locally after the first download, and
again in-process (`_model_cache`) after the first request in a given run.

Click **"Load sample images for this model"** to try it immediately without finding
your own test images — bundled samples live under `demo/sample_images/<dataset>/`.

To point at a different model repo or pinned revision while testing (e.g. a fork you
pushed weights to), set the environment variables `app.py` reads at import time:

```bash
HF_REPO_ID=your-username/your-model-repo HF_MODEL_REVISION=main python app.py
```

## Running the tests

The test suite (`demo/test_app.py`) is deliberately offline — it stubs out model
loading, so it never downloads a checkpoint or hits the network. Run it from the
`demo/` directory with the demo's dependencies plus `pytest` installed:

```bash
cd demo
pip install -r requirements.txt pytest
pytest test_app.py -v
```

All tests should pass before you open a PR. If you're fixing a bug or adding a
guardrail (file-size limits, corrupt-upload handling, threshold reclassification,
CSV export, etc.), add a test that would have caught it — existing tests in that
file are a good template (see the `_patch_get_model`/`_make_image_file` helpers).

## Working on the notebook

`Image Data.ipynb` is research code, not a library — keep edits minimal and
reproducible:

- Set random seeds where you add new training/evaluation cells (see issue history
  around reproducibility).
- Don't commit large datasets or model checkpoints into the repo; trained weights
  belong on the HuggingFace Hub model repo, referenced from the notebook/app by
  filename.
- If you change a training pipeline that the demo models depend on (preprocessing,
  input size, architecture), note it in your PR description so `demo/app.py`'s
  `preprocess()` function can be kept in sync — the demo assumes its preprocessing
  exactly matches training.

## Pull request process

1. Fork the repo and create a branch off `main` (or the current default branch)
   named for what it does, e.g. `fix-threshold-slider-reset`.
2. Keep PRs small and focused on one issue/change — easier to review, easier to
   revert if something's wrong.
3. For `demo/` changes: run `pytest demo/test_app.py` locally and make sure it
   passes; add/update tests for any behavior change.
4. For notebook changes: re-run affected cells top-to-bottom before committing so
   outputs are consistent, and mention in the PR description what you verified.
5. Write a clear PR description: what changed, why, and how you tested it. Link
   the issue it resolves (e.g. `Closes #23`).
6. Be responsive to review feedback — most PRs need at least one round of changes.

Please don't modify the paper authorship, citation, or DOI information in the
README or model cards as part of unrelated contributions.

## Questions

Open an issue if anything here is unclear or you're not sure whether a change fits
the project's scope — happy to discuss before you put in the work.
