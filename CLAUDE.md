# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the JSTARS paper "From Anomaly Maps to Animal Detection: Unsupervised Anomaly Detection for Caribou (Rangifer tarandus) Monitoring". DRAEM-style unsupervised anomaly detection on 512×512 aerial patches: train only on *empty* (animal-free) patches, synthesize anomalies by compositing real animal silhouettes onto them, then detect animals at test time as anomalies. PaDiM and PaDiM-NF (normalizing-flow) are the baselines.

No test suite, no linter config, no packaging. Everything runs as scripts.

## Running

Everything assumes the container working dir `/DRAEM` (paths in `main.py` are absolute and container-side).

```bash
# Build (context is the repo root; ./docker/build does the same with --no-cache/--pull flags)
docker build -t draem_image:local -f docker/Dockerfile .

# Shell in container, or run a command in it (datasets default to ./DATASETS)
./docker/run
./docker/run python main.py
DATASETS_DIR=/mnt/data/caribou ./docker/run

# Full pipeline: SAM download → SAM masks → DRAEM train → test + PDF report
python main.py

# Standalone inference from a frozen checkpoint (only real CLI in the repo)
python infer_Draem.py --ckpt ckpt.pt --patches /dir/of/patches --out /outdir [--bs 16] [--keep-size]

# Binary maps → boxes/points + CSV + overlays (edit the module-level constants first)
python post_processing.py

# Rebuild the silhouette bank from SAM masks (edit paths at the bottom of the file)
python Extract_silhouet_boxes_from_masks.py
```

## Configuration

All experiment config lives in `build_args()` in `main.py` as a hardcoded `Namespace` — `main()` calls `build_args(use_cli=False)`, so **the argparse branch is dead code**. To change a dataset path, threshold, LR, or model, edit that Namespace. Torchvision transform pipelines are built at module level in `main.py` and passed through `args` (`train_img_tf`, `val_img_tf`, …); train uses raw 512 crops resized in the dataset, val/test resize to 256.

Not everything is in the Namespace: `main()` hardcodes the test-2 output dirs (`/DRAEM/Publish_res/{heatmaps,anomalymaps,binarymaps}` and the metrics PDF) at the call site of `run_test`.

## Pipeline and data contracts

`main.py` → `load_sam.download_file` → `generate_masks_sam.generate` → `train.train` → `evaluate.run_test`.

1. **SAM mask generation** (`generate_masks_sam.py`): reads point annotations from `args.csv_file_dir` (CSV with `images,x,y`), turns each point into a box (55px, or 35px when a KD-tree neighbor is within 7px), prompts SAM ViT-H, ORs the per-box masks into one mask per image, writes to `args.masks_path`. Skips entirely if that dir is non-empty.
2. **Silhouette bank** (`Silhouette_Bank/`): paired `sil_boxes/<name>.png` (RGB crop) and `masks/<name>.png`. `TrainDataset` first looks for per-group subdirs (`<sil_root>/<group>/sil_boxes` + `/masks`), then falls back to the flat layout that ships in this repo. Pairing is by identical filename.
3. **Training** (`train.py`, `train_dataset.py`): two `TrainDataset` instances over the same empty-patch background dir — one `mode="clean"`, one `mode="anom"` — are iterated in lockstep and concatenated into a 50/50 clean/anomalous batch each step. Backgrounds **must be exactly 512×512** or `__getitem__` raises. Anomalies are composited with edge-aware feathering; scaling is always on, other augs only with `apply_augmentations`.
4. **Evaluation** (`evaluate.py`, 2k lines, one giant `run_test` with three model branches): writes heatmaps, anomaly maps, binary maps, PR/ROC/threshold-sweep plots and a metrics PDF.
5. **Post-processing** (`post_processing.py`): color-keyed binary maps + optional `.npy` heatmaps → geodesic growth from seeds, morphology, hole fill, shape filters, NMS → boxes/points CSV scored against GT points.

## Model and loss

`model.py` is DRAEM plus two additions: `SSPCAB` (self-supervised predictive block, returns an auxiliary `ssp_loss` from the decoder) and `TauHead` (1-input logistic regression on the mean reconstruction error → per-patch detection logit; the learned threshold is `τ ≈ −bias/weight`).

`DiscriminativeSubNetwork.forward` returns a 4-tuple `(seg_map, ssp_loss, avg_err, tau_logit)` — not the plain DRAEM single output. Any new call site must unpack all four.

Training loss = `MSE(rec, clean) + SSIM(rec, clean) + Focal(softmax(seg), mask) + 0.1·ssp_loss`, plus `args.det_loss_w · BCEWithLogits(tau_logit, patch_label)`. Optimizer is AdamW over rec params + the seg subnet's four submodules (`encoder_segment`, `decoder_segment`, `reduce_channels`, `tau_head`) with linear warmup then cosine decay.

## Thresholds

Deliberate design: `segmentation_threshold` and `detection_threshold` are **fixed in `main.py` for the whole experiment and never searched**. `train()` validates they are in [0,1], stamps them into the checkpoint under `thr`, and evaluation warns but still prefers the `main.py` values when the checkpoint disagrees. Don't add threshold tuning to the DRAEM path without being asked — it would invalidate the paper's protocol.

## Checkpoints and model dispatch

A DRAEM checkpoint is `{"rec": sd, "seg": sd, "thr": {"seg","det"}, "epoch": int}`; PaDiM uses a `"padim"` key, PaDiM-NF a `"padim_nf"` key. `evaluate.sniff_ckpt_type` detects the type from those keys and **overrides `args.model`** if they disagree. DataParallel `module.` prefixes are stripped on save (`_state_dict_without_dataparallel`) and again on load.

`train()` writes **one** checkpoint, after the loop, from the weights of the last epoch it ran — `args.checkpoint_dir/args.checkpoint_name` (default `draem_final.pt`). The returned key is named `best_ckpt`, but no best-epoch weights are kept: early stopping only cuts training short, it does not restore a better epoch.

The baselines live in `Padim/`: `padim_adapter.PaDiMModel_padim` (vanilla PaDiM, `prec` key in its state), `padim_nf_adapter.PaDiMNFAdapter` (multi-headed normalizing flow over PaDiM features), and `MAF.py` (MAF/RealNVP flows). Both adapters wrap a `ResNetFeatures` backbone and share a `gaussian_blur_map` smoother.

`train.py` only implements the DRAEM path — it raises `NotImplementedError` for `model="padim"`/`"padim_nf"`; those checkpoints come from elsewhere and are only *consumed* by `evaluate.py`.

## Segmentation metric convention ("hybrid")

`_append_hybrid_pixels` in `train.py` (mirrored in `evaluate.py`): for a patch **with** GT foreground, only the GT-positive pixels contribute (as positives); for an empty patch, **all** pixels contribute (as negatives). Pixel scores are pooled globally across the whole split before computing F1/AP/AUROC — per-image metrics are never averaged. `args.eval_scope="hybrid"` selects this; `metrics.evaluate_core_pixel_metrics` also has `blob`/`core`/`inmask` variants.

Early stopping monitors `args.early_stop_metric` on segmentation only, starting at `monitor_after`, and segmentation validation runs every `seg_eval_stride` epochs (detection metrics run every epoch).

## Supporting modules

- `loss.py`: vendored DRAEM `FocalLoss` + `SSIM`.
- `metrics.py`: the pixel-metric scopes (`hybrid`/`blob`/`core`/`inmask`), `compute_global_seg_metrics`, `evaluate_detection_metrics`, `det_score_from_map` (patch score from a map, default p95), and the connected-component → box → NMS helpers that `image_score_with_postproc` chains.
- `utils.py`: overlays, binary-map writing, PR/ROC/sweep plots, `BalancedSampler` (`evaluate.py` has its own `StratifiedBatchSampler`).
- `infer_Draem.py`: self-contained DRAEM loader + inference, independent of `evaluate.run_test` — use it when you only need maps, not metrics.

## Gotchas

- **`evaluate.py:21` imports `threshold_tunning`, which is not in the repo** — so `import evaluate` (and therefore `python main.py`) fails at import time. Only `apply_calibration` is needed, and only in the PaDiM-NF branch (`evaluate.py:300`).
- `train()` raises if CUDA is unavailable, and `batch_size` is rounded up to a multiple of the GPU count.
- Python deps are `docker/requirements.txt`; torch/torchvision come from the base image (`pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime`, pinned by the `ARG`s at the top of the Dockerfile) — adding them to requirements would clobber the CUDA build. `numpy<2` is pinned because `imgaug` (hard-imported by `train_dataset.py`, even with `apply_augmentations=False`) breaks on newer numpy, and `imgaug` is installed `--no-deps` in a second pip command because its `install_requires` pulls the GL-linked `opencv-python`, which shadows `opencv-python-headless` and makes `import cv2` die on `libGL.so.1`. Its real deps are listed in `requirements.txt` instead.
- `docker/run` mounts the repo over `/DRAEM`, shadowing the image's own `COPY` — host edits are live, and the image's copy is only what you get without the mount. It runs as the calling user with `HOME=/tmp` so results are not root-owned, which also means `pip install` inside the container fails; use `docker run --user root` for that. See `docker/README.md`.
- `post_processing.py` and `Extract_silhouet_boxes_from_masks.py` take no arguments — their inputs are module-level constants / bottom-of-file assignments pointing at absolute paths from the authors' machine.
- `utils.mask_exists` returns False for anything but `.jpg`; `TestDataset` compensates with a same-extension fallback, so a patch whose mask is missing is silently labeled *normal*.
- Style is heavily line-broken (one argument per line) and several files re-import the same modules mid-file. Match the surrounding style rather than reformatting.
