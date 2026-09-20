# FaunaDRAEM

Code repository for the following paper:

>G. Serati, S. Foucher and J. Théau, "From Anomaly Maps to Animal Detection: Unsupervised Anomaly Detection for Caribou (Rangifer tarandus) Monitoring," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2026.3733293. [preprint](https://arxiv.org/pdf/2307.06720)

## What this is

Unsupervised anomaly detection for caribou in 512×512 aerial patches. A DRAEM-style network is
trained only on *empty* (animal-free) patches; anomalies are synthesized during training by
compositing real animal silhouettes onto those backgrounds, and at test time animals show up as
anomalies. PaDiM and PaDiM-NF (normalizing flow) are the baselines.

The additions over stock DRAEM are an SSPCAB block in the discriminative decoder and a `TauHead`
that turns the mean reconstruction error into a per-patch detection logit. Segmentation and
detection thresholds are fixed for the whole experiment rather than tuned per run.

## Quick start

```bash
./docker/build       # build the CUDA image (see docker/README.md)
./docker/run         # shell in the container, repo mounted at /DRAEM
```

Inside the container:

```bash
python main.py                              # SAM masks → train → test + PDF report
python infer_Draem.py --ckpt ckpt.pt --patches /dir/of/patches --out /outdir
python post_processing.py                   # binary maps → boxes/points + CSV
```

Everything is scripts — no packaging, no test suite. `docker/README.md` covers the container,
including the GPU/CUDA pinning and a troubleshooting table.

## Configuration

All experiment settings live in `build_args()` in `main.py` as a hardcoded `Namespace`: dataset
paths, learning rate, epochs, thresholds, model choice (`draem`, `padim`, `padim_nf`). Edit it
directly — the argparse branch in that function is never called. Note the shipped config runs
`epochs=1`; raise it to reproduce the paper's training.

Paths in `main.py` are absolute and container-side, all under `/DRAEM`.

## Data

Not included. You supply:

- **Empty background patches**, exactly 512×512, for training
- **Test patches** plus their masks
- **Point annotations** as CSV (`images,x,y`); `generate_masks_sam.py` turns each point into a box,
  prompts SAM ViT-H, and writes one mask per image. The SAM checkpoint downloads automatically on
  the first `main.py` run.

`Silhouette_Bank/` ships with the repo: 100 paired silhouette crops and masks used to synthesize
anomalies. `Extract_silhouet_boxes_from_masks.py` rebuilds it from SAM masks.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Entry point and the experiment configuration |
| `model.py`, `loss.py` | DRAEM networks (+ SSPCAB, TauHead), focal and SSIM losses |
| `train.py`, `train_dataset.py` | Training loop; clean/anomalous patch synthesis |
| `evaluate.py`, `metrics.py` | Test-time evaluation, pixel and patch metrics, plots, PDF report |
| `infer_Draem.py` | Standalone inference from a frozen checkpoint |
| `post_processing.py` | Anomaly maps → boxes/points, scored against ground-truth points |
| `Padim/` | PaDiM and PaDiM-NF baselines (evaluation only) |
| `docker/` | Container image, build/run scripts, container docs |
