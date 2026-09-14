from __future__ import annotations
import math
import os
import random
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import FocalLoss, SSIM
from metrics import compute_global_seg_metrics, evaluate_detection_metrics
from model import DiscriminativeSubNetwork, ReconstructiveSubNetwork
from test_dataset import TestDataset
from train_dataset import TrainDataset
from utils import (
    overlay_anomaly_on_rgb,
    overlay_heatmap_on_image,
    plot_all_training_metrics,
    save_binary_anomaly_map,
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

cv2.setNumThreads(0)

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def weights_init(module: nn.Module) -> None:
    """Initialize convolution and batch-normalization layers."""
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


def _state_dict_without_dataparallel(
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    """Return a checkpoint state dict without DataParallel prefixes."""
    if isinstance(module, nn.DataParallel):
        module = module.module

    return module.state_dict()


def _save_draem_ckpt(
    rec: nn.Module,
    seg: nn.Module,
    path: Path,
    args: Any,
    epoch: int,
) -> None:
    """Save the single final DRAEM checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "rec": _state_dict_without_dataparallel(rec),
            "seg": _state_dict_without_dataparallel(seg),
            "thr": {
                "seg": float(args.segmentation_threshold),
                "det": float(args.detection_threshold),
            },
            "epoch": int(epoch),
        },
        str(path),
    )


def _validate_fixed_thresholds(
    args: Any,
) -> tuple[float, float]:
    """Read and validate fixed thresholds from main.py."""
    seg_thr = float(args.segmentation_threshold)
    det_thr = float(args.detection_threshold)

    if not 0.0 <= seg_thr <= 1.0:
        raise ValueError(
            "segmentation_threshold must be in [0, 1], "
            f"got {seg_thr}"
        )

    if not 0.0 <= det_thr <= 1.0:
        raise ValueError(
            "detection_threshold must be in [0, 1], "
            f"got {det_thr}"
        )

    return seg_thr, det_thr


def _make_loader_kwargs(
    num_workers: int,
) -> dict[str, Any]:
    """Build DataLoader options for worker and no-worker modes."""
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return kwargs


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python from the PyTorch worker seed."""
    del worker_id

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    """Create a reproducible DataLoader generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    """Warm up once, then apply the 80% and 90% LR decays."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be >= 1")

    warmup_epochs = max(
        0,
        min(int(warmup_epochs), total_epochs),
    )

    milestone_1 = max(
        1,
        int(0.8 * total_epochs),
    )

    milestone_2 = max(
        milestone_1 + 1,
        int(0.9 * total_epochs),
    )

    milestone_2 = min(
        milestone_2,
        total_epochs,
    )

    def lr_factor(epoch_index: int) -> float:
        epoch_number = epoch_index + 1

        if (
            warmup_epochs > 0
            and epoch_number <= warmup_epochs
        ):
            return epoch_number / float(warmup_epochs)

        if epoch_number >= milestone_2:
            return 0.04

        if epoch_number >= milestone_1:
            return 0.20

        return 1.0

    return LambdaLR(
        optimizer,
        lr_lambda=lr_factor,
    )


def _append_hybrid_pixels(
    gt_masks: torch.Tensor,
    prob_map: torch.Tensor,
    y_true_parts: list[np.ndarray],
    y_score_parts: list[np.ndarray],
) -> None:
    """Collect pixels using the existing hybrid evaluation rule."""
    positives = gt_masks > 0.5

    for batch_index in range(gt_masks.size(0)):
        positive_mask = positives[
            batch_index,
            0,
        ]

        if positive_mask.any():
            scores = (
                prob_map[
                    batch_index,
                    0,
                ][positive_mask]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            y_true_parts.append(
                np.ones(
                    scores.size,
                    dtype=np.uint8,
                )
            )

            y_score_parts.append(
                scores.astype(np.float32)
            )
        else:
            scores = (
                prob_map[
                    batch_index,
                    0,
                ]
                .detach()
                .float()
                .cpu()
                .numpy()
                .ravel()
            )

            y_true_parts.append(
                np.zeros(
                    scores.size,
                    dtype=np.uint8,
                )
            )

            y_score_parts.append(
                scores.astype(np.float32)
            )


def _save_validation_artifacts(
    args: Any,
    images: torch.Tensor,
    prob_map: torch.Tensor,
    paths: list[str] | tuple[str, ...],
    seg_thr: float,
) -> None:
    """Save validation images only when main.py enables them."""
    if not bool(
        getattr(
            args,
            "save_val_artifacts",
            False,
        )
    ):
        return

    binary_dir = Path(args.binary_dir)
    heatmap_dir = Path(args.heatmap_dir)
    anomaly_map_dir = Path(args.anomalymap_dir)

    binary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    heatmap_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    anomaly_map_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index in range(images.size(0)):
        name = Path(paths[index]).stem

        prob_cpu = (
            prob_map[
                index,
                0,
            ]
            .detach()
            .float()
            .cpu()
        )

        save_binary_anomaly_map(
            prob_cpu,
            thr=seg_thr,
            save_path=(
                binary_dir
                / f"{name}_mask_bw.png"
            ),
            save_color_path=(
                binary_dir
                / f"{name}_mask_color.png"
            ),
            fg_rgb=(255, 255, 0),
            bg_rgb=(10, 20, 60),
        )

        heat_rgb = overlay_heatmap_on_image(
            images[index],
            prob_cpu,
        )

        cv2.imwrite(
            str(
                heatmap_dir
                / f"{name}_heat.png"
            ),
            cv2.cvtColor(
                heat_rgb,
                cv2.COLOR_RGB2BGR,
            ),
        )

        overlay_rgb = overlay_anomaly_on_rgb(
            images[index],
            prob_cpu,
            thr=seg_thr,
        )

        cv2.imwrite(
            str(
                anomaly_map_dir
                / f"{name}_overlay.png"
            ),
            cv2.cvtColor(
                overlay_rgb,
                cv2.COLOR_RGB2BGR,
            ),
        )


def _metric_from_result(
    metric_key: str,
    metrics: dict[str, float],
) -> float:
    """Read the metric used only for early stopping."""
    normalized_key = metric_key.lower()

    aliases = {
        "dice": "dice",
        "f1": "f1",
        "ap": "ap",
        "auc": "auc",
        "auroc": "auc",
    }

    if normalized_key not in aliases:
        raise ValueError(
            "early_stop_metric must be one of: "
            "dice, f1, ap, auc, auroc"
        )

    return float(
        metrics[
            aliases[normalized_key]
        ]
    )


def train(
    args: Any,
) -> dict[str, Any]:
    """Train DRAEM with fixed thresholds and one final checkpoint."""
    model_name = str(
        getattr(
            args,
            "model",
            "draem",
        )
    ).lower()

    if model_name != "draem":
        raise NotImplementedError(
            "This cleaned train.py contains the DRAEM training path. "
            f"Received model={model_name!r}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DRAEM training requires at least one CUDA device."
        )

    device = torch.device("cuda:0")
    gpu_count = torch.cuda.device_count()

    seg_thr_fixed, det_thr_fixed = (
        _validate_fixed_thresholds(args)
    )

    print(
        "[thresholds] fixed for entire DRAEM run: "
        f"seg={seg_thr_fixed:.4f}, "
        f"det={det_thr_fixed:.4f}"
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_checkpoint_path = (
        checkpoint_dir
        / getattr(
            args,
            "checkpoint_name",
            "draem_final.pt",
        )
    )

    Path(
        args.plot_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    max_silhouettes = int(
        getattr(
            args,
            "max_silhouettes",
            50,
        )
    )

    ds_clean = TrainDataset(
        args.img_root,
        args.sil_root,
        args.mask_root,
        resize=(256, 256),
        transforms_img=args.train_img_tf,
        transforms_mask=args.train_msk_tf,
        max_silhouettes=max_silhouettes,
        apply_aug=False,
        cache_silhouettes=False,
        save_synthetic=False,
        save_reconstructed=False,
        mode="clean",
        p_clean=1.0,
    )


    ds_anom = TrainDataset(
        args.img_root,
        args.sil_root,
        args.mask_root,
        resize=(256, 256),
        transforms_img=args.train_img_tf,
        transforms_mask=args.train_msk_tf,
        max_silhouettes=max_silhouettes,

        apply_aug=False,

        cache_silhouettes=True,

        save_synthetic=bool(
            getattr(
                args,
                "save_synthetic_train",
                False,
            )
        ),

        save_reconstructed=False,
        mode="anom",
        p_clean=0.0,
    )

    val_ds = TestDataset(
        args.imag_non_empty_dir,
        args.masks_path,
        transforms_img=args.val_img_tf,
        transforms_mask=args.val_msk_tf,
    )

    batch_total = int(
        args.batch_size
    )

    if batch_total < 2:
        raise ValueError(
            "batch_size must be >= 2 for "
            "50/50 clean/anomaly batches."
        )

    if batch_total % gpu_count != 0:
        adjusted_batch = (
            (
                batch_total
                + gpu_count
                - 1
            )
            // gpu_count
        ) * gpu_count

        print(
            f"[batch] adjusted global batch "
            f"from {batch_total} "
            f"to {adjusted_batch} "
            f"for {gpu_count} GPUs"
        )

        batch_total = adjusted_batch

    batch_clean = (
        batch_total // 2
    )

    batch_anom = (
        batch_total
        - batch_clean
    )

    num_workers = int(
        getattr(
            args,
            "num_workers",
            max(
                2,
                gpu_count * 2,
            ),
        )
    )

    loader_kwargs = (
        _make_loader_kwargs(
            num_workers
        )
    )

    seed = int(
        getattr(
            args,
            "seed",
            42,
        )
    )

    dl_clean = DataLoader(
        ds_clean,
        batch_size=batch_clean,
        shuffle=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=_make_generator(
            seed + 1
        ),
        **loader_kwargs,
    )

    dl_anom = DataLoader(
        ds_anom,
        batch_size=batch_anom,
        shuffle=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=_make_generator(
            seed + 2
        ),
        **loader_kwargs,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_total,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=_make_generator(
            seed + 3
        ),
        **loader_kwargs,
    )

    steps_per_epoch = max(
    1,
    len(ds_clean) // batch_total,
    )

    print(
        f"[training] dataset={len(ds_clean)}, "
        f"batch={batch_total} "
        f"({batch_clean} clean + {batch_anom} anomaly), "
        f"steps/epoch={steps_per_epoch}"
    )

    if steps_per_epoch <= 0:
        raise RuntimeError(
            "No training steps are available. "
            "Check dataset sizes, batch size, and drop_last."
        )

    rec = ReconstructiveSubNetwork(
        3,
        3,
    ).to(device)

    seg = DiscriminativeSubNetwork(
        6,
        2,
    ).to(device)

    rec.apply(
        weights_init
    )

    seg.apply(
        weights_init
    )

    if gpu_count > 1:
        print(
            "[info] DataParallel on discriminative network "
            f"across {gpu_count} GPUs"
        )

        seg = nn.DataParallel(
            seg,
            device_ids=list(
                range(gpu_count)
            ),
            output_device=0,
        )

    seg_core = (
        seg.module
        if isinstance(
            seg,
            nn.DataParallel,
        )
        else seg
    )

    reconstruction_loss = (
        nn.MSELoss()
    )

    ssim_loss = SSIM()
    focal_loss = FocalLoss()

    detection_loss = (
        nn.BCEWithLogitsLoss()
    )

    seg_parameters = (
        list(
            seg_core.encoder_segment.parameters()
        )
        + list(
            seg_core.decoder_segment.parameters()
        )
        + list(
            seg_core.reduce_channels.parameters()
        )
        + list(
            seg_core.tau_head.parameters()
        )
    )

    optimizer = AdamW(
        [
            {
                "params": rec.parameters()
            },
            {
                "params": seg_parameters
            },
        ],
        lr=float(args.lr),
        weight_decay=float(
            args.weight_decay
        ),
    )

    scheduler = _build_scheduler(
        optimizer,
        total_epochs=int(
            args.epochs
        ),
        warmup_epochs=int(
            getattr(
                args,
                "warmup_epochs",
                0,
            )
        ),
    )


    losses_tr: list[float] = []
    det_bce: list[float] = []

    seg_epochs: list[int] = []
    seg_dice_hist: list[float] = []
    seg_auc_hist: list[float] = []
    seg_f1_hist: list[float] = []
    seg_ap_hist: list[float] = []

    det_epochs: list[int] = []
    det_f1_hist: list[float] = []
    det_auc_hist: list[float] = []
    det_ap_hist: list[float] = []

    seg_eval_stride = max(
        1,
        int(
            getattr(
                args,
                "seg_eval_stride",
                1,
            )
        ),
    )

    metric_key = str(
        getattr(
            args,
            "early_stop_metric",
            "ap",
        )
    ).lower()

    monitor_after = int(
        getattr(
            args,
            "monitor_after",
            1,
        )
    )

    patience = int(
        getattr(
            args,
            "patience",
            0,
        )
    )

    min_delta = float(
        getattr(
            args,
            "min_delta",
            0.0,
        )
    )

    best_val = -math.inf
    no_improvement_count = 0
    completed_epoch = 0

    warned_empty_anomaly = False

    for epoch_index in range(
        int(args.epochs)
    ):
        epoch_number = (
            epoch_index + 1
        )

        completed_epoch = (
            epoch_number
        )

        rec.train()
        seg.train()

        running_total_loss = 0.0
        running_detection_bce = 0.0
        seen_batches = 0

        clean_iterator = iter(
            dl_clean
        )

        anomaly_iterator = iter(
            dl_anom
        )

        progress_bar = tqdm(
            range(
                steps_per_epoch
            ),
            total=steps_per_epoch,
            desc=(
                f"Epoch "
                f"{epoch_number}/"
                f"{args.epochs}"
            ),
            leave=False,
        )

        for _ in progress_bar:
            io_start = (
                time.perf_counter()
            )

            (
                c_img,
                c_msk,
                c_syn,
                c_lbl,
            ) = next(
                clean_iterator
            )

            (
                a_img,
                a_msk,
                a_syn,
                a_lbl,
            ) = next(
                anomaly_iterator
            )

            io_end = (
                time.perf_counter()
            )

            if not warned_empty_anomaly:
                missing_label = bool(
                    (
                        a_lbl.view(-1)
                        < 0.5
                    )
                    .any()
                    .item()
                )

                missing_mask = bool(
                    (
                        a_msk
                        .flatten(1)
                        .sum(dim=1)
                        <= 0
                    )
                    .any()
                    .item()
                )

                if (
                    missing_label
                    or missing_mask
                ):
                    print(
                        "[WARN] TrainDataset(mode='anom') produced "
                        "at least one empty anomalous sample. "
                        "This should be fixed in train_dataset.py "
                        "because it weakens the intended 50/50 batch."
                    )

                    warned_empty_anomaly = True

            clean = torch.cat(
                [
                    c_img,
                    a_img,
                ],
                dim=0,
            )

            masks = torch.cat(
                [
                    c_msk,
                    a_msk,
                ],
                dim=0,
            )

            synth = torch.cat(
                [
                    c_syn,
                    a_syn,
                ],
                dim=0,
            )

            labels = torch.cat(
                [
                    c_lbl,
                    a_lbl,
                ],
                dim=0,
            )

            permutation = torch.randperm(
                clean.size(0)
            )

            clean = (
                clean[
                    permutation
                ]
                .contiguous()
            )

            masks = (
                masks[
                    permutation
                ]
                .contiguous()
            )

            synth = (
                synth[
                    permutation
                ]
                .contiguous()
            )

            labels = (
                labels[
                    permutation
                ]
                .contiguous()
            )

            clean = clean.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            synth = synth.to(
                device,
                non_blocking=True,
            )

            labels = (
                labels
                .to(
                    device,
                    non_blocking=True,
                )
                .view(
                    -1,
                    1,
                )
                .float()
            )

            transfer_end = (
                time.perf_counter()
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            reconstructed = rec(
                synth
            )

            seg_input = torch.cat(
                [
                    reconstructed,
                    synth,
                ],
                dim=1,
            ).contiguous()

            (
                seg_logits,
                ssp_loss,
                _,
                tau_logits,
            ) = seg(
                seg_input
            )

            seg_probabilities = torch.softmax(
                seg_logits,
                dim=1,
            )

            segmentation_loss = (
                reconstruction_loss(
                    reconstructed.float(),
                    clean.float(),
                )
                + ssim_loss(
                    reconstructed.float(),
                    clean.float(),
                )
                + focal_loss(
                    seg_probabilities.float(),
                    masks.float(),
                )
                + 0.1 * ssp_loss.float()
            )

            seg_scalar = segmentation_loss.mean()

            if float(args.det_loss_w) > 0:
                tau_scalar = detection_loss(
                    tau_logits.float(),
                    labels.float(),
                ).mean()

                total_loss = (
                    seg_scalar
                    + float(args.det_loss_w)
                    * tau_scalar
                )
            else:
                tau_scalar = torch.zeros(
                    (),
                    dtype=seg_scalar.dtype,
                    device=seg_scalar.device,
                )

                total_loss = seg_scalar

            forward_end = time.perf_counter()

            total_loss.backward()
            optimizer.step()

            backward_end = time.perf_counter()


            running_detection_bce += float(
                tau_scalar.item()
            )

            running_total_loss += float(
                total_loss.item()
            )

            seen_batches += 1

            progress_bar.set_postfix(
                {
                    "io_ms": int(
                        (
                            io_end
                            - io_start
                        )
                        * 1e3
                    ),
                    "h2d_ms": int(
                        (
                            transfer_end
                            - io_end
                        )
                        * 1e3
                    ),
                    "fwd_ms": int(
                        (
                            forward_end
                            - transfer_end
                        )
                        * 1e3
                    ),
                    "bwd_ms": int(
                        (
                            backward_end
                            - forward_end
                        )
                        * 1e3
                    ),
                    "lr": (
                        f"{optimizer.param_groups[0]['lr']:.2e}"
                    ),
                }
            )

        if seen_batches == 0:
            raise RuntimeError(
                "Training epoch completed with zero batches."
            )

        losses_tr.append(
            running_total_loss
            / seen_batches
        )

        det_bce.append(
            running_detection_bce
            / seen_batches
        )

        scheduler.step()

        rec.eval()
        seg.eval()

        do_seg_eval = (
            (
                epoch_number
                % seg_eval_stride
            )
            == 0
            or epoch_number
            == int(
                args.epochs
            )
        )

        det_label_parts: list[
            torch.Tensor
        ] = []

        det_prob_parts: list[
            torch.Tensor
        ] = []

        y_true_parts: list[
            np.ndarray
        ] = []

        y_score_parts: list[
            np.ndarray
        ] = []

        with torch.no_grad():
            validation_bar = tqdm(
                val_dl,
                desc=f"Val {epoch_number}/{args.epochs}",
                leave=False,
            )

            for (
                images,
                gt_masks,
                paths,
                patch_labels,
            ) in validation_bar:
                images = images.to(
                    device,
                    non_blocking=True,
                )

                gt_masks = gt_masks.to(
                    device,
                    non_blocking=True,
                )

                patch_labels = patch_labels.to(
                    device,
                    non_blocking=True,
                )

                reconstructed_val = rec(
                    images
                )

                (
                    seg_logits,
                    _,
                    _,
                    tau_logits,
                ) = seg(
                    torch.cat(
                        [
                            reconstructed_val,
                            images,
                        ],
                        dim=1,
                    )
                )

                prob_map = torch.softmax(
                    seg_logits,
                    dim=1,
                )[:, 1:2]

                if (
                    prob_map.shape[2:]
                    != gt_masks.shape[2:]
                ):
                    prob_map = F.interpolate(
                        prob_map,
                        size=gt_masks.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

   
                det_label_parts.append(
                    patch_labels
                    .detach()
                    .view(-1)
                    .cpu()
                )

                det_prob_parts.append(
                    torch.sigmoid(
                        tau_logits
                    )
                    .detach()
                    .view(-1)
                    .cpu()
                )


                if do_seg_eval:
                    _append_hybrid_pixels(
                        gt_masks,
                        prob_map,
                        y_true_parts,
                        y_score_parts,
                    )

                _save_validation_artifacts(
                    args,
                    images,
                    prob_map,
                    paths,
                    seg_thr_fixed,
                )

        if (
            det_label_parts
            and det_prob_parts
        ):
            det_labels_np = (
                torch.cat(
                    det_label_parts
                )
                .numpy()
                .astype(
                    np.int64
                )
            )

            det_probs_np = (
                torch.cat(
                    det_prob_parts
                )
                .numpy()
                .astype(
                    np.float32
                )
            )

            det_metrics = (
                evaluate_detection_metrics(
                    det_labels_np,
                    det_probs_np,
                    thresh=det_thr_fixed,
                )
            )

            det_epochs.append(
                epoch_number
            )

            det_f1_hist.append(
                det_metrics["f1"]
            )

            det_auc_hist.append(
                det_metrics["auc"]
            )

            det_ap_hist.append(
                det_metrics["ap"]
            )

            print(
                f"[VAL det @fixed {det_thr_fixed:.4f}] "
                f"F1={det_metrics['f1']:.4f} "
                f"AP={det_metrics['ap']:.4f} "
                f"AUC={det_metrics['auc']:.4f} "
                f"P={det_metrics['precision']:.4f} "
                f"R={det_metrics['recall']:.4f}"
            )

        current_seg_metrics: (
            dict[str, float]
            | None
        ) = None

        if (
            do_seg_eval
            and y_true_parts
            and y_score_parts
        ):
            y_true_all = (
                np.concatenate(
                    y_true_parts
                )
                .astype(
                    np.uint8
                )
            )

            y_score_all = (
                np.concatenate(
                    y_score_parts
                )
                .astype(
                    np.float32
                )
            )

            current_seg_metrics = (
                compute_global_seg_metrics(
                    y_true_all,
                    y_score_all,
                    seg_thr_fixed,
                )
            )

            seg_epochs.append(
                epoch_number
            )

            seg_dice_hist.append(
                current_seg_metrics[
                    "dice"
                ]
            )

            seg_f1_hist.append(
                current_seg_metrics[
                    "f1"
                ]
            )

            seg_ap_hist.append(
                current_seg_metrics[
                    "ap"
                ]
            )

            seg_auc_hist.append(
                current_seg_metrics[
                    "auc"
                ]
            )

            print(
                f"[VAL seg @fixed {seg_thr_fixed:.4f}] "
                f"Dice={current_seg_metrics['dice']:.4f} "
                f"F1={current_seg_metrics['f1']:.4f} "
                f"AP={current_seg_metrics['ap']:.4f} "
                f"AUROC={current_seg_metrics['auc']:.4f}"
            )

        if (
            current_seg_metrics
            is not None
            and epoch_number
            >= monitor_after
            and patience > 0
        ):
            monitor_value = (
                _metric_from_result(
                    metric_key,
                    current_seg_metrics,
                )
            )

            print(
                f"[monitor] "
                f"{metric_key}="
                f"{monitor_value:.4f} "
                f"(best={best_val:.4f})"
            )

            if (
                math.isfinite(
                    monitor_value
                )
                and monitor_value
                > best_val
                + min_delta
            ):
                best_val = (
                    monitor_value
                )

                no_improvement_count = 0
            else:
                no_improvement_count += 1

                if (
                    no_improvement_count
                    >= patience
                ):
                    print(
                        "[early-stop] patience exhausted "
                        "on fresh segmentation evaluations"
                    )

                    break

    if completed_epoch <= 0:
        raise RuntimeError(
            "Training ended before completing an epoch."
        )

    _save_draem_ckpt(
        rec,
        seg,
        final_checkpoint_path,
        args,
        completed_epoch,
    )

    print(
        "[checkpoint] saved single final checkpoint: "
        f"{final_checkpoint_path}"
    )

    print(
        "[checkpoint] fixed threshold metadata: "
        f"seg={seg_thr_fixed:.4f}, "
        f"det={det_thr_fixed:.4f}"
    )

    plot_all_training_metrics(
        losses_tr,
        seg_dice_hist,
        seg_auc_hist,
        seg_f1_hist,
        seg_ap_hist,
        det_bce,
        det_f1_hist,
        det_auc_hist,
        det_ap_hist,
        args.plot_dir,
        seg_epochs=seg_epochs,
        det_epochs=det_epochs,
    )

    rec.eval()
    seg.eval()

    print(
        "Training complete after "
        f"{completed_epoch} epoch(s). "
        "No automatic threshold search was performed."
    )

    return {
        "val_images": torch.empty(0),
        "pred_masks": torch.empty(0),
        "image_paths": [],
        "best_ckpt": str(
            final_checkpoint_path
        ),
        "best_val": best_val,
        "seg_dice": seg_dice_hist,
        "det_f1": det_f1_hist,
    }