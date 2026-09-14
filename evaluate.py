# eValuation
import os, csv, torch, torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import cv2
import numpy as np                 
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
import matplotlib.pyplot as _plt
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM
import torch.nn.functional as F
from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from test_dataset import TestDataset
from utils import overlay_heatmap_on_image, overlay_anomaly_on_rgb, save_binary_anomaly_map, _plot_pr_curve,_plot_roc_curve, plot_seg_threshold_sweep, plot_seg_threshold_sweep, gaussian_peak_heatmap
from torch.utils.data import Sampler
from Padim.padim_adapter import PaDiMModel_padim , gaussian_blur_map
from Padim.padim_nf_adapter import PaDiMNFAdapter
from threshold_tunning import apply_calibration
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, average_precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from metrics import compute_global_seg_metrics,evaluate_detection_metrics

class StratifiedBatchSampler(Sampler):
    """
    Yields batches with at least `pos_per_batch` positives (if available),
    filling the rest with negatives. Preserves overall class imbalance.
    """
    def __init__(self, labels, batch_size, pos_per_batch=1, shuffle=True, seed=42):
        self.labels = np.asarray(labels).astype(int)
        self.batch_size = int(batch_size)
        self.pos_per_batch = int(pos_per_batch)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.pos_idx = np.where(self.labels == 1)[0].tolist()
        self.neg_idx = np.where(self.labels == 0)[0].tolist()

    def __iter__(self):
        pos = self.pos_idx[:]
        neg = self.neg_idx[:]
        if self.shuffle:
            self.rng.shuffle(pos); self.rng.shuffle(neg)

        p = n = 0
        total = len(pos) + len(neg)
        emitted = 0
        while emitted < total:
            k_pos = min(self.pos_per_batch, max(0, len(pos) - p))
            k_neg = min(self.batch_size - k_pos, max(0, len(neg) - n))
            if k_pos + k_neg == 0:
                break
            batch = pos[p:p+k_pos] + neg[n:n+k_neg]
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch
            p += k_pos; n += k_neg; emitted += (k_pos + k_neg)

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))



def load_checkpoint(model, ckpt_path, model_name='rec'):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['rec'] if model_name == 'rec' else ckpt['seg']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model, ckpt.get('thr', {})  


@torch.no_grad()
def sniff_ckpt_type(ckpt_path: str) -> str:
    """
    Returns one of: 'draem', 'padim', 'padim_nf', or 'unknown'
    """
    p = Path(ckpt_path)

    if p.is_dir():
        ckpt_path = resolve_ckpt_path(ckpt_path, prefer=("best_padim_nf*.pt","best_padim*.pt","best_*.pt","*.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ("rec" in ckpt) or ("seg" in ckpt):
        return "draem"
    if "padim_nf" in ckpt:
        return "padim_nf"
    if ("padim" in ckpt) or ("padim_vanilla" in ckpt):
        return "padim"
    return "unknown"


@torch.no_grad()
def load_padim_from_ckpt(ckpt_path: str, device: torch.device):  
    ckpt_path = resolve_ckpt_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    key = "padim"  
    if key not in ckpt:
        raise KeyError("Checkpoint has no 'padim' state. Expected Defard-vanilla PaDiM.")
    state = ckpt[key]
    thr   = ckpt.get("thr", {})

    if "prec" not in state:  
        raise KeyError("This checkpoint is not vanilla PaDiM (missing 'prec').")

    model = PaDiMModel_padim(  
        backbone = state.get("backbone", "wide_resnet50_2"),
        n_select = int(state["n_select"]),
        img_size = tuple(state["img_size"]),
        device   = str(device),
    )
    model.load_state_dict_padim(state)  
    model.eval()
    return model, thr
@torch.no_grad()
def load_padim_nf_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt_path = resolve_ckpt_path(ckpt_path, prefer=("best_padim_nf*.pt","*.pt"))
    ckpt = torch.load(ckpt_path, map_location=device)
    if "padim_nf" not in ckpt:
        raise KeyError("Checkpoint has no 'padim_nf' key.")
    state = ckpt["padim_nf"]
    thr   = ckpt.get("thr", {})

    model = PaDiMNFAdapter(
        backbone = state.get("backbone","wide_resnet50_2"),
        n_select = int(state.get("n_select",550)),
        img_size = tuple(state["img_size"]),
        nf_type  = state.get("nf_type","maf"),
        n_heads  = int(state.get("n_heads",1)),
        device   = str(device),
    )
    model.load_state_dict_nf(state)
    model.eval()
    return model, thr

def pick_best_padim_threshold(y_true_all, y_score_all, for_metric="f1"):
    # threshold sweep via PR curve
    p, r, t = precision_recall_curve(y_true_all, y_score_all)
    f1 = (2*p*r)/(p+r+1e-6)
    print("TEST best F1 achievable:", float(np.nanmax(f1)))
    
    if for_metric.lower() in ("dice", "f1"):
        f1_arr = (2 * p * r) / (p + r + 1e-6)
        best_idx = int(np.nanargmax(f1_arr))
        best_thr = 0.0 if best_idx == 0 else (1.0 if best_idx >= len(t) else float(t[best_idx-1]))
    else:
        best_thr = 0.5  
    
    print(f"Best Threshold for {for_metric}: {best_thr}")
    return best_thr
from sklearn.metrics import precision_recall_fscore_support
@torch.no_grad()
def padim_nf_threshold_sweep_plot(
    ckpt_path: str,
    img_dir: str,
    mask_dir: str,
    png_out: str,
    *,
    device: str = "cuda",
    batch: int = 8,
    seg_thr_override: float | None = None,
    max_grid: int = 400,
) -> tuple[float, float]:
    """Sweep the PaDiM-NF patch-level detection threshold."""

    dev = torch.device(
        device
        if torch.cuda.is_available()
        else "cpu"
    )

    nf, thr_meta = load_padim_nf_from_ckpt(
        ckpt_path,
        dev,
    )

    seg_thr = float(
        np.clip(
            (
                seg_thr_override
                if seg_thr_override is not None
                else thr_meta.get(
                    "seg",
                    0.5,
                )
            ),
            0.0,
            1.0,
        )
    )

    det_thr_ckpt = float(
        np.clip(
            thr_meta.get(
                "det",
                0.5,
            ),
            0.0,
            1.0,
        )
    )

    calib = thr_meta.get(
        "calib",
        {
            "type": "minmax",
        },
    )

    tf_img = T.Compose(
        [
            T.Resize(
                (256, 256)
            ),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    tf_mask = T.Compose(
        [
            T.Resize(
                (256, 256),
                interpolation=T.InterpolationMode.NEAREST,
            ),
            T.ToTensor(),
        ]
    )

    ds = TestDataset(
        img_dir,
        mask_dir,
        tf_img,
        tf_mask,
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=4,
    )

    det_labels = []
    det_scores = []

    for (
        imgs,
        gts,
        _,
        lbl,
    ) in tqdm(
        dl,
        desc="sweep-padim-nf",
    ):
        imgs = imgs.to(
            dev
        )

        gts = gts.to(
            dev
        )

        pmap_raw = nf.predict_maps(
            imgs,
            normalize=False,
        )

        if (
            pmap_raw.shape[2:]
            != gts.shape[2:]
        ):
            pmap_raw = F.interpolate(
                pmap_raw,
                size=gts.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        batch_size_actual = (
            pmap_raw.size(0)
        )

        for index in range(
            batch_size_actual
        ):
            raw_np = (
                pmap_raw[
                    index,
                    0,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            prob_np = apply_calibration(
                raw_np,
                calib,
            )

            im_score = float(
                np.percentile(
                    prob_np,
                    95,
                )
            )

            det_scores.append(
                im_score
            )

        det_labels.append(
            lbl.cpu().numpy()
        )

    if not det_scores:
        raise RuntimeError(
            "No PaDiM-NF detection scores collected."
        )

    y_true = np.concatenate(
        det_labels
    ).astype(
        int
    )

    y_score = np.asarray(
        det_scores,
        dtype=float,
    )

    lo = float(
        np.nanmin(
            y_score
        )
    )

    hi = float(
        np.nanmax(
            y_score
        )
    )

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or np.isclose(
            lo,
            hi,
        )
    ):
        threshold_grid = np.array(
            [
                (
                    lo
                    if np.isfinite(lo)
                    else 0.0
                )
            ],
            dtype=float,
        )

    else:
        threshold_grid = np.linspace(
            lo,
            hi,
            max_grid,
        )

    f1_values = []

    for threshold in threshold_grid:
        predictions = (
            y_score >= threshold
        ).astype(
            int
        )

        (
            _,
            _,
            f1,
            _,
        ) = precision_recall_fscore_support(
            y_true,
            predictions,
            average="binary",
            zero_division=0,
        )

        f1_values.append(
            float(f1)
        )

    f1_values = np.asarray(
        f1_values,
        dtype=float,
    )

    best_index = int(
        np.nanargmax(
            f1_values
        )
    )

    best_threshold = float(
        threshold_grid[
            best_index
        ]
    )

    best_f1 = float(
        f1_values[
            best_index
        ]
    )

    Path(
        png_out
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(6, 4)
    )

    plt.plot(
        threshold_grid,
        f1_values,
        label="F1",
    )

    plt.axvline(
        best_threshold,
        linestyle=":",
        label=(
            f"best="
            f"{best_threshold:.3f}"
        ),
    )

    plt.title(
        "PaDiM-NF detection threshold sweep"
    )

    plt.xlabel(
        "threshold"
    )

    plt.ylabel(
        "F1"
    )

    plt.grid(
        True
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        png_out
    )

    plt.close()

    print(
        "[PaDiM-NF] "
        f"stored det threshold="
        f"{det_thr_ckpt:.3f}"
    )

    print(
        "[PaDiM-NF] "
        f"best test threshold="
        f"{best_threshold:.3f}, "
        f"F1={best_f1:.4f}"
    )

    return (
        best_threshold,
        best_f1,
    )

from scipy import ndimage as ndi
def evaluate_padim_metrics(y_true_all, y_score_all, threshold):
    y_pred = (y_score_all > threshold).astype(np.uint8)
    TP = np.sum((y_pred == 1) & (y_true_all == 1))
    FP = np.sum((y_pred == 1) & (y_true_all == 0))
    FN = np.sum((y_pred == 0) & (y_true_all == 1))

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    dice      = 2 * TP / (2 * TP + FP + FN + 1e-6)

 
    try:
        ap  = average_precision_score(y_true_all, y_score_all)
        auc = roc_auc_score(y_true_all, y_score_all)
    except ValueError:
        ap, auc = 0.0, 0.0

    return {"dice": float(dice), "f1": float(f1), "ap": float(ap), "auc": float(auc)}

def resolve_ckpt_path(ckpt_path: str, prefer: tuple[str,...]=("best_padim_ep1.pt","best_*.pt","*.pt")) -> str:
    p = Path(ckpt_path)
    if p.is_file():
        return str(p)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path is not a file: {ckpt_path}")

    for pat in prefer:
        cands = sorted(p.glob(pat))
        if cands:
            return str(cands[0])
    raise FileNotFoundError(f"No .pt files found in directory: {ckpt_path}")



import pathlib
import importlib as _importlib
_Path = _importlib.import_module("pathlib").Path

def safe_minmax_per_image(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:  # change
    """
    x: [B,1,H,W]
    If dynamic range is too small, return zeros for that image to avoid
    amplifying interpolation/patch-grid artifacts into checkerboards.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
    B = x.size(0)
    outs = []
    for b in range(B):
        v = x[b:b+1]
        vmin = v.amin()
        vmax = v.amax()
        if (vmax - vmin) < eps:
            outs.append(torch.zeros_like(v))
        else:
            outs.append((v - vmin) / (vmax - vmin))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def run_test(
    ckpt_path: str,
    img_dir: str,
    mask_dir: str,
    args,
    batch: int = 8,
    device: str = "cuda",
    heatmap_dir: str | None = None,
    binary_dir: str | None = None,
    anomaly_map_dir: str | None = None,
    output_pdf_path: str | None = None,
    csv_file_path: str | None = None,
) -> Dict[str, torch.Tensor | float]:

    model_name = getattr(args, "model", "draem").lower()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if model_name not in ("draem", "padim", "padim_nf"):
        model_name = "auto"
    if model_name == "auto":
        model_name = sniff_ckpt_type(ckpt_path) or "draem"
    sniffed = sniff_ckpt_type(ckpt_path)
    if sniffed != "unknown" and sniffed != model_name:
        print(f"[warn] args.model='{model_name}' but checkpoint looks like '{sniffed}'. Using '{sniffed}'.")
        model_name = sniffed

    # PaDiM-NF BRANCH 
    if model_name == "padim_nf":
        nf, thr_meta = load_padim_nf_from_ckpt(ckpt_path, dev)
        assert getattr(nf, "calibrated", False), "Calibration missing in loaded PaDiM-NF checkpoint."  # change

        seg_thr = float(np.clip(thr_meta.get("seg", 0.5), 0.0, 1.0))
        det_thr = float(np.clip(thr_meta.get("det", 0.5), 0.0, 1.0))
        calib   = thr_meta.get("calib", {"type": "identity"})
        print(f"[TEST-PaDiM-NF] using seg_thr={seg_thr:.3f} det_thr={det_thr:.3f}")

        if getattr(args, "seg_thr_override", None) is not None:
            seg_thr = float(np.clip(args.seg_thr_override, 0.0, 1.0))
        if getattr(args, "det_thr_override", None) is not None:
            det_thr = float(np.clip(args.det_thr_override, 0.0, 1.0))


        tf_img  = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tf_mask = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        dataset = TestDataset(img_dir, mask_dir, tf_img, tf_mask)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)

        det_labels, det_scores = [], []
        y_true_all, y_score_all = [], []
        all_paths, all_scores = [], []


        for imgs, gts, paths, lbl in tqdm(loader, desc="test-2-padim-nf"):
            imgs, gts = imgs.to(dev), gts.to(dev)

     
            pmap_prob = nf.predict_maps(imgs, normalize=True)  


            pmap_raw = nf.predict_maps(imgs, normalize=False)  


            if pmap_raw.shape[2:] != gts.shape[2:]:
                pmap_raw = F.interpolate(pmap_raw, size=gts.shape[2:], mode="bicubic", align_corners=False)  

            pmap_blur = gaussian_blur_map(pmap_raw, sigma=4.0, ksize=7)  

            pmap_prob_vis = safe_minmax_per_image(pmap_blur, eps=1e-4)  

            B = pmap_prob.size(0)
            cur_scores = []

            for b in range(B):

                prob_np = pmap_prob[b, 0].detach().cpu().numpy().astype(np.float32)  

         
                im_score = float(np.percentile(prob_np, 95))
                cur_scores.append(im_score)



                if getattr(args, "save_npy_heatmaps", True):
                    npy_dir = _Path(args.plot_dir) / "NPY_HEATMAPS"
                    npy_dir.mkdir(parents=True, exist_ok=True)
                    img_name = _Path(paths[b]).stem
                    np.save(npy_dir / f"{img_name}_heat.npy", prob_np)


                m = (gts[b, 0].detach().cpu().numpy() > 0.5)
                y_true_all.append(m.astype(np.uint8).ravel())
                y_score_all.append(prob_np.ravel())

               
                prob_vis_np = pmap_prob_vis[b, 0].detach().cpu().numpy().astype(np.float32)  
                gamma_vis = 2.0  
                vis_for_mask = np.clip(prob_vis_np, 0.0, 1.0) ** gamma_vis        
               
                rank = float(np.mean(prob_np <= float(seg_thr)))                 
                rank = float(np.clip(rank, 0.0, 1.0))                             
                vis_thr = 0.8

                if heatmap_dir:
                    heat_rgb = overlay_heatmap_on_image(
                        imgs[b],
                        torch.from_numpy(prob_vis_np),
                        gamma=2.0,              
                        do_autocontrast=False
                    )
                    _Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(_Path(heatmap_dir) / f"{_Path(paths[b]).stem}_heatmap.png"),
                                cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))

               

                if anomaly_map_dir:
                    anom_rgb = overlay_anomaly_on_rgb(
                        imgs[b],
                        torch.from_numpy(vis_for_mask),  
                        thr=float(vis_thr)               
                    )
                    _Path(anomaly_map_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(_Path(anomaly_map_dir) / f"{_Path(paths[b]).stem}_anomap.png"),
                                cv2.cvtColor(anom_rgb, cv2.COLOR_RGB2BGR))

              
                if binary_dir:
                    _Path(binary_dir).mkdir(parents=True, exist_ok=True)
                    save_binary_anomaly_map(
                        torch.from_numpy(vis_for_mask),  
                        thr=float(vis_thr),              
                        save_color_path=_Path(binary_dir) / f"{_Path(paths[b]).stem}_mask_color.png",
                        fg_rgb=(255, 255, 0),
                        bg_rgb=(10, 20, 60),
                    )

            det_labels.append(lbl.cpu().numpy())
            det_scores.append(np.asarray(cur_scores, dtype=float))
            all_paths.extend(list(paths))
            all_scores.extend(cur_scores)

        # --- Concatenate for metrics
        y_true_all  = np.concatenate(y_true_all).astype(np.uint8)    if y_true_all  else np.empty(0, np.uint8)
        y_score_all = np.concatenate(y_score_all).astype(np.float32) if y_score_all else np.empty(0, np.float32)
        det_labels  = np.concatenate(det_labels).astype(int)         if det_labels  else np.empty(0, int)
        det_scores  = np.concatenate(det_scores).astype(float)       if det_scores  else np.empty(0, float)

        assert y_true_all.shape == y_score_all.shape, (
            f"Pixel arrays mismatch: y_true_all={y_true_all.shape}, y_score_all={y_score_all.shape}"
        )

        if not np.isfinite(y_score_all).all():
            raise ValueError("y_score_all contains NaN/Inf values.")
        min_s, max_s = (float(y_score_all.min()) if y_score_all.size else 0.0,
                        float(y_score_all.max()) if y_score_all.size else 0.0)
        if (min_s < 0.0 - 1e-6) or (max_s > 1.0 + 1e-6):
            print(f"[WARN] pixel scores out of [0,1]: min={min_s:.4f} max={max_s:.4f}")

        if y_true_all.size == 0 or y_score_all.size == 0:
            print("[WARN] No pixel data collected; skipping segmentation metrics/sweep.")

        if det_labels.size and det_scores.size:
            if not np.isfinite(det_scores).all():
                raise ValueError("det_scores contains NaN/Inf values.")
        else:
            print("[WARN] No detection data collected; skipping detection metrics.")

        if y_true_all.size and y_score_all.size:
            seg_plot_png = str(_Path(args.plot_dir) / "padim_nf_seg_threshold_sweep.png")
            best_seg_thr = plot_seg_threshold_sweep(y_true_all, y_score_all, seg_plot_png)
            print(f"[PaDiM-NF][TEST] seg sweep: best_thr={best_seg_thr:.3f}  → saved: {seg_plot_png}")
        else:
            print("[PaDiM-NF][TEST] seg sweep skipped (no pixel data).")


        if getattr(args, "resweep_thresholds", False):
            if y_true_all.size:
                seg_thr = float(np.clip(pick_best_padim_threshold(y_true_all, y_score_all, for_metric="f1"), 0.0, 1.0))
            if det_labels.size:
                det_thr = float(np.clip(evaluate_detection_metrics(det_labels, det_scores, thresh=None)["best_thr"], 0.0, 1.0))

        seg_m = evaluate_padim_metrics(y_true_all, y_score_all, seg_thr) if y_true_all.size else {
            "dice": float("nan"), "f1": float("nan"), "ap": float("nan"), "auc": float("nan")}

 
        if y_true_all.size and y_score_all.size:
            try:
                from sklearn.metrics import precision_recall_fscore_support
                seg_pred = (y_score_all >= float(seg_thr)).astype(np.uint8)
                seg_prec, seg_rec, _, _ = precision_recall_fscore_support(
                    y_true_all.astype(np.uint8), seg_pred, average="binary", zero_division=0
                )
            except Exception as e:
                print(f"[WARN][PaDiM-NF] seg precision/recall failed: {e}")
                seg_prec = float("nan"); seg_rec = float("nan")
        else:
            seg_prec = float("nan"); seg_rec = float("nan")

        det_m = evaluate_detection_metrics(det_labels, det_scores, thresh=det_thr) if det_labels.size else {
            "f1": float("nan"), "ap": float("nan"), "auc": float("nan"), "best_thr": det_thr,
            "precision": float("nan"), "recall": float("nan")
        }

 
        if csv_file_path is None:
            csv_file_path = str(_Path(img_dir).parent / "padim_nf_patch_labels.csv")
        with open(csv_file_path, "w", newline="") as csv_f:
            csv_w = csv.writer(csv_f)
            csv_w.writerow(["Patch_ID", "Anomaly_Label"])
            for p, s in zip(all_paths, all_scores):
                csv_w.writerow([_Path(p).name, int(s >= det_thr)])

        print(f"[TEST seg][PaDiM-NF] thr={seg_thr:.4f}  "
              f"Dice={seg_m['dice']:.4f}  F1={seg_m['f1']:.4f}  "
              f"AP={seg_m['ap']:.4f}  AUROC={seg_m['auc']:.4f}  "
              f"P={seg_prec:.4f}  R={seg_rec:.4f}")
        print(f"[TEST det][PaDiM-NF] thr={det_thr:.4f}  "
              f"F1={det_m.get('f1', float('nan')):.4f}  AP={det_m.get('ap', float('nan')):.4f}  "
              f"AUROC={det_m.get('auc', float('nan')):.4f}  "
              f"P={det_m.get('precision', float('nan')):.4f}  R={det_m.get('recall', float('nan')):.4f}")
        try:
            _Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
            print("[PaDiM-NF] plot_dir =", args.plot_dir)
        except Exception as e:
            print("[PaDiM-NF] could not create plot_dir:", e)

        if y_true_all.size and y_score_all.size:
            seg_pr_path  = str(_Path(args.plot_dir) / "pr_segmentation_padim_nf.pdf")
            seg_roc_path = str(_Path(args.plot_dir) / "roc_segmentation_padim_nf.pdf")

            try:
                _plot_pr_curve(
                    y_true=y_true_all.astype(np.uint8).ravel(),
                    y_score=y_score_all.astype(np.float32).ravel(),
                    out_path=seg_pr_path,
                    title="Segmentation PR (PaDiM-NF)",
                    op_thr=float(seg_thr),
                    op_kind="score>=seg_thr",
                )
                print("[PaDiM-NF] saved:", seg_pr_path)
            except Exception as e:
                print("[PaDiM-NF] PR(seg) failed:", e)

            try:
                _plot_roc_curve(
                    y_true=y_true_all.astype(np.uint8).ravel(),
                    y_score=y_score_all.astype(np.float32).ravel(),
                    out_path=seg_roc_path,
                    title="Segmentation ROC (PaDiM-NF)",
                    op_thr=float(seg_thr),
                    op_kind="score>=seg_thr",
                )
                print("[PaDiM-NF] saved:", seg_roc_path)
            except Exception as e:
                print("[PaDiM-NF] ROC(seg) failed:", e)
        else:
            print("[PaDiM-NF] No pixel data for seg PR/ROC.")

        if det_labels.size and det_scores.size:
            det_pr_path  = str(_Path(args.plot_dir) / "pr_detection_padim_nf.pdf")
            det_roc_path = str(_Path(args.plot_dir) / "roc_detection_padim_nf.pdf")

            try:
                _plot_pr_curve(
                    y_true=det_labels.astype(np.uint8).ravel(),
                    y_score=det_scores.astype(np.float32).ravel(),
                    out_path=det_pr_path,
                    title="Detection PR (PaDiM-NF)",
                    op_thr=float(det_thr),
                    op_kind="score>=det_thr",
                )
                print("[PaDiM-NF] saved:", det_pr_path)
            except Exception as e:
                print("[PaDiM-NF] PR(det) failed:", e)

            try:
                _plot_roc_curve(
                    y_true=det_labels.astype(np.uint8).ravel(),
                    y_score=det_scores.astype(np.float32).ravel(),
                    out_path=det_roc_path,
                    title="Detection ROC (PaDiM-NF)",
                    op_thr=float(det_thr),
                    op_kind="score>=det_thr",
                )
                print("[PaDiM-NF] saved:", det_roc_path)
            except Exception as e:
                print("[PaDiM-NF] ROC(det) failed:", e)
        else:
            print("[PaDiM-NF] No patch data for det PR/ROC.")
        return {
            "seg_dice_global": seg_m["dice"],
            "seg_f1_global":   seg_m["f1"],
            "seg_ap_global":   seg_m["ap"],
            "seg_auc_global":  seg_m["auc"],
            "seg_precision_global": seg_prec,
            "seg_recall_global":    seg_rec,
            "det_f1":       det_m.get("f1", float("nan")),
            "det_ap":       det_m.get("ap", float("nan")),
            "det_auroc":    det_m.get("auc", float("nan")),
            "det_precision": det_m.get("precision", float("nan")),
            "det_recall":    det_m.get("recall", float("nan")),
            "images": torch.empty(0),
            "probs": torch.empty(0)
        }
    # PaDiM (vanilla) BRANCH 
    elif model_name == "padim":
        padim, thr_meta = load_padim_from_ckpt(ckpt_path, dev)

        if "seg" in thr_meta:
            seg_thr = float(thr_meta["seg"])
        else:
            seg_thr = float(getattr(args, "segmentation_threshold", 0.25))  

        if "det" in thr_meta:
            det_thr = float(thr_meta["det"])
        else:
            det_thr = float(getattr(args, "detection_threshold", 0.30))     

        if getattr(args, "seg_thr_override", None) is not None:
            seg_thr = float(args.seg_thr_override)
        if getattr(args, "det_thr_override", None) is not None:
            det_thr = float(args.det_thr_override)

        print(f"[TEST-PaDiM] using seg_thr={seg_thr:.3f} det_thr={det_thr:.3f}")


        if getattr(args, "seg_thr_override", None) is not None:
            seg_thr = float(np.clip(args.seg_thr_override, 0.0, 1.0))
        if getattr(args, "det_thr_override", None) is not None:
            det_thr = float(np.clip(args.det_thr_override, 0.0, 1.0))

        tf_img  = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tf_mask = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        dataset = TestDataset(img_dir, mask_dir, tf_img, tf_mask)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)

        det_labels, det_scores = [], []
        y_true_all, y_score_all = [], []
        all_paths, all_scores = [], []


        for imgs, gts, paths, lbl in tqdm(loader, desc="test-2-padim"):
            imgs, gts = imgs.to(dev), gts.to(dev)

            pmap_raw = padim.predict_maps(imgs, normalize=False)

   
            if pmap_raw.shape[2:] != gts.shape[2:]:
                pmap_raw = F.interpolate(
                    pmap_raw, size=gts.shape[2:], mode="bicubic", align_corners=False
                )


            pmap_blur = gaussian_blur_map(pmap_raw, sigma=4.0, ksize=7)

            B = pmap_blur.size(0)
            cur_scores = [] 
            for b in range(B):

                raw_np  = pmap_blur[b, 0].detach().cpu().numpy()
                vmin, vmax = float(raw_np.min()), float(raw_np.max())
                denom = max(vmax - vmin, 1e-8)
                prob_np = (raw_np - vmin) / denom  
                im_score = float(
                    np.percentile(
                        prob_np,
                        95,
                    )
                )

                cur_scores.append(
                    im_score
                )

                m = (gts[b, 0].detach().cpu().numpy() > 0.5)
                y_true_all.append(m.astype(np.uint8).ravel())
                y_score_all.append(prob_np.ravel())

   
                if heatmap_dir:
                    Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
                    heat_rgb = overlay_heatmap_on_image(imgs[b], torch.from_numpy(prob_np))
                    cv2.imwrite(str(Path(heatmap_dir) / f"{Path(paths[b]).stem}_heatmap.png"),
                                cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))

                if anomaly_map_dir:
                    Path(anomaly_map_dir).mkdir(parents=True, exist_ok=True)
                    anom_rgb = overlay_anomaly_on_rgb(imgs[b], torch.from_numpy(prob_np), thr=float(seg_thr))
                    cv2.imwrite(str(Path(anomaly_map_dir) / f"{Path(paths[b]).stem}_anomap.png"),
                                cv2.cvtColor(anom_rgb, cv2.COLOR_RGB2BGR))

                if binary_dir:
                    Path(binary_dir).mkdir(parents=True, exist_ok=True)
                    save_binary_anomaly_map(
                        torch.from_numpy(prob_np),
                        thr=float(seg_thr),
                        save_color_path=Path(binary_dir) / f"{Path(paths[b]).stem}_mask_color.png",
                        fg_rgb=(255, 255, 0),
                        bg_rgb=(10, 20, 60),
                    )

            det_labels.append(lbl.cpu().numpy())
            det_scores.append(np.asarray(cur_scores, dtype=float))
            all_paths.extend(list(paths))
            all_scores.extend(cur_scores)


        y_true_all  = np.concatenate(y_true_all).astype(np.uint8)    if y_true_all  else np.empty(0, np.uint8)
        y_score_all = np.concatenate(y_score_all).astype(np.float32) if y_score_all else np.empty(0, np.float32)
        det_labels  = np.concatenate(det_labels).astype(int)         if det_labels  else np.empty(0, int)
        det_scores  = np.concatenate(det_scores).astype(float)       if det_scores  else np.empty(0, float)
        

        seg_m = evaluate_padim_metrics(y_true_all, y_score_all, seg_thr) if y_true_all.size else {
            "dice": float("nan"), "f1": float("nan"), "ap": float("nan"), "auc": float("nan")}
 

        if (
            y_true_all.size
            and y_score_all.size
        ):
            seg_pred = (
                y_score_all
                >= float(seg_thr)
            ).astype(
                np.uint8
            )

            (
                seg_prec,
                seg_rec,
                seg_f1,
                _,
            ) = precision_recall_fscore_support(
                y_true_all.astype(
                    np.uint8
                ),
                seg_pred,
                average="binary",
                zero_division=0,
            )

            try:
                seg_ap = average_precision_score(
                    y_true_all,
                    y_score_all,
                )

                seg_auc = roc_auc_score(
                    y_true_all,
                    y_score_all,
                )

            except ValueError:
                seg_ap = float("nan")
                seg_auc = float("nan")

            seg_m = {
                "f1": float(seg_f1),
                "precision": float(seg_prec),
                "recall": float(seg_rec),
                "ap": float(seg_ap),
                "auc": float(seg_auc),
            }

        else:
            seg_m = {
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "ap": float("nan"),
                "auc": float("nan"),
            }

            seg_prec = float("nan")
            seg_rec = float("nan")

        try:
            Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        except Exception as _:
            pass

        # Segmentation PR
        if y_true_all.size and y_score_all.size:
            try:
                seg_pr_path = str(Path(args.plot_dir) / "pr_segmentation_padim.png")
                _plot_pr_curve(
                    y_true=y_true_all.astype(np.uint8).ravel(),
                    y_score=y_score_all.astype(np.float32).ravel(),
                    out_path=seg_pr_path,
                    title="Segmentation PR (hybrid pixels, PaDiM)",
                    op_thr=float(seg_thr),
                    op_kind="score>=seg_thr"
                )
            except Exception as e:
                print(f"[PR][PaDiM] segmentation plot failed: {e}")
        else:
            print("[PR][PaDiM] segmentation: no pixels to plot.")


        if y_true_all.size and y_score_all.size:
            try:
                seg_roc_path = str(Path(args.plot_dir) / "roc_segmentation_padim.png")
                _plot_roc_curve(
                    y_true=y_true_all.astype(np.uint8).ravel(),
                    y_score=y_score_all.astype(np.float32).ravel(),
                    out_path=seg_roc_path,
                    title="Segmentation ROC (hybrid pixels, PaDiM)",
                    op_thr=float(seg_thr),
                    op_kind="score>=seg_thr"
                )
            except Exception as e:
                print(f"[ROC][PaDiM] segmentation plot failed: {e}")
        else:
            print("[ROC][PaDiM] segmentation: no pixels to plot.")


        if det_labels.size and det_scores.size:
            try:
                det_pr_path = str(Path(args.plot_dir) / "pr_detection_padim.png")
                _plot_pr_curve(
                    y_true=det_labels.astype(np.uint8).ravel(),
                    y_score=det_scores.astype(np.float32).ravel(),
                    out_path=det_pr_path,
                    title="Detection PR (patch-level, PaDiM)",
                    op_thr=float(det_thr),
                    op_kind="score>=det_thr"
                )
            except Exception as e:
                print(f"[PR][PaDiM] detection plot failed: {e}")
        else:
            print("[PR][PaDiM] detection: no patch preds to plot.")


        if det_labels.size and det_scores.size:
            try:
                det_roc_path = str(Path(args.plot_dir) / "roc_detection_padim.png")
                _plot_roc_curve(
                    y_true=det_labels.astype(np.uint8).ravel(),
                    y_score=det_scores.astype(np.float32).ravel(),
                    out_path=det_roc_path,
                    title="Detection ROC (patch-level, PaDiM)",
                    op_thr=float(det_thr),
                    op_kind="score>=det_thr"
                )
            except Exception as e:
                print(f"[ROC][PaDiM] detection plot failed: {e}")
        else:
            print("[ROC][PaDiM] detection: no patch preds to plot.")


        if csv_file_path is None:
            csv_file_path = str(Path(img_dir).parent / "padim_patch_labels.csv")
        with open(csv_file_path, "w", newline="") as csv_f:
            csv_w = csv.writer(csv_f)
            csv_w.writerow(["Patch_ID", "Anomaly_Label"])
            for p, s in zip(all_paths, all_scores):
                csv_w.writerow([Path(p).name, int(s >= det_thr)])


        print(
    f"[TEST seg][PaDiM] "
    f"thr={seg_thr:.4f}  "
    f"F1={seg_m['f1']:.4f}  "
    f"P={seg_prec:.4f}  "
    f"R={seg_rec:.4f}  "
    f"AP={seg_m['ap']:.4f}  "
    f"AUROC={seg_m['auc']:.4f}"
)
        print(f"[TEST det][PaDiM] thr={det_thr:.4f}  "
              f"F1={det_m.get('f1', float('nan')):.4f}  AP={det_m.get('ap', float('nan')):.4f}  "
              f"AUROC={det_m.get('auc', float('nan')):.4f}  "
              f"P={det_m.get('precision', float('nan')):.4f}  R={det_m.get('recall', float('nan')):.4f}")
        try:
            Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if y_true_all.size and y_score_all.size:
            try:
                seg_pr_path = str(Path(args.plot_dir) / "pr_segmentation_padim_nf.png")
                _plot_pr_curve(
                    y_true=y_true_all.astype(np.uint8).ravel(),
                    y_score=y_score_all.astype(np.float32).ravel(),
                    out_path=seg_pr_path,
                    title="Segmentation PR (PaDiM-NF)",
                    op_thr=float(seg_thr),
                    op_kind="score>=seg_thr",
                )
            except Exception as e:
                print(f"[PR][PaDiM-NF] segmentation plot failed: {e}")

            try:
                seg_roc_path = str(Path(args.plot_dir) / "roc_segmentation_padim_nf.png")
                _plot_roc_curve(
                    y_true=y_true_all.astype(np.uint8).ravel(),
                    y_score=y_score_all.astype(np.float32).ravel(),
                    out_path=seg_roc_path,
                    title="Segmentation ROC (PaDiM-NF)",
                    op_thr=float(seg_thr),
                    op_kind="score>=seg_thr",
                )
            except Exception as e:
                print(f"[ROC][PaDiM-NF] segmentation plot failed: {e}")
        else:
            print("[PR/ROC][PaDiM-NF] segmentation: no pixels to plot.")


        if det_labels.size and det_scores.size:
            try:
                det_pr_path = str(Path(args.plot_dir) / "pr_detection_padim_nf.png")
                _plot_pr_curve(
                    y_true=det_labels.astype(np.uint8).ravel(),
                    y_score=det_scores.astype(np.float32).ravel(),
                    out_path=det_pr_path,
                    title="Detection PR (PaDiM-NF)",
                    op_thr=float(det_thr),
                    op_kind="score>=det_thr",
                )
            except Exception as e:
                print(f"[PR][PaDiM-NF] detection plot failed: {e}")

            try:
                det_roc_path = str(Path(args.plot_dir) / "roc_detection_padim_nf.png")
                _plot_roc_curve(
                    y_true=det_labels.astype(np.uint8).ravel(),
                    y_score=det_scores.astype(np.float32).ravel(),
                    out_path=det_roc_path,
                    title="Detection ROC (PaDiM-NF)",
                    op_thr=float(det_thr),
                    op_kind="score>=det_thr",
                )
            except Exception as e:
                print(f"[ROC][PaDiM-NF] detection plot failed: {e}")
        else:
            print("[PR/ROC][PaDiM-NF] detection: no patch preds to plot.")
        return {
            "seg_dice_global": seg_m["dice"],
            "seg_f1_global":   seg_m["f1"],
            "seg_ap_global":   seg_m["ap"],
            "seg_auc_global":  seg_m["auc"],
            "seg_precision_global": seg_prec,   
            "seg_recall_global":    seg_rec,    
            "det_f1": det_m.get("f1", float("nan")),
            "det_ap": det_m.get("ap", float("nan")),
            "det_auroc": det_m.get("auc", float("nan")),
            "det_precision": det_m.get("precision", float("nan")),    
            "det_recall":    det_m.get("recall", float("nan")),       
            "images": torch.empty(0),
            "probs": torch.empty(0)
        }
    # DRAEM BRANCH 
    else:
        
        rec = ReconstructiveSubNetwork(
            3,
            3,
        ).to(dev)

        seg = DiscriminativeSubNetwork(
            6,
            2,
        ).to(dev)

        rec, thr_meta = load_checkpoint(
            rec,
            ckpt_path,
            "rec",
        )

        seg, _ = load_checkpoint(
            seg,
            ckpt_path,
            "seg",
        )

        rec.eval()
        seg.eval()

        
        seg_thr = float(
            getattr(
                args,
                "segmentation_threshold",
                0.16,
            )
        )

        det_thr = float(
            getattr(
                args,
                "detection_threshold",
                0.20,
            )
        )

        anomaly_map_thr = 0.80

        ckpt_seg_thr = thr_meta.get(
            "seg"
        )

        ckpt_det_thr = thr_meta.get(
            "det"
        )

        if (
            ckpt_seg_thr is not None
            and not np.isclose(
                float(ckpt_seg_thr),
                seg_thr,
            )
        ):
            print(
                "[WARN] checkpoint segmentation threshold "
                f"({float(ckpt_seg_thr):.4f}) differs from "
                f"main.py ({seg_thr:.4f}); using main.py."
            )

        if (
            ckpt_det_thr is not None
            and not np.isclose(
                float(ckpt_det_thr),
                det_thr,
            )
        ):
            print(
                "[WARN] checkpoint detection threshold "
                f"({float(ckpt_det_thr):.4f}) differs from "
                f"main.py ({det_thr:.4f}); using main.py."
            )

        print(
            "[TEST thresholds] "
            f"seg-metrics={seg_thr:.4f}, "
            f"det={det_thr:.4f}, "
            f"anomaly-map={anomaly_map_thr:.2f}"
        )

        # Data

        tf_img = T.Compose(
            [
                T.Resize(
                    (256, 256)
                ),
                T.ToTensor(),
                T.Normalize(
                    [0.5] * 3,
                    [0.5] * 3,
                ),
            ]
        )

        tf_mask = T.Compose(
            [
                T.Resize(
                    (256, 256),
                    interpolation=IM.NEAREST,
                ),
                T.ToTensor(),
            ]
        )

        dataset = TestDataset(
            img_dir,
            mask_dir,
            tf_img,
            tf_mask,
        )

        labels = [
            1 if is_abnormal else 0
            for is_abnormal
            in dataset.is_abnormal
        ]

        if sum(labels) == 0:
            print(
                "[WARN] No positives in test set; "
                "using plain batching."
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch,
                shuffle=False,
                num_workers=4,
            )

        else:
            batch_sampler = StratifiedBatchSampler(
                labels=labels,
                batch_size=batch,
                pos_per_batch=1,
                shuffle=True,
                seed=42,
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=4,
            )


        if heatmap_dir:
            Path(
                heatmap_dir
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        if anomaly_map_dir:
            Path(
                anomaly_map_dir
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        if binary_dir:
            Path(
                binary_dir
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        Path(
            args.plot_dir
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf = (
            _PdfPages(
                output_pdf_path
            )
            if output_pdf_path
            else None
        )

        if csv_file_path is None:
            csv_file_path = str(
                Path(img_dir).parent
                / "patch_labels.csv"
            )

        imgs_all: list[
            torch.Tensor
        ] = []

        probs_all: list[
            torch.Tensor
        ] = []

        det_labels_parts: list[
            np.ndarray
        ] = []

        det_prob_parts: list[
            np.ndarray
        ] = []

        y_true_parts: list[
            np.ndarray
        ] = []

        y_score_parts: list[
            np.ndarray
        ] = []

        with open(
            csv_file_path,
            "w",
            newline="",
        ) as csv_f:
            csv_w = csv.writer(
                csv_f
            )

            csv_w.writerow(
                [
                    "Patch_ID",
                    "Anomaly_Label",
                ]
            )

            for (
                imgs,
                gts,
                paths,
                lbl,
            ) in tqdm(
                loader,
                desc="test-2-draem",
            ):
                imgs = imgs.to(
                    dev,
                    non_blocking=True,
                )

                gts = gts.to(
                    dev,
                    non_blocking=True,
                )

                rec_out = rec(
                    imgs
                )

                (
                    seg_logits,
                    _,
                    _,
                    tau_logit,
                ) = seg(
                    torch.cat(
                        [
                            rec_out,
                            imgs,
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
                    != gts.shape[2:]
                ):
                    prob_map = F.interpolate(
                        prob_map,
                        size=gts.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                tau_prob = torch.sigmoid(
                    tau_logit
                ).view(-1)

                if (
                    float(
                        getattr(
                            args,
                            "det_loss_w",
                            0.0,
                        )
                    )
                    == 0.0
                ):
                    tau_prob = torch.ones_like(
                        tau_prob
                    )

                det_labels_parts.append(
                    lbl.detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.int64
                    )
                )

                det_prob_parts.append(
                    tau_prob.detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32
                    )
                )

                positive_pixels = (
                    gts > 0.5
                )

                for batch_index in range(
                    gts.size(0)
                ):
                    positive_mask = (
                        positive_pixels[
                            batch_index,
                            0,
                        ]
                    )

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
                            scores.astype(
                                np.float32
                            )
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
                            scores.astype(
                                np.float32
                            )
                        )


                predicted_abnormal = (
                    tau_prob
                    >= det_thr
                )

                csv_w.writerows(
                    (
                        Path(path).name,
                        int(
                            prediction.item()
                        ),
                    )
                    for path, prediction
                    in zip(
                        paths,
                        predicted_abnormal,
                    )
                )


                for image_index in range(
                    imgs.size(0)
                ):
                    name = Path(
                        paths[
                            image_index
                        ]
                    ).stem

                    probability = (
                        prob_map[
                            image_index,
                            0,
                        ]
                        .detach()
                        .float()
                        .cpu()
                    )

                    if heatmap_dir:
                        heat_rgb = (
                            overlay_heatmap_on_image(
                                imgs[
                                    image_index
                                ],
                                probability,
                            )
                        )

                        cv2.imwrite(
                            str(
                                Path(
                                    heatmap_dir
                                )
                                / (
                                    f"{name}"
                                    "_heatmap.png"
                                )
                            ),
                            cv2.cvtColor(
                                heat_rgb,
                                cv2.COLOR_RGB2BGR,
                            ),
                        )

        
                    if anomaly_map_dir:
                        anomaly_rgb = (
                            overlay_anomaly_on_rgb(
                                imgs[
                                    image_index
                                ],
                                probability,
                                thr=anomaly_map_thr,
                            )
                        )

                        cv2.imwrite(
                            str(
                                Path(
                                    anomaly_map_dir
                                )
                                / (
                                    f"{name}"
                                    "_anomap.png"
                                )
                            ),
                            cv2.cvtColor(
                                anomaly_rgb,
                                cv2.COLOR_RGB2BGR,
                            ),
                        )

                    if binary_dir:
                        save_binary_anomaly_map(
                            probability,
                            thr=anomaly_map_thr,
                            save_color_path=(
                                Path(
                                    binary_dir
                                )
                                / (
                                    f"{name}"
                                    "_mask_color.png"
                                )
                            ),
                            fg_rgb=(
                                255,
                                255,
                                0,
                            ),
                            bg_rgb=(
                                10,
                                20,
                                60,
                            ),
                        )

                # Optional compatibility outputs.
                imgs_all.append(
                    imgs.detach().cpu()
                )

                probs_all.append(
                    prob_map
                    .detach()
                    .cpu()
                )

        if (
            y_true_parts
            and y_score_parts
        ):
            y_true_np = (
                np.concatenate(
                    y_true_parts
                )
                .astype(
                    np.uint8
                )
            )

            y_score_np = (
                np.concatenate(
                    y_score_parts
                )
                .astype(
                    np.float32
                )
            )

        else:
            y_true_np = np.empty(
                0,
                dtype=np.uint8,
            )

            y_score_np = np.empty(
                0,
                dtype=np.float32,
            )


        if (
            y_true_np.size
            and y_score_np.size
        ):
            if (
                y_true_np.shape
                != y_score_np.shape
            ):
                raise RuntimeError(
                    "Global segmentation arrays have "
                    "different shapes: "
                    f"{y_true_np.shape} vs "
                    f"{y_score_np.shape}"
                )

            if not np.isfinite(
                y_score_np
            ).all():
                raise ValueError(
                    "Segmentation scores contain NaN/Inf."
                )

            seg_global = (
                compute_global_seg_metrics(
                    y_true_np,
                    y_score_np,
                    seg_thr,
                )
            )

            seg_f1_global = float(
                seg_global["f1"]
            )

            seg_precision_global = float(
                seg_global[
                    "precision"
                ]
            )

            seg_recall_global = float(
                seg_global[
                    "recall"
                ]
            )

            seg_ap_global = float(
                seg_global["ap"]
            )

            seg_auc_global = float(
                seg_global["auc"]
            )

            print(
                "[TEST seg GLOBAL hybrid] "
                f"thr={seg_thr:.4f}  "
                f"F1={seg_f1_global:.4f}  "
                f"P={seg_precision_global:.4f}  "
                f"R={seg_recall_global:.4f}  "
                f"AP={seg_ap_global:.4f}  "
                f"AUROC={seg_auc_global:.4f}"
            )

            seg_pr_path = str(
                Path(
                    args.plot_dir
                )
                / "pr_segmentation.png"
            )

            try:
                _plot_pr_curve(
                    y_true=(
                        y_true_np
                        .astype(
                            np.uint8
                        )
                        .ravel()
                    ),
                    y_score=(
                        y_score_np
                        .astype(
                            np.float32
                        )
                        .ravel()
                    ),
                    out_path=(
                        seg_pr_path
                    ),
                    title=(
                        "Segmentation PR "
                        "(global hybrid pixels)"
                    ),
                    op_thr=seg_thr,
                    op_kind=(
                        "score>seg_thr"
                    ),
                )

            except Exception as exc:
                print(
                    "[PR] segmentation "
                    f"plot failed: {exc}"
                )

  
            seg_roc_path = str(
                Path(
                    args.plot_dir
                )
                / "roc_segmentation.png"
            )

            try:
                _plot_roc_curve(
                    y_true=(
                        y_true_np
                        .astype(
                            np.uint8
                        )
                        .ravel()
                    ),
                    y_score=(
                        y_score_np
                        .astype(
                            np.float32
                        )
                        .ravel()
                    ),
                    out_path=(
                        seg_roc_path
                    ),
                    title=(
                        "Segmentation ROC "
                        "(global hybrid pixels)"
                    ),
                    op_thr=seg_thr,
                    op_kind=(
                        "score>seg_thr"
                    ),
                )

            except Exception as exc:
                print(
                    "[ROC] segmentation "
                    f"plot failed: {exc}"
                )

        else:
            seg_f1_global = float(
                "nan"
            )

            seg_precision_global = float(
                "nan"
            )

            seg_recall_global = float(
                "nan"
            )

            seg_ap_global = float(
                "nan"
            )

            seg_auc_global = float(
                "nan"
            )

            print(
                "[TEST seg] No global hybrid "
                "pixels were collected."
            )


        det_labels_np = (
            np.concatenate(
                det_labels_parts
            )
            .astype(
                np.int64
            )
            if det_labels_parts
            else np.empty(
                0,
                dtype=np.int64,
            )
        )

        det_probs_np = (
            np.concatenate(
                det_prob_parts
            )
            .astype(
                np.float32
            )
            if det_prob_parts
            else np.empty(
                0,
                dtype=np.float32,
            )
        )

        if (
            det_labels_np.size
            and det_probs_np.size
        ):
            if (
                det_labels_np.shape
                != det_probs_np.shape
            ):
                raise RuntimeError(
                    "Detection arrays have "
                    "different shapes: "
                    f"{det_labels_np.shape} vs "
                    f"{det_probs_np.shape}"
                )

            if not np.isfinite(
                det_probs_np
            ).all():
                raise ValueError(
                    "Detection probabilities "
                    "contain NaN/Inf."
                )

            det_m = (
                evaluate_detection_metrics(
                    det_labels_np,
                    det_probs_np,
                    thresh=det_thr,
                )
            )

            print(
                "[TEST det] "
                f"thr={det_thr:.4f}  "
                f"F1={det_m['f1']:.4f}  "
                f"P={det_m['precision']:.4f}  "
                f"R={det_m['recall']:.4f}  "
                f"AP={det_m['ap']:.4f}  "
                f"AUROC={det_m['auc']:.4f}"
            )

        else:
            det_m = {
                "f1": float(
                    "nan"
                ),
                "precision": float(
                    "nan"
                ),
                "recall": float(
                    "nan"
                ),
                "ap": float(
                    "nan"
                ),
                "auc": float(
                    "nan"
                ),
            }

            print(
                "[TEST det] No patch-level "
                "predictions collected."
            )

        if (
            det_labels_np.size
            and det_probs_np.size
        ):
            det_pr_path = str(
                Path(
                    args.plot_dir
                )
                / "pr_detection.png"
            )

            try:
                _plot_pr_curve(
                    y_true=(
                        det_labels_np
                        .astype(
                            np.uint8
                        )
                        .ravel()
                    ),
                    y_score=(
                        det_probs_np
                        .astype(
                            np.float32
                        )
                        .ravel()
                    ),
                    out_path=(
                        det_pr_path
                    ),
                    title=(
                        "Detection PR "
                        "(patch-level)"
                    ),
                    op_thr=det_thr,
                    op_kind=(
                        "score>=det_thr"
                    ),
                )

            except Exception as exc:
                print(
                    "[PR] detection "
                    f"plot failed: {exc}"
                )

            det_roc_path = str(
                Path(
                    args.plot_dir
                )
                / "roc_detection.png"
            )

            try:
                _plot_roc_curve(
                    y_true=(
                        det_labels_np
                        .astype(
                            np.uint8
                        )
                        .ravel()
                    ),
                    y_score=(
                        det_probs_np
                        .astype(
                            np.float32
                        )
                        .ravel()
                    ),
                    out_path=(
                        det_roc_path
                    ),
                    title=(
                        "Detection ROC "
                        "(patch-level)"
                    ),
                    op_thr=det_thr,
                    op_kind=(
                        "score>=det_thr"
                    ),
                )

            except Exception as exc:
                print(
                    "[ROC] detection "
                    f"plot failed: {exc}"
                )

        if pdf is not None:
            fig, ax = _plt.subplots(
                figsize=(
                    7,
                    5,
                )
            )

            ax.text(
                0.5,
                0.68,
                (
                    "Anomaly localization "
                    "(pixel-level, global hybrid)\n"
                    f"  F1        : "
                    f"{seg_f1_global:.4f}\n"
                    f"  Precision : "
                    f"{seg_precision_global:.4f}\n"
                    f"  Recall    : "
                    f"{seg_recall_global:.4f}\n"
                    f"  AP        : "
                    f"{seg_ap_global:.4f}\n"
                    f"  AUROC     : "
                    f"{seg_auc_global:.4f}\n"
                    f"  Metric thr: "
                    f"{seg_thr:.4f}"
                ),
                ha="center",
                va="center",
                fontsize=11,
                family="monospace",
            )

            ax.text(
                0.5,
                0.24,
                (
                    "Anomaly detection "
                    "(patch-level)\n"
                    f"  F1        : "
                    f"{det_m['f1']:.4f}\n"
                    f"  Precision : "
                    f"{det_m['precision']:.4f}\n"
                    f"  Recall    : "
                    f"{det_m['recall']:.4f}\n"
                    f"  AP        : "
                    f"{det_m['ap']:.4f}\n"
                    f"  AUROC     : "
                    f"{det_m['auc']:.4f}\n"
                    f"  Det thr   : "
                    f"{det_thr:.4f}"
                ),
                ha="center",
                va="center",
                fontsize=11,
                family="monospace",
            )

            ax.axis(
                "off"
            )

            fig.tight_layout()

            pdf.savefig(
                fig
            )

            _plt.close(
                fig
            )

            pdf.close()


        return {
            "seg_f1_global": (
                seg_f1_global
            ),
            "seg_precision_global": (
                seg_precision_global
            ),
            "seg_recall_global": (
                seg_recall_global
            ),
            "seg_ap_global": (
                seg_ap_global
            ),
            "seg_auc_global": (
                seg_auc_global
            ),

            "det_f1": det_m.get(
                "f1",
                float("nan"),
            ),
            "det_precision": det_m.get(
                "precision",
                float("nan"),
            ),
            "det_recall": det_m.get(
                "recall",
                float("nan"),
            ),
            "det_ap": det_m.get(
                "ap",
                float("nan"),
            ),
            "det_auroc": det_m.get(
                "auc",
                float("nan"),
            ),

            "images": (
                torch.cat(
                    imgs_all
                )
                if imgs_all
                else torch.empty(
                    0
                )
            ),

            "probs": (
                torch.cat(
                    probs_all
                )
                if probs_all
                else torch.empty(
                    0
                )
            ),
        }