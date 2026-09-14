import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support
@torch.no_grad()
def gaussian_peak_heatmap(
    pmap: torch.Tensor,        
    q: float = 0.99,
    peak_k: int = 11,
    sigma_px: float = 2.5,
    max_peaks: int = 300,
) -> torch.Tensor:
    B, _, H, W = pmap.shape
    device = pmap.device

    x_s = F.avg_pool2d(pmap, kernel_size=3, stride=1, padding=1)
    thr = torch.quantile(x_s.flatten(1), q, dim=1).view(B, 1, 1, 1)
    x_j = x_s + 1e-6 * torch.rand_like(x_s)

    mx = F.max_pool2d(x_j, kernel_size=peak_k, stride=1, padding=peak_k//2)
    peaks = (x_j == mx) & (x_j > thr)   # (B,1,H,W)

    k = int(max(3, round(6 * sigma_px))) | 1
    r = k // 2
    yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij"
    )
    g = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma_px**2))
    g = g / (g.max() + 1e-12)
    g = g.view(1, 1, k, k)

    out = torch.zeros_like(pmap)

    for b in range(B):
        coords = torch.nonzero(peaks[b, 0], as_tuple=False)  # (N,2)
        if coords.numel() == 0:
            continue
        if coords.size(0) > max_peaks:
            vals = x_s[b, 0, coords[:, 0], coords[:, 1]]
            topk = torch.topk(vals, k=max_peaks, largest=True).indices
            coords = coords[topk]

        amps = pmap[b, 0, coords[:, 0], coords[:, 1]].clamp(0.0, 1.0)

        for (y, x0), a in zip(coords, amps):
            y0 = int(y.item()); x1 = int(x0.item())
            y_min = max(0, y0 - r); y_max = min(H, y0 + r + 1)
            x_min = max(0, x1 - r); x_max = min(W, x1 + r + 1)

            gy_min = y_min - (y0 - r); gy_max = gy_min + (y_max - y_min)
            gx_min = x_min - (x1 - r); gx_max = gx_min + (x_max - x_min)

            patch = a * g[:, :, gy_min:gy_max, gx_min:gx_max]

            out[b:b+1, :, y_min:y_max, x_min:x_max] = torch.maximum(
                out[b:b+1, :, y_min:y_max, x_min:x_max],
                patch
            )

    return out.clamp(0.0, 1.0)
def overlay_heatmap_on_image(
        image: torch.Tensor | np.ndarray,
        mask : torch.Tensor | np.ndarray,      
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
        gamma : float = 2.0, 
        do_autocontrast: bool = False         
) -> np.ndarray:


    if torch.is_tensor(image):
        img = image.permute(1, 2, 0).cpu().numpy()
    else:
        img = image.copy()

    if img.dtype != np.uint8:
        img = ((img - img.min()) / max(img.ptp(), 1e-8) * 255).astype(np.uint8)

    if torch.is_tensor(mask):
        m = mask.squeeze().cpu().numpy()
    else:
        m = mask.squeeze()

    m = np.clip(m, 0, 1)        
    if gamma != 1.0:
        m = m ** gamma

    if do_autocontrast:           
        m -= m.min()
        if m.max() > 0:
            m /= m.max()

    m_u8 = (m * 255).astype(np.uint8)

    heat = cv2.applyColorMap(m_u8, colormap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    return blended

import cv2
import numpy as np
import torch
def mask_exists(image_filename: str, mask_dir: str) -> bool:
    """Return True when the corresponding mask file exists."""
    image_filename = image_filename.split(":")[0]

    if "Zone.Identifier" in image_filename:
        return False

    if not image_filename.lower().endswith(".jpg"):
        return False

    mask_filename = (
        os.path.splitext(image_filename)[0]
        + ".jpg"
    )

    mask_path = os.path.join(
        mask_dir,
        mask_filename,
    )

    return os.path.exists(mask_path)

def overlay_anomaly_on_rgb(
        image,
        prob_map,
        color     = (255, 0, 0),   # RGB red
        alpha     = 0.6,
        thr       = 0.8,           # segmentation threshold
):
    """
    Paint only anomalous pixels on top of `image` with `color`.

    Args
    ----
    image     : (3,H,W) torch  **or** (H,W,3) numpy
    prob_map  : (1,H,W) or (H,W)     – values 0…1
    color     : RGB tuple
    alpha     : overlay opacity
    thr       : probability threshold for foreground

    Returns
    -------
    blended   : (H,W,3)  uint8  RGB
    """
    if torch.is_tensor(image):
        img = image.permute(1, 2, 0).detach().cpu().numpy()   # (H,W,3)
    else:
        img = image.copy()

    if img.dtype != np.uint8:
        img = ((img - img.min()) / max(img.ptp(), 1e-8) * 255).astype(np.uint8)

   
    if torch.is_tensor(prob_map):
        m = prob_map.squeeze().detach().cpu().numpy()
    else:
        m = prob_map.squeeze()



    mask_bin = (m > thr).astype(np.uint8) * 255              


    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[:] = color                                        
    overlay = cv2.bitwise_and(overlay, overlay, mask=mask_bin)

    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended
import os
import numpy as np, cv2, torch
from pathlib import Path

def save_binary_anomaly_map(
    prob_map,                             
    thr: float,                           
    save_path: str | Path | None = None,        
    save_color_path: str | Path | None = None,  
    fg_rgb: tuple[int,int,int] = (255, 255, 0), 
    bg_rgb: tuple[int,int,int] = (10, 20, 60),  
    return_arrays: bool = False                  
):
    """
    Converts prob_map → binary mask using (prob > thr).
    - Saves 1-channel binary {0,255} mask to `save_path` (if provided).
    - Optionally saves a 3-channel colored mask (yellow FG, dark-blue BG) to `save_color_path`.
    """
   
    if torch.is_tensor(prob_map):
        m = prob_map.detach().float().cpu().numpy()
    else:
        m = np.asarray(prob_map, dtype=np.float32)
    m = np.squeeze(m)
    m = np.clip(m, 0.0, 1.0)

   
    mask01 = (m > float(thr)).astype(np.uint8)
    bin_u8 = (mask01 * 255).astype(np.uint8)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), bin_u8)


    color_rgb = None
    if save_color_path is not None:
        h, w = mask01.shape
        color_rgb = np.empty((h, w, 3), dtype=np.uint8)
        color_rgb[:] = bg_rgb
        color_rgb[mask01 == 1] = fg_rgb
        Path(save_color_path).parent.mkdir(parents=True, exist_ok=True)
        # cv2 expects BGR on write
        cv2.imwrite(str(save_color_path), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))

    if return_arrays:
        return bin_u8, color_rgb


def plot_seg_threshold_sweep(y_true_all, y_score_all, out_png):
    out_dir = Path(out_png).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    p, r, t = precision_recall_curve(y_true_all, y_score_all) 
    f1 = (2 * p * r) / (p + r + 1e-6)

    best_idx = int(np.nanargmax(f1))
    if best_idx <= 0:
        best_thr = 0.0
    elif best_idx >= len(t)+1:
        best_thr = 1.0
    else:
        best_thr = float(t[best_idx-1])


    if len(t) > 0:
        x = t
        y_f1 = f1[1:]  
        y_p  = p[1:]
        y_r  = r[1:]
    else:

        x = np.array([0.0])
        y_f1 = np.array([f1[-1]])
        y_p  = np.array([p[-1]])
        y_r  = np.array([r[-1]])

    plt.figure(figsize=(6, 4))
    plt.plot(x, y_f1, label="F1-score")
    plt.plot(x, y_p,  "--", label="precision")
    plt.plot(x, y_r,  "--", label="recall")
    plt.axvline(best_thr, linestyle=":", label=f"best th={best_thr:.3f}")
    plt.xlabel("segmentation threshold")
    plt.ylabel("score")
    plt.title("Threshold sweep (segmentation)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

    return best_thr

def plot_metric_over_epochs(metric_values, metric_name, filename, epochs_x=None):
    """
    Plot metric_values against epochs_x (list of epoch numbers).
    If epochs_x is None, use 1..len(values).
    """
    import matplotlib.pyplot as plt
    import os

    if epochs_x is None:
        epochs_x = list(range(1, len(metric_values) + 1))
    assert len(epochs_x) == len(metric_values), \
        f"{metric_name}: epochs_x and metric_values must have same length"

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_x, metric_values, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Over Epochs")
    plt.grid(True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"{metric_name} plot saved as {filename}")


def plot_all_training_metrics(
        losses_tr, seg_dice, seg_auc, seg_f1, seg_ap,
        det_bce, det_f1, det_auc, det_ap,
        out_dir: str,
        seg_epochs=None, det_epochs=None
):
    os.makedirs(out_dir, exist_ok=True)


    plot_metric_over_epochs(losses_tr, "train loss",
                            f"{out_dir}/seg_train_loss.png")

    plot_metric_over_epochs(det_bce, "τ-head BCE",
                            f"{out_dir}/det_train_bce.png", epochs_x=det_epochs)
    plot_metric_over_epochs(det_f1, "det f1",
                            f"{out_dir}/det_f1.png", epochs_x=det_epochs)
    plot_metric_over_epochs(det_auc, "det auroc",
                            f"{out_dir}/det_auroc.png", epochs_x=det_epochs)
    plot_metric_over_epochs(det_ap,  "det AP",
                            f"{out_dir}/det_ap.png", epochs_x=det_epochs)

    plot_metric_over_epochs(seg_dice, "seg dice",
                            f"{out_dir}/seg_dice.png", epochs_x=seg_epochs)
    plot_metric_over_epochs(seg_auc,  "seg auroc",
                            f"{out_dir}/seg_auroc.png", epochs_x=seg_epochs)
    plot_metric_over_epochs(seg_f1,   "seg f1",
                            f"{out_dir}/seg_f1.png", epochs_x=seg_epochs)
    plot_metric_over_epochs(seg_ap,   "seg AP",
                            f"{out_dir}/seg_ap.png", epochs_x=seg_epochs)

    print(f"[info] all training plots saved to → {out_dir}")

def _plot_pr_curve(y_true, y_score, out_path, title, op_thr=None, op_kind="score>=thr"):
    """
    CPU-only PR plotter. Also saves precision–recall curve data to CSV next to the plot.
    """
    # Coerce to NumPy on CPU
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)
    if hasattr(y_score, "detach"):
        y_score = y_score.detach().cpu().numpy()
    else:
        y_score = np.asarray(y_score)

    y_true  = y_true.astype(np.uint8).ravel()
    y_score = y_score.astype(np.float32).ravel()

    if y_true.size == 0 or y_score.size == 0:
        print(f"[PR] skip plot for {title}: empty inputs.")
        return


    p, r, thr = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

  
    csv_path = Path(out_path).with_suffix(".csv")
    thr = np.append(thr, np.nan)  # pad to match p/r lengths
    pr_data = np.column_stack([thr, p, r])
    np.savetxt(csv_path, pr_data, delimiter=",", header="threshold,precision,recall", comments="")
    print(f"[PR→CSV] saved → {csv_path}  | AP={ap:.4f}  | points={len(p)}")

  
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r, p, lw=2, label=f"PR (AP={ap:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.grid(True)

    if op_thr is not None:
        pred = (y_score >= float(op_thr)).astype(np.uint8)
        prec, rec, _, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        ax.scatter([rec], [prec], s=50, marker="o",
                   label=f"Op pt ({op_kind})\nP={prec:.3f}, R={rec:.3f}")

    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[PR] saved → {out_path}")

def _plot_roc_curve(y_true, y_score, out_path, title, op_thr=None, op_kind="score>=thr"):
    """
    CPU-only ROC plotter. Also saves (fpr, tpr, threshold) values to a CSV file.
    """

    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)
    if hasattr(y_score, "detach"):
        y_score = y_score.detach().cpu().numpy()
    else:
        y_score = np.asarray(y_score)

    y_true  = y_true.astype(np.uint8).ravel()
    y_score = y_score.astype(np.float32).ravel()

    if y_true.size == 0 or y_score.size == 0:
        print(f"[ROC] skip plot for {title}: empty inputs.")
        return


    pos = np.any(y_true == 1)
    neg = np.any(y_true == 0)
    if not (pos and neg):
        print(f"[ROC] skip plot for {title}: need both classes (got pos={pos}, neg={neg}).")
        return

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, thr = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
 
    csv_path = Path(out_path).with_suffix(".csv")
    roc_data = np.column_stack([fpr, tpr, thr])
    np.savetxt(csv_path, roc_data, delimiter=",", header="FPR,TPR,threshold", comments="")
    print(f"[ROC→CSV] saved → {csv_path}  | AUC={roc_auc:.4f}  | points={len(fpr)}")

    # Plot 
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], lw=1, ls="--")  # chance line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True)

    if op_thr is not None:
        thr_val = float(op_thr)
        pred = (y_score >= thr_val).astype(np.uint8)
        tp = np.logical_and(pred == 1, y_true == 1).sum()
        fp = np.logical_and(pred == 1, y_true == 0).sum()
        fn = np.logical_and(pred == 0, y_true == 1).sum()
        tn = np.logical_and(pred == 0, y_true == 0).sum()
        tpr_op = tp / max(tp + fn, 1)
        fpr_op = fp / max(fp + tn, 1)
        ax.scatter([fpr_op], [tpr_op], s=50, marker="o",
                   label=f"Op pt ({op_kind})\nFPR={fpr_op:.3f}, TPR={tpr_op:.3f}")

        op_csv_path = Path(out_path).with_name(Path(out_path).stem + "_op_point.csv")
        np.savetxt(op_csv_path,
                   np.array([[thr_val, fpr_op, tpr_op]]),
                   delimiter=",",
                   header="threshold,FPR,TPR", comments="")
        print(f"[ROC→CSV] op point → {op_csv_path}")

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ROC] saved → {out_path}")

# SAMPLER 
from torch.utils.data import Sampler
import random
class BalancedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.clean_indices = [i for i, (_, _, _, mask_exists) in enumerate(self.dataset) if mask_exists == 0.0]
        self.anomalous_indices = [i for i, (_, _, _, mask_exists) in enumerate(self.dataset) if mask_exists == 1.0]
        
        min_size = min(len(self.clean_indices), len(self.anomalous_indices))
        self.clean_indices = self.clean_indices[:min_size]
        self.anomalous_indices = self.anomalous_indices[:min_size]

        self.indices = self.clean_indices + self.anomalous_indices
        
    def __iter__(self):

        random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

