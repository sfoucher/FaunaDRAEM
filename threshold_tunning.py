import numpy as np, matplotlib.pyplot as plt, torch, torchvision.transforms as T
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from torch.nn import functional as F
from metrics import minmax_per_image
from sklearn.metrics import precision_recall_curve
from metrics import image_score_with_postproc, evaluate_detection_metrics

def pick_seg_thr_at_fpr(y_true_pixels: np.ndarray,
                        y_score_pixels: np.ndarray,
                        target_fpr: float = 1e-3) -> float:
    """
    Calibrate pixel threshold on validation PROB scores to achieve a target FPR on NORMAL pixels.
    y_true_pixels: 0/1 for each pixel (flattened across images)
    y_score_pixels: PROB∈[0,1] for each pixel (same length)
    target_fpr: e.g., 1e-3 (0.1%) or 5e-3 (0.5%)
    Returns: seg_thr in [0,1]
    """
    if y_true_pixels.size == 0:
        return 0.5
    normal_scores = y_score_pixels[y_true_pixels == 0]
    if normal_scores.size == 0:
        return 0.5
    q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
    return float(np.quantile(normal_scores, q))


def pick_det_thr_percentile(normal_im_scores: np.ndarray,
                            q: float = 0.995) -> float:
    """
    Calibrate detection threshold as a percentile of NORMAL image scores.
    normal_im_scores: detection scores (e.g., postproc-based) for normal images from validation
    q: e.g., 0.995 (99.5th percentile)
    """
    if normal_im_scores.size == 0:
        return 0.5
    return float(np.quantile(normal_im_scores, q))

def collect_logits_and_labels(ckpt_path, img_dir, mask_dir, batch=64, device="cuda"):
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode as IM
    from tqdm import tqdm

    from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork
    from Scripts.evaluate import load_checkpoint
    from test_dataset import TestDataset
    ckpt_path = _resolve_ckpt_path(ckpt_path) 
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    rec = ReconstructiveSubNetwork(3,3).to(dev)
    seg = DiscriminativeSubNetwork(6,2).to(dev)


    rec, _ = load_checkpoint(rec, ckpt_path, "padim_nf")

    seg, _ = load_checkpoint(seg, ckpt_path, "padim_nf")
    rec.eval(); seg.eval()

    tf_img  = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([.5]*3,[.5]*3)])
    tf_mask = T.Compose([T.Resize((256,256), interpolation=IM.NEAREST), T.ToTensor()])

    loader = torch.utils.data.DataLoader(
        TestDataset(img_dir, mask_dir, tf_img, tf_mask),
        batch_size=batch, shuffle=False, num_workers=4
    )

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, _, _, lbl in tqdm(loader, desc="collect τ-logits"):
            imgs = imgs.to(dev)
            rec_out = rec(imgs)

            _, _, _, tau_logit = seg(torch.cat([rec_out, imgs], 1))  
            all_logits.append(tau_logit.squeeze(1).cpu())
            all_labels.append(lbl.squeeze().cpu())

    return torch.cat(all_logits), torch.cat(all_labels)
def _resolve_ckpt_path(ckpt_path: str) -> str:  
    p = Path(ckpt_path)
    if p.is_file():
        return str(p)
    if p.is_dir():

        for pat in ("best_*.pt", "*.pt"):
            cands = sorted(p.glob(pat))
            if cands:
                return str(cands[0])
        raise FileNotFoundError(f"No checkpoint files found under: {ckpt_path}")
    raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

def threshold_sweep_plot(
        ckpt_path, img_dir, mask_dir, png_out,
        step=0.01, device="cuda"):
    ckpt_path = _resolve_ckpt_path(ckpt_path)
    logits, labels = collect_logits_and_labels(
        ckpt_path, img_dir, mask_dir, device=device)

    probs  = torch.sigmoid(logits).numpy()
    labels = labels.numpy().astype(int)

    ths      = np.arange(0.0, 1.0 + step/2, step)
    f1_list  = []
    prec_ls  = []
    rec_ls   = []

    for th in ths:
        pred_bin = (probs >= th).astype(int)
        p,r,f,_  = precision_recall_fscore_support(
                       labels, pred_bin, average='binary', zero_division=0)
        prec_ls.append(p); rec_ls.append(r); f1_list.append(f)

    best_idx   = int(np.argmax(f1_list))
    best_th    = ths[best_idx]
    best_f1    = f1_list[best_idx]

    print(f"Best threshold = {best_th:.2f}  →  F1 = {best_f1:.4f}")

    plt.figure(figsize=(8,6))
    plt.plot(ths, f1_list , label="F1-score")
    plt.plot(ths, prec_ls, label="precision", ls="--")
    plt.plot(ths, rec_ls , label="recall",   ls="--")
    plt.axvline(best_th, color="k", ls=":", label=f"best th={best_th:.2f}")
    plt.xlabel("detection threshold")
    plt.ylabel("score")
    plt.title("Confidence ↔ score sweep (τ-head)")
    plt.grid(True); plt.legend()
    Path(png_out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[info] plot saved to {png_out}")

    return best_th, best_f1, {
        "thresholds": ths,
        "f1": f1_list,
        "precision": prec_ls,
        "recall": rec_ls,
    }
def apply_calibration(x: np.ndarray, calib: dict) -> np.ndarray:
    t = calib.get("type", "identity")
    if t == "pctl":
        lo, hi = float(calib["lo"]), float(calib["hi"])
        y = (x - lo) / max(hi - lo, 1e-12)
        return np.clip(y, 0.0, 1.0)
    if t == "minmax":                     
        return minmax_per_image(x)
    return x
def fit_percentile_calibration(norm_pixels: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> dict:
    lo = float(np.percentile(norm_pixels, lo_q))
    hi = float(np.percentile(norm_pixels, hi_q))
    if hi <= lo:
        hi = lo + 1e-6
    return {"type": "pctl", "lo": lo, "hi": hi, "lo_q": lo_q, "hi_q": hi_q}
def tune_thresholds_on_validation(
    val_dl,
    model,
    device,
    min_area: int = 15,
    use_nms: bool = True,
    iou_thr: float = 0.1,
):
    model.eval()

   
    y_true_all, y_score_all = [], []                                 
    normal_pixels = []                                              

    with torch.no_grad():
        for imgs, gts, paths, lbl in val_dl:
            imgs, gts = imgs.to(device), gts.to(device)
            pmap_raw = model.predict_maps(imgs, normalize=False)
            if pmap_raw.shape[2:] != gts.shape[2:]:
                pmap_raw = F.interpolate(pmap_raw, size=gts.shape[2:], mode="bilinear", align_corners=False)

            B = imgs.size(0)
            for b in range(B):
                raw_np = pmap_raw[b, 0].detach().cpu().numpy()
                m = (gts[b, 0].detach().cpu().numpy() > 0.5)

                y_true_all.append(m.astype(np.uint8).ravel())
                y_score_all.append(raw_np.ravel())

                if (~m).any():
                    normal_pixels.append(raw_np[~m])

    y_true_all  = np.concatenate(y_true_all) if y_true_all else np.empty(0, np.uint8)
    y_score_all = (np.concatenate(y_score_all).astype(np.float32)
                   if y_score_all else np.empty(0, np.float32))
    normal_pixels = (np.concatenate(normal_pixels).astype(np.float32)
                     if normal_pixels else np.empty(0, np.float32))

  
    if normal_pixels.size:
        calib = fit_percentile_calibration(normal_pixels, lo_q=1.0, hi_q=99.0)
        y_score_all = apply_calibration(y_score_all, calib)         
    else:
        calib = {"type": "identity"}

    if y_true_all.size and y_score_all.size:
        seg_thr = float(pick_seg_thr_at_fpr(y_true_all, y_score_all, fpr_target=2e-2)) 
    else:
        seg_thr = 0.5

    det_labels, det_scores = [], []
    with torch.no_grad():
        for imgs, gts, paths, lbl in val_dl:
            imgs, gts = imgs.to(device), gts.to(device)
            pmap_raw = model.predict_maps(imgs, normalize=False)
            if pmap_raw.shape[2:] != gts.shape[2:]:
                pmap_raw = F.interpolate(pmap_raw, size=gts.shape[2:], mode="bilinear", align_corners=False)

            cur_scores = []
            for b in range(pmap_raw.size(0)):
                raw_np  = pmap_raw[b, 0].detach().cpu().numpy()
                prob_np = apply_calibration(raw_np, calib)            
                im_score, _ = image_score_with_postproc(
                    prob_np, seg_thr, min_area=min_area, use_nms=use_nms, iou_thr=iou_thr
                )
                if not np.isfinite(im_score): im_score = 0.0        
                cur_scores.append(float(im_score))

            det_scores.append(np.asarray(cur_scores, dtype=float))
            det_labels.append(lbl.cpu().numpy())

    det_labels = np.concatenate(det_labels).astype(int) if det_labels else np.empty(0, int)
    det_scores = np.concatenate(det_scores).astype(float) if det_scores else np.empty(0, float)

    det_m  = evaluate_detection_metrics(det_labels, det_scores, thresh=None)

    det_thr = float(det_m["best_thr"])
 
    if np.isnan(det_thr) or (np.nanstd(det_scores) < 1e-6):
        norm_scores = det_scores[det_labels == 0]
        if norm_scores.size:
            alt_thr = float(np.percentile(norm_scores, 99.0))  # no clipping
            f_s = evaluate_detection_metrics(det_labels, det_scores, thresh=det_thr)["f1"]
            f_a = evaluate_detection_metrics(det_labels, det_scores, thresh=alt_thr)["f1"]
            if f_a > f_s:
                det_thr = alt_thr


 
    if not np.isfinite(det_thr):
        print(f"[warn] det_thr is not finite: {det_thr}")
    det_thr = float(det_thr)  



    return float(seg_thr), float(det_thr), calib

def seg_mask_from_raw(raw_map: np.ndarray, seg_thr_01: float,
                      min_area=0, do_median=False):
    seg01 = minmax_per_image(raw_map)  
    if do_median:
        seg01 = cv2.medianBlur((seg01*255).astype(np.uint8), 3).astype(np.float32)/255.0
    mask = (seg01 >= seg_thr_01).astype(np.uint8)
    if min_area > 0:
  
        import cv2
        num, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[cc == i] = 1
        mask = keep
    return mask

def best_det_threshold_f1(scores, labels, grid=None):

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if grid is None:

        qs = np.quantile(scores, np.linspace(0, 1, 501))
        grid = np.unique(qs)
    best_f1, best_thr = -1.0, None
    for thr in grid:
        pred = (scores >= thr).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1 = 2*prec*rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

def best_seg_threshold_f1(val_raw_maps, val_gt_masks):
   
    thrs = np.linspace(0, 1, 101) 
    best_f1, best_thr = -1.0, None
    for t in thrs:
        tp=fp=fn=0
        for raw, gt in zip(val_raw_maps, val_gt_masks):
            pred = seg_mask_from_raw(raw, t)
            tp += np.logical_and(pred==1, gt==1).sum()
            fp += np.logical_and(pred==1, gt==0).sum()
            fn += np.logical_and(pred==0, gt==1).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2*prec*rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr, best_f1

def pick_best_seg_threshold(y_true_all, y_score_all, for_metric="f1"):

    p, r, t = precision_recall_curve(y_true_all, y_score_all)  
    if for_metric.lower() in ("dice", "f1"):
        f1_arr = (2*p*r)/(p+r+1e-6)
        best_idx = int(np.nanargmax(f1_arr))
        best_thr = 0.0 if best_idx==0 else (1.0 if best_idx>=len(t) else float(t[best_idx-1]))
    else:
        best_thr = 0.5 
    return best_thr