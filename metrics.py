
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, label
from sklearn.metrics import average_precision_score, roc_auc_score
from dataclasses import dataclass

@torch.no_grad()
def minmax_per_image(pmap):
    if isinstance(pmap, torch.Tensor):
        B = pmap.size(0)
        vmin = pmap.view(B, -1).min(dim=1).values.view(B, 1, 1, 1)
        vmax = pmap.view(B, -1).max(dim=1).values.view(B, 1, 1, 1)
        denom = torch.clamp(vmax - vmin, min=1e-8)
        return (pmap - vmin) / denom
    else:  # numpy fallback
        pmin, pmax = np.min(pmap), np.max(pmap)
        denom = max(pmax - pmin, 1e-8)
        return (pmap - pmin) / denom

def extract_core_pixel_mask(true_mask: torch.Tensor, patch_radius=2) -> torch.Tensor:
    core_mask = torch.zeros_like(true_mask)
    for b in range(true_mask.size(0)):
        mask_np = true_mask[b, 0].cpu().numpy().astype(np.uint8)
        labeled, num_features = label(mask_np)
        for i in range(1, num_features + 1):
            blob = (labeled == i).astype(np.uint8)
            if blob.sum() == 0:
                continue
            dist = distance_transform_edt(blob)
            yx = np.unravel_index(np.argmax(dist), dist.shape)
            y, x = yx
            y0, y1 = max(0, y - patch_radius), min(dist.shape[0], y + patch_radius + 1)
            x0, x1 = max(0, x - patch_radius), min(dist.shape[1], x + patch_radius + 1)
            core_mask[b, 0, y0:y1, x0:x1] = 1
    return core_mask

def evaluate_core_pixel_metrics(true_mask, pred_probs, args):

    if pred_probs.shape[2:] != true_mask.shape[2:]:
        pred_probs = F.interpolate(pred_probs, size=true_mask.shape[2:],
                                   mode="bilinear", align_corners=False)

    scope = getattr(args, "eval_scope", "blob")   # "blob" | "core" | "inmask" | "hybrid"
    thr   = float(getattr(args, "segmentation_threshold", 0.5))

    if scope == "core":
        r = getattr(args, "core_radius", 2)
        mask_eval = extract_core_pixel_mask(true_mask, patch_radius=r).bool()
        y_true_t  = (true_mask > 0.5).float()
    elif scope == "inmask":

        mask_eval = (true_mask > 0.5).bool()
        y_true_t  = torch.ones_like(true_mask, dtype=torch.float32)
    elif scope == "hybrid":
 
        B, _, H, W = true_mask.shape
        pos = (true_mask > 0.5)
        has_pos = pos.view(B, -1).sum(dim=1) > 0

        mask_eval = torch.zeros_like(true_mask, dtype=torch.bool)
        y_true_t  = torch.zeros_like(true_mask, dtype=torch.float32)

        for b in range(B):
            if has_pos[b]:
                mask_eval[b] = pos[b]                     
                y_true_t[b][pos[b]] = 1.0               
            else:
                mask_eval[b] = torch.ones((1, H, W), dtype=torch.bool, device=true_mask.device)
                y_true_t[b]  = 0.0                       
    else:  
        mask_eval = torch.ones_like(true_mask, dtype=torch.bool)
        y_true_t  = (true_mask > 0.5).float()

    y_score = pred_probs[mask_eval].detach().cpu().numpy()
    y_true  = y_true_t  [mask_eval].detach().cpu().numpy()
    if y_score.size == 0 or y_true.size == 0:
        return {"dice": 0.0, "f1": 0.0, "ap": 0.0, "auc": 0.0}

    y_pred    = (y_score > thr).astype(np.uint8)
    y_true_bin = (y_true  > 0.5).astype(np.uint8)

    TP = np.sum((y_pred == 1) & (y_true_bin == 1))
    FP = np.sum((y_pred == 1) & (y_true_bin == 0))
    FN = np.sum((y_pred == 0) & (y_true_bin == 1))

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    dice      = 2 * TP / (2 * TP + FP + FN + 1e-6)

    try:
        ap  = average_precision_score(y_true_bin, y_score)
        auc = roc_auc_score(y_true_bin, y_score)
    except ValueError:
        ap, auc = 0.0, 0.0

    return {"dice": float(dice), "f1": float(f1), "ap": float(ap), "auc": float(auc), "precision": float(precision), "recall": float(recall)}

def compute_global_seg_metrics(
    y_true_all,
    y_score_all,
    thr,
):
    y_pred = (
        y_score_all > thr
    ).astype(np.uint8)

    y_true_bin = (
        y_true_all > 0.5
    ).astype(np.uint8)

    tp = np.sum(
        (y_pred == 1)
        & (y_true_bin == 1)
    )

    fp = np.sum(
        (y_pred == 1)
        & (y_true_bin == 0)
    )

    fn = np.sum(
        (y_pred == 0)
        & (y_true_bin == 1)
    )

    precision = (
        tp
        / (tp + fp + 1e-6)
    )

    recall = (
        tp
        / (tp + fn + 1e-6)
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-6
        )
    )

    dice = (
        2 * tp
        / (
            2 * tp
            + fp
            + fn
            + 1e-6
        )
    )

    try:
        ap = average_precision_score(
            y_true_bin,
            y_score_all,
        )

        auc = roc_auc_score(
            y_true_bin,
            y_score_all,
        )
    except ValueError:
        ap = 0.0
        auc = 0.0

    return {
        "dice": float(dice),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "ap": float(ap),
        "auc": float(auc),
    }


def det_score_from_map(raw_map, mode="p95", topk_ratio=0.01, normalize=True):
    if normalize:
        raw_map = minmax_per_image(raw_map)
    flat = raw_map.ravel()
    if mode == "p95":      return float(np.percentile(flat, 95))
    if mode == "max":      return float(flat.max())
    if mode == "topk-mean":
        k = max(1, int(len(flat) * topk_ratio))
        return float(np.mean(np.partition(flat, -k)[-k:]))
    raise ValueError("Unknown mode")

def evaluate_detection_metrics(gt: np.ndarray, prob: np.ndarray, thresh: float | None = 0.5) -> dict:
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score, precision_recall_curve
    gt = gt.astype(int).ravel()
    prob = prob.astype(float).ravel()

    if thresh is None:
        p, r, t = precision_recall_curve(gt, prob) 
        f1_arr = (2 * p * r) / (p + r + 1e-6)
        best_idx = int(np.nanargmax(f1_arr))
        if t.size == 0:
            best_thr = 0.5
        else:
            j = min(max(best_idx - 1, 0), len(t) - 1)
            best_thr = float(t[j])
    else:
        best_thr = float(thresh)

    pred_bin = (prob >= best_thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        gt, pred_bin, average='binary', zero_division=0
    )
    try:
        ap  = average_precision_score(gt, prob)
        auc = roc_auc_score(gt, prob)
    except ValueError:
        ap, auc = 0.0, 0.0

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "ap": float(ap),
        "auc": float(auc),
        "best_thr": best_thr,
    }

@dataclass
class _PostprocCfg:
    min_area: int = 15
    use_nms: bool = True
    iou_thr: float = 0.1

_POSTPROC_CFG = _PostprocCfg()

def configure_postproc_from_args(args) -> None:
    """
    Set global postproc defaults from args once (run start).
    Only affects calls that do not explicitly pass these params.
    """
    _POSTPROC_CFG.min_area = int(getattr(args, "min_area", _POSTPROC_CFG.min_area))
    _POSTPROC_CFG.use_nms  = bool(getattr(args, "use_nms",  _POSTPROC_CFG.use_nms))
    _POSTPROC_CFG.iou_thr  = float(getattr(args, "iou_threshold", _POSTPROC_CFG.iou_thr))
    
def remove_small_components(bin_mask: np.ndarray, min_area: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      kept_mask: HxW uint8 (0/1) after area filtering
      labels   : HxW int labels from CC (0 = background)
      stats    : (num, 5) stats from CC
      comp_ids : 1D array of component ids present after area filtering
    """
    if bin_mask.dtype != np.uint8:
        bin_mask = (bin_mask > 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    kept = np.zeros_like(bin_mask, dtype=np.uint8)
    comp_ids = []
    for i in range(1, num):  
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            kept[labels == i] = 1
            comp_ids.append(i)
    return kept, labels, stats, np.asarray(comp_ids, dtype=np.int32)

def boxes_from_components(stats: np.ndarray, comp_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    From CC stats + a list of kept component ids, build boxes and per-box scores.
    Boxes format: [x1,y1,x2,y2]; score: area.
    """
    boxes, scores = [], []
    for i in comp_ids:
        x, y, w, h, area = stats[i]
        boxes.append([x, y, x + w, y + h])
        scores.append(float(area))
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(boxes, np.float32), np.asarray(scores, np.float32)

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return np.asarray(keep, dtype=int)    
def image_score_with_postproc(
    pmap: np.ndarray,
    seg_thr: float,
    *,
    min_area: int | None = None,
    use_nms: bool | None = None,
    iou_thr: float | None = None,
):
    if min_area is None: min_area = _POSTPROC_CFG.min_area
    if use_nms  is None: use_nms  = _POSTPROC_CFG.use_nms
    if iou_thr  is None: iou_thr  = _POSTPROC_CFG.iou_thr

    seg_thr = float(np.clip(seg_thr, 0.0, 1.0))
    pmap = np.asarray(pmap, np.float32)
    pmap = np.clip(pmap, 0.0, 1.0)

    bin_raw = (pmap >= seg_thr).astype(np.uint8)
    kept_mask, labels, stats, comp_ids = remove_small_components(bin_raw, min_area=min_area)
    if comp_ids.size == 0:
        return 0.0, np.zeros_like(bin_raw, dtype=np.uint8)

    if use_nms:
        boxes, scores = boxes_from_components(stats, comp_ids)
        keep_idx = nms_numpy(boxes, scores, iou_thr=iou_thr)
        kept_comp_ids = comp_ids[keep_idx]
        kept_mask = np.isin(labels, kept_comp_ids).astype(np.uint8)

    vals = pmap[kept_mask > 0]
    if vals.size == 0:
        return 0.0, kept_mask

    k = max(20, int(0.05 * vals.size))
    k = min(k, vals.size)
    topk = np.partition(vals, -k)[-k:]
    im_score = float(np.mean(topk))
    return im_score, kept_mask