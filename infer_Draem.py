import argparse, os
from pathlib import Path
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# Dataset 
class PatchFolder(Dataset):
    def __init__(self, root, resize=(256, 256)):
        self.paths = sorted([
            str(p) for p in Path(root).glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ])
        if not self.paths:
            raise FileNotFoundError(f"No images found under: {root}")

        self.resize = resize
        self.tf = T.Compose([
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize([.5, .5, .5], [.5, .5, .5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        orig_hw = torch.tensor([img.height, img.width], dtype=torch.int32)
        x = self.tf(img)
        return x, path, orig_hw



# Checkpoint loading 
def _strip_module(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}

def load_draem_from_ckpt(ckpt_path: str, device: torch.device):
    """
    Expects a dict with keys {'rec','seg','thr'} as saved in your training code.
    Returns: rec, seg (frozen, eval), thr_meta(dict)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or ("rec" not in ckpt or "seg" not in ckpt):
        raise RuntimeError(
            f"Checkpoint {ckpt_path} does not have expected keys {{'rec','seg',...}}."
        )

    rec = ReconstructiveSubNetwork(3, 3).to(device)
    seg = DiscriminativeSubNetwork(6, 2).to(device)

    rec_sd = _strip_module(ckpt["rec"])
    seg_sd = _strip_module(ckpt["seg"])
    rec.load_state_dict(rec_sd, strict=True)
    seg.load_state_dict(seg_sd, strict=True)

    # Freeze
    for p in rec.parameters(): p.requires_grad = False
    for p in seg.parameters(): p.requires_grad = False
    rec.eval(); seg.eval()

    thr_meta = ckpt.get("thr", {})  
    return rec, seg, thr_meta

def save_heatmap_overlay(prob_2d: np.ndarray, img_path: str, out_png: Path,
                         vmax: float = 1.0, alpha: float = 0.35):
    """
    Saves an overlay: original RGB image + colored heatmap with transparency.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # read original image (BGR if cv2)
    if _HAS_CV2:
        base = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR uint8
        H, W = prob_2d.shape
        base = cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA)

        hm8 = np.clip((prob_2d / max(vmax, 1e-6)) * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)  # BGR

        overlay = cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0.0)
        cv2.imwrite(str(out_png), overlay)
    else:
        
        base = Image.open(img_path).convert("RGB")
        H, W = prob_2d.shape
        base = base.resize((W, H), resample=Image.BILINEAR)
        hm8 = np.clip((prob_2d / max(vmax, 1e-6)) * 255.0, 0, 255).astype(np.uint8)
        hm = Image.fromarray(hm8, mode="L").convert("RGB")
        overlay = Image.blend(base, hm, alpha=alpha)
        overlay.save(str(out_png))


#  Utilities 
def save_heatmap_numpy_png(prob_2d: np.ndarray, out_png: Path, out_npy: Path):
    """
    prob_2d: float32 in [0,1], shape [H,W].
    Saves a colored PNG (+ raw .npy).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(out_npy), prob_2d.astype(np.float32))

    vmax = 1.0
    hm8 = np.clip(np.round((prob_2d / vmax) * 255.0), 0, 255).astype(np.uint8)

    if _HAS_CV2:
        color = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)  # BGR
        cv2.imwrite(str(out_png), color)
    else:

        Image.fromarray(hm8, mode="L").save(str(out_png))


def save_binary_mask(prob_2d: np.ndarray, thr: float, out_mask_png: Path):
    """
    Saves a binary (0/255) mask at threshold thr.
    """
    out_mask_png.parent.mkdir(parents=True, exist_ok=True)
    m = (prob_2d >= float(thr)).astype(np.uint8) * 255
    if _HAS_CV2:
        cv2.imwrite(str(out_mask_png), m)
    else:
        Image.fromarray(m, mode="L").save(str(out_mask_png))


#  Inference core 
@torch.no_grad()
def run_inference(
    ckpt_path: str,
    patches_dir: str,
    out_dir: str,
    device: str = "cuda",
    batch_size: int = 16,
    use_amp: bool = True,
    resize_back_to_original: bool = False,
    also_save_binary: bool = True,
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    rec, seg, thr_meta = load_draem_from_ckpt(ckpt_path, dev)

    seg_thr = 0.8 if also_save_binary else None

    if seg_thr is not None:
        print(f"[info] Using saved validation seg-threshold: {seg_thr:.4f}")
    else:
        print("[info] No saved seg-threshold found; binary masks will be skipped.")

    ds = PatchFolder(patches_dir, resize=(256, 256))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    out_dir = Path(out_dir)
    heat_dir = out_dir / "heatmaps"
    npy_dir  = out_dir / "heatmaps_npy"
    bin_dir  = out_dir / "binary" if seg_thr is not None else None

    autocast_ctx = (
    torch.amp.autocast(device_type="cuda", enabled=(use_amp and dev.type == "cuda"))
    if torch.cuda.is_available() else contextlib.nullcontext()
)

    total = 0
    for batch in dl:
        imgs, paths, orig_hw = batch
        imgs = imgs.to(dev, non_blocking=True)

        with autocast_ctx:
            rec_out = rec(imgs)
            seg_logits, _, _, _ = seg(torch.cat([rec_out, imgs], dim=1))  # [B,2,H,W]
            probs = torch.softmax(seg_logits, dim=1)[:, 1:2]              

        for i in range(probs.size(0)):
            name = Path(paths[i]).stem
            p = probs[i, 0].detach().float().cpu().numpy() 

            if resize_back_to_original:
                H = int(orig_hw[i, 0].item())  
                W = int(orig_hw[i, 1].item())
                if _HAS_CV2:
                    p = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)  
                else:
                    p = np.array(Image.fromarray(p).resize((W, H), resample=Image.BILINEAR))

            print(name, "min/max/mean:", float(p.min()), float(p.max()), float(p.mean()))
            print(os.name, "fraction>0.7:", float((p > 0.7).mean()))

            save_heatmap_numpy_png(
                prob_2d=p,
                out_png=heat_dir / f"{name}_heat.png",
                out_npy=npy_dir  / f"{name}_heat.npy",
            )

            save_heatmap_overlay(
                prob_2d=p,
                img_path=paths[i],
                out_png=heat_dir / f"{name}_overlay.png",
                vmax=0.1,
                alpha=0.45,
            )
            if seg_thr is not None and also_save_binary:
                save_binary_mask(p, seg_thr, bin_dir / f"{name}_mask.png")

            total += 1

    print(f"[done] Wrote heatmaps for {total} patches to: {out_dir}")



def main():
    ap = argparse.ArgumentParser("Frozen DRAEM inference (patches → heatmaps)")
    ap.add_argument("--ckpt", required=True, help="Path to best checkpoint .pt")
    ap.add_argument("--patches", required=True, help="Folder with patch images")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--bs", type=int, default=16, help="Batch size")
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP on CUDA")
    ap.add_argument("--keep-size", action="store_true",
                    help="Resize heatmaps back to original patch size before saving")
    ap.add_argument("--no-binary", action="store_true",
                    help="Do not save binary masks even if a threshold is present in ckpt")
    args = ap.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        patches_dir=args.patches,
        out_dir=args.out,
        device=args.device,
        batch_size=args.bs,
        use_amp=(not args.no_amp),
        resize_back_to_original=args.keep_size,
        also_save_binary=(not args.no_binary),
    )

if __name__ == "__main__":
    main()
