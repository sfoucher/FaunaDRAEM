
# path: Padim/padim_adapter.py
# Vanilla PaDiM (Defard et al., ICPRW'21) under the original class name--> ORIGINAL Padim in the paper
# - Wide-ResNet-50-2 backbone
# - L1 (stride-4) + L2 (stride-8) + L3 (stride-16) features, aligned to stride-4
# - Random selection to D=100 dims (deterministic uniform)
# - Full covariance per spatial location; store precision (Σ^{-1})
# - Mahalanobis distance maps; RAW by default (normalize=False)


from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
#  Gaussian blur for anomaly maps 
def gaussian_blur_map(pmap_raw: torch.Tensor, sigma: float = 4.0, ksize: int = 7) -> torch.Tensor:
    """
    Depthwise Gaussian blur for anomaly maps.
    Expects Tensor [B, C, H, W] (typically C=1). Returns same shape/dtype/device.

    - Uses reflect padding to avoid border darkening.
    - If sigma <= 0 or ksize < 3 or even -> returns input (no-op) for invalid params.
    """
    if not isinstance(pmap_raw, torch.Tensor):
        raise TypeError("gaussian_blur_map expects a torch.Tensor of shape [B,C,H,W].")
    if pmap_raw.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(pmap_raw.shape)}")
    if sigma is None or sigma <= 0 or ksize is None or ksize < 3:
        return pmap_raw
    if ksize % 2 == 0: 
        ksize += 1

    B, C, H, W = pmap_raw.shape
    device = pmap_raw.device
    dtype  = pmap_raw.dtype

    # 1D Gaussian kernel
    radius = ksize // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(x * x) / (2.0 * (sigma ** 2)))
    g = g / g.sum()


    kernel_h = g.view(1, 1, 1, ksize)              
    kernel_v = g.view(1, 1, ksize, 1)              
    kernel_h = kernel_h.repeat(C, 1, 1, 1)         
    kernel_v = kernel_v.repeat(C, 1, 1, 1)

    pad = (radius, radius, radius, radius)
    x = F.pad(pmap_raw, pad, mode="reflect")

    out = F.conv2d(x, kernel_h, bias=None, stride=1, padding=0, groups=C)
    out = F.conv2d(out, kernel_v, bias=None, stride=1, padding=0, groups=C)
    return out
# Feature extractor (WRN50-2)
class ResNetFeatures(nn.Module):
    """Return concatenated L1+L2+L3 features aligned to layer1 (stride-4)."""
    def __init__(self, backbone: str = "wide_resnet50_2"):
        super().__init__()
        assert backbone in ("resnet50", "wide_resnet50_2")
        if backbone == "wide_resnet50_2":
            net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
  
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.l1 = net.layer1  
        self.l2 = net.layer2  
        self.l3 = net.layer3  
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     
        f1 = self.l1(x)      
        f2 = self.l2(f1)    
        f3 = self.l3(f2)     
        tgt = f1.shape[-2:]  
        if f2.shape[-2:] != tgt:
            f2 = F.interpolate(f2, size=tgt, mode="bilinear", align_corners=False)
        if f3.shape[-2:] != tgt:
            f3 = F.interpolate(f3, size=tgt, mode="bilinear", align_corners=False)
        return torch.cat([f1, f2, f3], dim=1)  

# Vanilla PaDiM with full covariance per location
class PaDiMModel_padim(nn.Module):
    """
    Defard et al. (ICPRW'21) PaDiM:
      - Wide-ResNet-50-2 backbone
      - L1+L2+L3 features aligned to layer1 (stride-4)
      - Random D-dim selection (default 100)
      - Fit per-location Gaussian with FULL covariance (store Σ^{-1})
      - Predict RAW Mahalanobis distance maps (normalize=False for metrics)
    """
    def __init__(self, backbone: str = "wide_resnet50_2",
                 n_select: int = 550, 
                 img_size: Tuple[int,int] = (256,256),
                 device: str = "cuda"):
        super().__init__()
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone  = backbone
        self.n_select  = int(n_select)
        self.img_size  = tuple(img_size)

        self.feat = ResNetFeatures(backbone).to(self.device)

        # buffers
        self.register_buffer("mean", torch.empty(0), persistent=False)      
        self.register_buffer("prec", torch.empty(0), persistent=False)      
        self.register_buffer("proj_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.feat_size = None  # (Hf, Wf)


    @torch.no_grad()
    def _select_indices(self, C: int) -> torch.Tensor:
        g = torch.Generator(device=self.device); g.manual_seed(0)
        idx = torch.randperm(C, generator=g, device=self.device)[:self.n_select]
        return torch.sort(idx).values  

    @torch.no_grad()
    def fit(self, train_clean_loader) -> None:
        """
        Fit μ(h,w) and Σ^{-1}(h,w) using ONLY normal images.
        """
        sums = None                
        sums_xxt = None            
        n = 0
        proj_idx = None
        Hf = Wf = None

        for x in train_clean_loader:
            if isinstance(x, (list, tuple)): x = x[0]
            x = x.to(self.device, non_blocking=True)
            f_all = self.feat(x)                                  
            B, C, Hf, Wf = f_all.shape
            if proj_idx is None:
                proj_idx = self._select_indices(C)
            f = f_all[:, proj_idx, :, :]                         
            D = f.shape[1]

            if sums is None:
                sums = torch.zeros((D, Hf, Wf), device=self.device)
                sums_xxt = torch.zeros((Hf*Wf, D, D), device=self.device)

            sums += f.sum(dim=0)                                  

      
            fx = f.view(B, D, Hf*Wf).permute(2, 0, 1).contiguous()   
            # (HW, D, D) += ∑_b (x_b^T x_b)
            sums_xxt += torch.bmm(fx.transpose(1, 2), fx)            

            n += B

        # moments
        mean = sums / max(n, 1)                                     
        E_xxt = sums_xxt / max(n, 1)                                
        mu_hw = mean.permute(1,2,0).contiguous().view(Hf*Wf, -1)    
        mu_outer = torch.matmul(mu_hw.unsqueeze(2), mu_hw.unsqueeze(1))  
        cov = E_xxt - mu_outer                                      

        
        eps = 1e-2 
        eye = torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0)
        cov = cov + eps * eye


        L = torch.linalg.cholesky(cov)        
        prec = torch.cholesky_inverse(L)                                   
        self.prec = torch.cholesky_inverse(L)
        # store
        self.mean = mean
        self.prec = prec
        self.proj_idx = proj_idx.detach()
        self.feat_size = (Hf, Wf)

    @torch.no_grad()
    def predict_maps(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Return RAW Mahalanobis maps by default.
        Set normalize=True ONLY for visualization.
        """
        x = x.to(self.device, non_blocking=True)
        f_all = self.feat(x)                          
        f = f_all[:, self.proj_idx, :, :]             
        B, D, Hf, Wf = f.shape

        mu = self.mean.unsqueeze(0)                   
        diff = (f - mu).permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, D) 
        tmp  = torch.einsum('bhd,hde->bhe', diff, self.prec)                
        m2   = (tmp * diff).sum(dim=2)                                      
        dmap = torch.sqrt(torch.clamp(m2, min=0.0)).view(B, 1, Hf, Wf)
        dmap = F.interpolate(dmap, size=self.img_size, mode="bilinear", align_corners=False)

        if not normalize:
            return dmap  

   
        outs = []
        for b in range(B):
            v = dmap[b:b+1]
            vmin = float(v.min()); vmax = float(v.max())
            outs.append((v - vmin)/(vmax - vmin) if vmax > vmin else torch.zeros_like(v))
        return torch.cat(outs, dim=0)

    # Checkpoint 
    def state_dict_padim(self) -> Dict:
        return {
            "mean": self.mean.detach().cpu(),
            "prec": self.prec.detach().cpu(),          
            "proj_idx": self.proj_idx.detach().cpu(),
            "img_size": self.img_size,
            "feat_size": self.feat_size,
            "backbone": self.backbone,
            "n_select": self.n_select,
        }

    def load_state_dict_padim(self, state: Dict) -> None:
        self.mean     = state["mean"].to(self.device)
        self.prec     = state["prec"].to(self.device)
        self.proj_idx = state["proj_idx"].to(self.device, dtype=torch.long)
        self.img_size  = tuple(state["img_size"])
        self.feat_size = tuple(state["feat_size"])
        self.backbone  = state["backbone"]
        self.n_select  = int(state["n_select"])

