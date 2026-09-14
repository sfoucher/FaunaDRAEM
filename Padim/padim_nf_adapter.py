
from __future__ import annotations
from typing import Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
def gaussian_blur_map(pmap_raw: torch.Tensor, sigma: float = 4.0, ksize: int = 7) -> torch.Tensor:
    """
    Depthwise Gaussian blur for anomaly maps.
    Expects [B, C, H, W]; returns same shape. Uses reflect padding.
    """
   
    if not isinstance(pmap_raw, torch.Tensor) or pmap_raw.ndim != 4:
        raise ValueError("gaussian_blur_map expects a [B,C,H,W] torch.Tensor")
    if sigma is None or sigma <= 0 or ksize is None or ksize < 3:
        return pmap_raw
    if ksize % 2 == 0:
        ksize += 1

    B, C, H, W = pmap_raw.shape
    device, dtype = pmap_raw.device, pmap_raw.dtype

  
    r = ksize // 2
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    g = g / (g.sum() + 1e-12)


    k_h = g.view(1, 1, 1, ksize).repeat(C, 1, 1, 1)
    k_v = g.view(1, 1, ksize, 1).repeat(C, 1, 1, 1)

  
    out = F.pad(pmap_raw, (r, r, r, r), mode="reflect")


    out = F.conv2d(out, k_h, groups=C)
    out = F.conv2d(out, k_v, groups=C)
    return out

class ResNetFeatures(nn.Module):
    def __init__(self, backbone: str = "wide_resnet50_2"):
        super().__init__()
        assert backbone in ("resnet50", "wide_resnet50_2")
        if backbone == "wide_resnet50_2":
            net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.l1, self.l2, self.l3 = net.layer1, net.layer2, net.layer3
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()
    

    @torch.no_grad()
    def forward(self, x):
        x  = self.stem(x)
        f1 = self.l1(x)                     
        f2 = self.l2(f1)                     
        f3 = self.l3(f2)                     
        tgt = f1.shape[-2:]
        if f2.shape[-2:] != tgt: f2 = F.interpolate(f2, size=tgt, mode="bicubic", align_corners=False)
        if f3.shape[-2:] != tgt: f3 = F.interpolate(f3, size=tgt, mode="bicubic", align_corners=False)
        return torch.cat([f1, f2, f3], dim=1)
# ---------- Multi-headed NF wrapper ----------
class MultiHeadedNF(nn.Module):
    def __init__(self, nf_ctor, n_heads: int, *args, **kwargs):
        super().__init__()
        self.nets = nn.ModuleList([nf_ctor(*args, **kwargs) for _ in range(n_heads)])

    def log_prob(self, x, y=None):
        # RETURN STACKED per-head log-probs: (N, H)
        return torch.stack([nf.log_prob(x, y) for nf in self.nets], dim=1)    
# expects padim/normalizing_flows/maf.py with RealNVP, MAF
from Padim.MAF import RealNVP, MAF

class PaDiMNFAdapter(nn.Module):
    """
    PaDiM + NF head (RAW −log p maps). Same feature extractor & D selection as vanilla PaDiM.
    """
    def __init__(self, backbone="wide_resnet50_2", n_select=225, n_heads=1, nf_type="maf", # old: 100
                 img_size=(256,256), device="cuda", n_blocks=7, hidden_size=130, n_hidden=1):
        super().__init__()
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone  = backbone
        self.n_select  = int(n_select)
        self.n_heads   = int(n_heads)
        self.nf_type   = nf_type
        self.img_size  = tuple(img_size)
        self.device = torch.device(device)
        self.feat      = ResNetFeatures(backbone).to(self.device)
       
        self.register_buffer("proj_mat", torch.empty(0, 0, device=self.device))
        self.feat_size = None
        NF = MAF if nf_type.lower() == "maf" else RealNVP

        self.net = MultiHeadedNF(NF, self.n_heads, n_blocks=n_blocks, 
                                 input_size=self.n_select, hidden_size=hidden_size, n_hidden=n_hidden).to(self.device)

 
        self.register_buffer("calib_mean", torch.tensor(0.0)) 
        self.register_buffer("calib_std",  torch.tensor(1.0)) 
        self.calibrated = False  
                 
    @torch.no_grad()
    def _select_indices(self, C: int) -> torch.Tensor:
        g = torch.Generator(device=self.device); g.manual_seed(0)
        idx = torch.randperm(C, generator=g, device=self.device)[:self.n_select]
        return idx.sort().values
    
    @torch.no_grad()
    def _init_semi_orthogonal(self, C: int, D: int) -> torch.Tensor:
        g = torch.Generator(device=self.device); g.manual_seed(0)

        A = torch.randn(C, D, generator=g, device=self.device)
        Q, _ = torch.linalg.qr(A, mode="reduced")  
        return Q
    
    def _embed_batch_flatten(self, x: torch.Tensor):
        with torch.no_grad():
            f_all = self.feat(x)                
            B, C, Hf, Wf = f_all.shape
            if self.proj_mat.numel() == 0:
                self.proj_mat = self._init_semi_orthogonal(C, self.n_select)   
                self.feat_size = (Hf, Wf)

            E = f_all.permute(0, 2, 3, 1).reshape(B*Hf*Wf, C) @ self.proj_mat
            return E, Hf, Wf

    def fit(self, train_clean_loader, n_epochs=50, lr=1e-3, weight_decay=1e-6, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        self.net.train()
        for ep in range(n_epochs):
            total = 0.0
            for xb in train_clean_loader:
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                xb = xb.to(self.device, non_blocking=True)
                E, _, _ = self._embed_batch_flatten(xb)
                heads_lp = self.net.log_prob(E, None)
                logps    = heads_lp.max(dim=1).values
                loss = -logps.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
            if scheduler:
                scheduler.step()
            print(f"[PaDiM-NF] epoch {ep+1}/{n_epochs} nll={total:.3f}")

        self.net.eval()

        with torch.no_grad():  
            acc_mean = 0.0     
            acc_m2   = 0.0      
            count    = 0        #
            for xb in train_clean_loader: 
                if isinstance(xb, (list, tuple)): xb = xb[0] 
                xb = xb.to(self.device, non_blocking=True)    
                nll = self._raw_nll_map(xb)                   
                nll = F.interpolate(nll, size=self.img_size, mode="bicubic", align_corners=False)  
                nll = gaussian_blur_map(nll, sigma=4.0, ksize=7)  
                v = nll.view(-1).float()                      
                   
                count_new = count + v.numel()                 
                delta = v.mean() - (acc_mean if count else 0.0)  
                acc_mean = (acc_mean*count + v.sum().item()) / max(count_new, 1)  
                acc_m2   = acc_m2 + (v.var(unbiased=False).item() + (delta**2)*count/max(count_new,1)) * v.numel()  
                count    = count_new                          
            var = acc_m2 / max(count, 1)                       
            self.calib_mean = torch.tensor(acc_mean, device=self.device)                 
            self.calib_std  = torch.tensor(max(var, 1e-8)**0.5, device=self.device)     
            self.calibrated = True    
               
    @torch.no_grad()
    def _raw_nll_map(self, x: torch.Tensor) -> torch.Tensor:  
        x = x.to(self.device, non_blocking=True)              
        f_all = self.feat(x)                                  
        B, C, Hf, Wf = f_all.shape                            
        if self.proj_mat.numel() == 0:                        
            self.proj_mat = self._init_semi_orthogonal(C, self.n_select)  
            self.feat_size = (Hf, Wf)                         
        X = f_all.permute(0, 2, 3, 1).reshape(B * Hf * Wf, C) 
        E = X @ self.proj_mat                                 
        heads_lp = self.net.log_prob(E, None)                
        logps    = heads_lp.max(dim=1).values                
        nll      = (-logps).view(B, 1, Hf, Wf)                
        return nll   
               
    @torch.no_grad()
    def predict_maps(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        f_all = self.feat(x)              
        B, C, Hf, Wf = f_all.shape

        if self.proj_mat.numel() == 0:
            self.proj_mat = self._init_semi_orthogonal(C, self.n_select)   #
            self.feat_size = (Hf, Wf)

        X = f_all.permute(0, 2, 3, 1).reshape(B * Hf * Wf, C)
        E = X @ self.proj_mat                                              

        heads_lp = self.net.log_prob(E, None)                              
        logps    = heads_lp.max(dim=1).values                              
        nll      = (-logps).view(B, 1, Hf, Wf)                             

        dmap = F.interpolate(nll, size=self.img_size, mode="bicubic", align_corners=False)  
        dmap = gaussian_blur_map(dmap, sigma=4.0, ksize=7)                                   

        if not normalize:
            return dmap

   
        mu, sd = self.calib_mean, self.calib_std                                              
        z = (dmap - mu) / (sd + 1e-12)                                                        
        from torch.distributions import Normal                                                
                                                               
        p = Normal(0.0, 1.0).cdf(z)                                     
        return p                 
    

    # checkpoint 
    def state_dict_nf(self)->Dict:
        return {
            "nf": self.net.state_dict(),
            "proj_mat": self.proj_mat.detach().cpu(),
            "img_size": self.img_size,
            "feat_size": self.feat_size,
            "backbone": self.backbone,
            "n_select": self.n_select,
            "n_heads":  self.n_heads,
            "nf_type":  self.nf_type,
            "calib_mean": self.calib_mean.detach().cpu().item(), 
            "calib_std":  self.calib_std.detach().cpu().item(),   
            "calibrated": self.calibrated,                        
        }

    def load_state_dict_nf(self, state: Dict)->None:
        self.net.load_state_dict(state["nf"])
        self.proj_mat = state["proj_mat"].to(self.device)
        self.img_size  = tuple(state["img_size"])
        self.feat_size = tuple(state["feat_size"])
        self.backbone  = state["backbone"]
        self.n_select  = int(state["n_select"])
        self.n_heads   = int(state.get("n_heads",1))
        self.nf_type   = state.get("nf_type","realnvp")
        self.calib_mean = torch.tensor(state.get("calib_mean", 0.0), device=self.device)  
        self.calib_std  = torch.tensor(state.get("calib_std", 1.0), device=self.device)   
        self.calibrated = bool(state.get("calibrated", False))                             
        self.net = self.net.to(self.device); self.eval()
