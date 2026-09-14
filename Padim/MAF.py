from __future__ import annotations
from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions import transforms as T
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation 

def mlp(in_f, hid_f, n_hidden, out_f, activation="relu"):
    act = nn.ReLU if activation == "relu" else nn.LeakyReLU
    layers = [nn.Linear(in_f, hid_f), act(inplace=True)]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hid_f, hid_f), act(inplace=True)]
    layers += [nn.Linear(hid_f, out_f)]
    return nn.Sequential(*layers)

# RealNVP (affine coupling)

class _CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_size, n_hidden, mask: torch.Tensor, s_scale: float = 3.0, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.dim = dim
        self.register_buffer("mask", mask.view(1, -1).to(self.device))  
        in_f = dim
        self.s_net = mlp(in_f, hidden_size, n_hidden, dim).to(self.device)  
        self.t_net = mlp(in_f, hidden_size, n_hidden, dim).to(self.device)  
        self.s_scale = float(s_scale)


        nn.init.zeros_(self.s_net[-1].weight); nn.init.zeros_(self.s_net[-1].bias)
        nn.init.zeros_(self.t_net[-1].weight); nn.init.zeros_(self.t_net[-1].bias)

    def forward(self, x: torch.Tensor, inverse: bool = False):
   
        self.mask = self.mask.to(x.device)  

        xa = x * self.mask
        xb = x * (1.0 - self.mask)

        s = self.s_net(xa)
        t = self.t_net(xa)


        s = torch.tanh(s) * self.s_scale

        if not inverse:
            yb = xb * torch.exp(s) + t
            y  = xa + yb
            logdet = (s * (1.0 - self.mask)).sum(dim=1)
        else:
            yb = (xb - t) * torch.exp(-s)
            y  = xa + yb
            logdet = (-s * (1.0 - self.mask)).sum(dim=1)

        # guard rails
        y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        logdet = torch.nan_to_num(logdet, nan=0.0, posinf=0.0, neginf=0.0)
        return y, logdet



    
class RealNVP(nn.Module):
    def __init__(self, n_blocks: int, input_size: int, hidden_size: int, n_hidden: int,
                 activation: str = "relu", **_):
        super().__init__()
        self.D = int(input_size)
        masks = []
        m = torch.zeros(self.D); m[::2] = 1.0
        for i in range(n_blocks):
            masks.append(m if i % 2 == 0 else 1.0 - m)
        self.blocks = nn.ModuleList([_CouplingLayer(self.D, hidden_size, n_hidden, mask) for mask in masks])

     
        self.register_buffer("base_loc",   torch.zeros(self.D))
        self.register_buffer("base_scale", torch.ones(self.D))

    def _flow(self, x: torch.Tensor, inverse: bool = False):
        logdet_total = torch.zeros(x.size(0), device=x.device)
        y = x
        if not inverse:
            for blk in self.blocks:
                y, logdet = blk(y, inverse=False)
                logdet_total = logdet_total + logdet
        else:
            for blk in reversed(self.blocks):
                y, logdet = blk(y, inverse=True)
                logdet_total = logdet_total + logdet
        return y, logdet_total

    def forward(self, x: torch.Tensor):
        z, logdet = self._flow(x, inverse=False)
        return z, logdet

    def log_prob(self, x: torch.Tensor, y=None) -> torch.Tensor:
        z, logdet = self.forward(x)
        

        base = torch.distributions.Independent(
            torch.distributions.Normal(self.base_loc.to(x.device), self.base_scale.to(x.device)), 1
        )
        
        return base.log_prob(z) + logdet
        


# MAF via torch.distributions transforms

class _TCompose(nn.Module):
    def __init__(self, transforms: Sequence[T.Transform]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor):
        logdet = torch.zeros(x.size(0), device=x.device)              # <<<
        y = x
        for tr in self.transforms:
            y_new = tr(y)
            # log|det J| for this transform
            logdet = logdet + tr.log_abs_det_jacobian(y, y_new).sum(dim=1)  # <<<
            y = y_new
        return y, logdet
_ACT = {
    "relu": F.relu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,   
    "tanh": torch.tanh,           
    "gelu": F.gelu,
}
class MAF(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
                 activation="relu", input_order="sequential", batch_norm=True, **_):
        super().__init__()
        self.D = int(input_size)


        act_fn = F.relu 
        transforms = []
        for _ in range(n_blocks):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.D,
                    hidden_features=hidden_size,
                    # num_blocks=n_hidden,
                    num_blocks= 1, 
                    use_residual_blocks=False, 
                    activation=act_fn,         
                    dropout_probability=0.0,
                )
            )
            transforms.append(ReversePermutation(features=self.D))

        self._seq = CompositeTransform(transforms)

        self.register_buffer("base_loc",   torch.zeros(self.D))
        self.register_buffer("base_scale", torch.ones(self.D))

    def forward(self, x: torch.Tensor):

        if next(self._seq.parameters(), None) is not None:
            self._seq.to(x.device)
        z, logabsdet = self._seq(x)
        return z, logabsdet

    def log_prob(self, x: torch.Tensor, y=None):
        z, logabsdet = self.forward(x)
        base = torch.distributions.Independent(
            torch.distributions.Normal(self.base_loc.to(z.device),
                                       self.base_scale.to(z.device)),
            reinterpreted_batch_ndims=1,
        )
        return base.log_prob(z) + logabsdet