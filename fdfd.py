import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FDFD(nn.Module):


    def __init__(
            self,
            channels: int,
            sigma_init: float = 1.0,
            sigma_min: float = 1e-4,
            sigma_max: float = 50.0,
            per_modality: bool = True,
            use_high: bool = False,
            cache_mask: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.C = channels
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.per_modality = bool(per_modality)
        self.use_high = bool(use_high)
        self.cache_mask = bool(cache_mask)

        self.feat_1 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        self.feat_2 = nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False)


        sigma_init = float(sigma_init)
        sigma_raw_init = math.log(math.expm1(max(sigma_init - self.sigma_min, 1e-6)))

        if self.per_modality:

            self.sigma_raw_rgb = nn.Parameter(
                torch.full((self.C,), sigma_raw_init, device=device, dtype=dtype)
            )
            self.sigma_raw_ir = nn.Parameter(
                torch.full((self.C,), sigma_raw_init, device=device, dtype=dtype)
            )
        else:
            # 共用一组 σ_c
            self.sigma_raw = nn.Parameter(
                torch.full((self.C,), sigma_raw_init, device=device, dtype=dtype)
            )

        self._mask_cache = {}

    @staticmethod
    def _fftshift2(x: torch.Tensor) -> torch.Tensor:

        H, W = x.shape[-2], x.shape[-1]
        return torch.roll(x, shifts=(H // 2, W // 2), dims=(-2, -1))

    @staticmethod
    def _make_freq_grid(H: int, W: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:

        u = torch.arange(H, device=device, dtype=dtype) - (H / 2.0)
        v = torch.arange(W, device=device, dtype=dtype) - (W / 2.0)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        return uu, vv

    def _sigma(self, modality: str) -> torch.Tensor:

        if self.per_modality:
            raw = self.sigma_raw_rgb if modality == "rgb" else self.sigma_raw_ir
        else:
            raw = self.sigma_raw

        sigma = F.softplus(raw) + self.sigma_min
        sigma = torch.clamp(sigma, max=self.sigma_max)
        return sigma  # [C]

    def _gaussian_mask(self, H: int, W: int, sigma: torch.Tensor, modality: str) -> torch.Tensor:

        key = (H, W, sigma.device, sigma.dtype, modality)

        if self.cache_mask and key in self._mask_cache:
            return self._mask_cache[key]

        uu, vv = self._make_freq_grid(H, W, device=sigma.device, dtype=sigma.dtype)
        rr2 = uu ** 2 + vv ** 2


        s2 = (sigma.view(-1, 1, 1) ** 2).clamp(min=1e-12)

        G = torch.exp(-rr2.view(1, H, W) / (2.0 * s2))  # [C,H,W]

        G = G.unsqueeze(0)  # [1,C,H,W]

        if self.cache_mask:
            self._mask_cache[key] = G

        return G

    def _decompose_one(self, F_spatial: torch.Tensor, modality: str) -> torch.Tensor:

        B, C, H, W = F_spatial.shape
        assert C == self.C, f"通道数不匹配: 输入 {C}, 期望 {self.C}"


        F_fre = torch.fft.fft2(F_spatial, dim=(-2, -1))


        F_shift = self._fftshift2(F_fre)


        sigma = self._sigma(modality)
        G = self._gaussian_mask(H, W, sigma, modality=modality)


        F_low = F_shift * G
        F_high = F_shift - F_low

        F_use = F_high if self.use_high else F_low


        F_real = F_use.real
        F_imag = F_use.imag
        F_out = torch.cat([F_real, F_imag], dim=1)
        F_out = self.feat_1(self.feat_2(F_out))

        return F_out

    def forward(self, F_rgb: torch.Tensor, F_ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        out_rgb = self._decompose_one(F_rgb, modality="rgb")
        out_ir = self._decompose_one(F_ir, modality="ir")
        return out_rgb, out_ir


