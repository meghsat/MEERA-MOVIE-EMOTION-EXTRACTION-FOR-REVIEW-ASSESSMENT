

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class ColorConsistencyLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W)
        mean_rgb = x.mean(dim=(2, 3), keepdim=True)  # (B, 3, 1, 1)
        mr, mg, mb = mean_rgb.split(1, dim=1)
        loss = ((mr - mg) ** 2 + (mr - mb) ** 2 + (mg - mb) ** 2).sqrt()
        return loss


class SpatialConsistencyLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        kernels = {
            "left":  [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            "right": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            "up":    [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            "down":  [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        }
        for name, k in kernels.items():
            kernel = torch.tensor(k, dtype=torch.float32)[None, None]
            self.register_buffer(f"weight_{name}", kernel)

        self.avg_pool = nn.AvgPool2d(4)

    def _gradient(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=1)

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        o_lum = original.mean(dim=1, keepdim=True)
        e_lum = enhanced.mean(dim=1, keepdim=True)

        o_pool, e_pool = self.avg_pool(o_lum), self.avg_pool(e_lum)

        diffs = []
        for direction in ("left", "right", "up", "down"):
            k = getattr(self, f"weight_{direction}")
            d = self._gradient(o_pool, k) - self._gradient(e_pool, k)
            diffs.append(d.pow(2))

        return torch.add_n(diffs)


class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size: int = 16, target_mean: float = 0.6) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(patch_size)
        self.register_buffer("target", torch.tensor(target_mean))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_mean = self.avg_pool(x.mean(dim=1, keepdim=True))
        return ((patch_mean - self.target) ** 2).mean()


class TotalVariationLoss(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
        norm = 2.0 * self.weight / x.size(0)
        return norm * (dh / ((x.size(2) - 1) * x.size(3)) + dw / (x.size(2) * (x.size(3) - 1)))


class SaturationLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_rgb = x.mean(dim=(2, 3), keepdim=True)
        deviation = (x - mean_rgb).pow(2).sum(dim=1, keepdim=True).sqrt()
        return deviation.mean()


class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg_feats = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList(
            nn.Sequential(*vgg_feats[s:e])
            for s, e in ((0, 4), (4, 9), (9, 16), (16, 23))
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
