from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50


class _ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        y = super().forward(x)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


class _ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: tuple[int, int, int]):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                _ASPPConv(in_channels, out_channels, atrous_rates[0]),
                _ASPPConv(in_channels, out_channels, atrous_rates[1]),
                _ASPPConv(in_channels, out_channels, atrous_rates[2]),
                _ASPPPooling(in_channels, out_channels),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(self.branches), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class _ResNetBackbone(nn.Module):
    def __init__(self, name: str, *, output_stride: int = 8, pretrained: bool = True):
        super().__init__()
        if output_stride not in (8, 16, 32):
            raise ValueError("output_stride must be 8, 16, or 32")

        if name == "resnet18":
            if output_stride != 32:
                raise ValueError("resnet18 backbone only supports output_stride=32 (no dilation in BasicBlock)")
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            out_channels = 512
            low_channels = 64
        elif name == "resnet34":
            if output_stride != 32:
                raise ValueError("resnet34 backbone only supports output_stride=32 (no dilation in BasicBlock)")
            base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            out_channels = 512
            low_channels = 64
        elif name == "resnet50":
            if output_stride == 8:
                replace_stride_with_dilation = [False, True, True]
                aspp_rates = (12, 24, 36)
            elif output_stride == 16:
                replace_stride_with_dilation = [False, False, True]
                aspp_rates = (6, 12, 18)
            else:
                replace_stride_with_dilation = [False, False, False]
                aspp_rates = (3, 6, 9)
            base = resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            out_channels = 2048
            low_channels = 256
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        if name in ("resnet18", "resnet34"):
            aspp_rates = (3, 6, 9)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.out_channels = out_channels
        self.low_channels = low_channels
        self.aspp_rates = aspp_rates

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low = self.layer1(x)  # 1/4
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)  # 1/8 or 1/16
        return x, low


@dataclass(frozen=True)
class DeepLabV3PlusConfig:
    num_classes: int = 11
    backbone: str = "resnet50"
    backbone_pretrained: bool = True
    output_stride: int = 8
    aspp_channels: int = 256
    low_level_channels: int = 48


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg: DeepLabV3PlusConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = _ResNetBackbone(
            cfg.backbone,
            output_stride=cfg.output_stride,
            pretrained=cfg.backbone_pretrained,
        )
        self.aspp = _ASPP(self.backbone.out_channels, cfg.aspp_channels, self.backbone.aspp_rates)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(self.backbone.low_channels, cfg.low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.low_level_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(cfg.aspp_channels + cfg.low_level_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cfg.num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        high, low = self.backbone(x)
        high = self.aspp(high)
        low = self.low_level_proj(low)

        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([high, low], dim=1)
        x = self.decoder(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
