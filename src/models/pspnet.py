from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34, resnet50


class _PPM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_sizes: tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(s, s)),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for s in pool_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(y)
        return torch.cat(feats, dim=1)


class _ResNetBackbone(nn.Module):
    def __init__(self, name: str, *, output_stride: int = 8, pretrained: bool = True):
        super().__init__()
        if output_stride not in (8, 16, 32):
            raise ValueError("output_stride must be 8, 16, or 32")

        if name == "resnet18":
            # torchvision BasicBlock does not support dilation>1
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
            elif output_stride == 16:
                replace_stride_with_dilation = [False, False, True]
            else:
                replace_stride_with_dilation = [False, False, False]
            base = resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
            out_channels = 2048
            low_channels = 256
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.out_channels = out_channels
        self.low_channels = low_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low = self.layer1(x)  # 1/4 resolution features
        x = self.layer2(low)
        aux = self.layer3(x)
        out = self.layer4(aux)
        return out, aux


@dataclass(frozen=True)
class PSPNetConfig:
    num_classes: int = 11
    backbone: str = "resnet50"
    backbone_pretrained: bool = True
    output_stride: int = 8
    ppm_out_channels: int = 128
    dropout: float = 0.1
    aux_loss: bool = True


class PSPNet(nn.Module):
    def __init__(self, cfg: PSPNetConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = _ResNetBackbone(
            cfg.backbone,
            output_stride=cfg.output_stride,
            pretrained=cfg.backbone_pretrained,
        )

        ppm_out = cfg.ppm_out_channels
        self.ppm = _PPM(self.backbone.out_channels, ppm_out, pool_sizes=(1, 2, 3, 6))

        ppm_concat_channels = self.backbone.out_channels + 4 * ppm_out
        self.classifier = nn.Sequential(
            nn.Conv2d(ppm_concat_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=cfg.dropout),
            nn.Conv2d(512, cfg.num_classes, kernel_size=1),
        )

        self.aux_classifier = None
        if cfg.aux_loss:
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(self._aux_in_channels(cfg.backbone), 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=cfg.dropout),
                nn.Conv2d(256, cfg.num_classes, kernel_size=1),
            )

    @staticmethod
    def _aux_in_channels(backbone: str) -> int:
        return 256 if backbone in ("resnet18", "resnet34") else 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_size = x.shape[-2:]
        feat, aux = self.backbone(x)
        feat = self.ppm(feat)
        logits = self.classifier(feat)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        if self.aux_classifier is None or not self.training:
            return logits

        aux_logits = self.aux_classifier(aux)
        aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
        return logits, aux_logits
