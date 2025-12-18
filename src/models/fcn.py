from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, resnet18, resnet34
from torchvision.models.segmentation import fcn_resnet50


class _FCNResNetLite(nn.Module):
    def __init__(self, name: str, *, num_classes: int, pretrained: bool = True):
        super().__init__()
        if name == "resnet18":
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            out_channels = 512
        elif name == "resnet34":
            base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone for _FCNResNetLite: {name}")

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.classifier = nn.Sequential(
            nn.Conv2d(out_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # output_stride=32
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"out": logits}


@dataclass(frozen=True)
class FCNConfig:
    num_classes: int = 11
    backbone: str = "resnet50"
    backbone_pretrained: bool = True


def build_fcn(cfg: FCNConfig) -> nn.Module:
    if cfg.backbone == "resnet50":
        try:
            return fcn_resnet50(
                weights=None,
                weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if cfg.backbone_pretrained else None,
                num_classes=cfg.num_classes,
            )
        except (TypeError, ValueError):
            return fcn_resnet50(
                weights=None,
                weights_backbone="IMAGENET1K_V1" if cfg.backbone_pretrained else None,
                num_classes=cfg.num_classes,
            )

    if cfg.backbone in ("resnet18", "resnet34"):
        return _FCNResNetLite(cfg.backbone, num_classes=cfg.num_classes, pretrained=cfg.backbone_pretrained)

    raise ValueError(f"Unsupported FCN backbone: {cfg.backbone}")

