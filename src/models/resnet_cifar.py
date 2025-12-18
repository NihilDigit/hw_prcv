from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_shortcut: bool = True,
        shortcut_type: str = "A",
    ):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_shortcut = use_shortcut
        self.shortcut_type = shortcut_type.upper()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if not use_shortcut:
            self.proj = None
        elif stride != 1 or in_channels != out_channels:
            if self.shortcut_type == "B":
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.proj = None  # Option A handled in forward
        else:
            self.proj = None

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_shortcut:
            return torch.zeros_like(x)  # never used; just for type completeness

        if self.proj is not None:
            return self.proj(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            return x

        # Option A (CIFAR): downsample by stride + zero-pad channels
        x = x[:, :, :: self.stride, :: self.stride]
        if self.out_channels > self.in_channels:
            pad_channels = self.out_channels - self.in_channels
            zeros = torch.zeros(
                x.size(0),
                pad_channels,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, zeros], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_shortcut:
            out = out + self._shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class _CifarStem(nn.Module):
    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.conv = _conv3x3(3, out_channels, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _CifarNetBase(nn.Module):
    def __init__(
        self,
        depth: int,
        num_classes: int = 10,
        use_shortcut: bool = True,
        shortcut_type: str = "A",
    ):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"CIFAR ResNet depth must be 6n+2, got {depth}")

        n = (depth - 2) // 6
        self.stem = _CifarStem(16)

        self.in_channels = 16
        self.layer1 = self._make_layer(16, n, stride=1, use_shortcut=use_shortcut, shortcut_type=shortcut_type)
        self.layer2 = self._make_layer(32, n, stride=2, use_shortcut=use_shortcut, shortcut_type=shortcut_type)
        self.layer3 = self._make_layer(64, n, stride=2, use_shortcut=use_shortcut, shortcut_type=shortcut_type)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
        use_shortcut: bool,
        shortcut_type: str,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
                use_shortcut=use_shortcut,
                shortcut_type=shortcut_type,
            )
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                    use_shortcut=use_shortcut,
                    shortcut_type=shortcut_type,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class PlainCifarNet(_CifarNetBase):
    def __init__(self, depth: int, num_classes: int = 10):
        super().__init__(depth=depth, num_classes=num_classes, use_shortcut=False)


class ResNetCifar(_CifarNetBase):
    def __init__(self, depth: int, num_classes: int = 10, shortcut_type: str = "A"):
        super().__init__(depth=depth, num_classes=num_classes, use_shortcut=True, shortcut_type=shortcut_type)


def cifar_plain(depth: int, num_classes: int = 10) -> PlainCifarNet:
    return PlainCifarNet(depth=depth, num_classes=num_classes)


def cifar_resnet(depth: int, num_classes: int = 10, shortcut_type: str = "A") -> ResNetCifar:
    return ResNetCifar(depth=depth, num_classes=num_classes, shortcut_type=shortcut_type)

