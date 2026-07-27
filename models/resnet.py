"""
ResNet for CIFAR-10 with GroupNorm.

Architecture from He et al. 2016 ("Deep Residual Learning for Image Recognition"),
section 4.2 (CIFAR-10 experiments). Replaces BatchNorm with GroupNorm for
federated learning compatibility.

Total layers = 6n + 2, where n is the number of basic blocks per stage:
  n=1 -> ResNet-8   (~78k parameters)
  n=3 -> ResNet-20  (~270k parameters)
  n=5 -> ResNet-32  (~470k parameters)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type


def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """
    Helper: build a GroupNorm layer with a sensible default num_groups.
    If num_channels is not divisible by num_groups, fall back to a divisor.
    """
    # Common case: 8 divides 16, 32, 64 (the channel counts in CIFAR-ResNets)
    if num_channels % num_groups == 0:
        return nn.GroupNorm(num_groups, num_channels)
    # Fallback: use the largest divisor of num_channels that is <= num_groups
    for g in range(num_groups, 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)  # GroupNorm with 1 group = LayerNorm-like


class BasicBlock(nn.Module):
    """
    Basic ResNet block: two 3x3 conv layers with skip connection.
    Used in CIFAR ResNets (ResNet-8, 20, 32, 56, 110).
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # First conv: may downsample (stride=2) at the start of a new stage
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn1 = _gn(out_channels)

        # Second conv: always stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = _gn(out_channels)

        # Skip connection: if shape changes (stride or channels), use 1x1 conv
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                _gn(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)  # skip connection
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet for CIFAR-10 / CIFAR-100, with GroupNorm.
    Total layers = 6n + 2.
    """

    def __init__(
            self,
            n: int = 3,                       # blocks per stage; n=3 -> ResNet-20
            num_classes: int = 10,
            block: Type[nn.Module] = BasicBlock,
    ):
        super().__init__()
        self.in_channels = 16

        # Initial conv: 3 -> 16, no downsampling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = _gn(16)

        # Three stages, each with n blocks. Channels double each stage; stride=2 between stages.
        self.stage1 = self._make_stage(block, out_channels=16, num_blocks=n, stride=1)
        self.stage2 = self._make_stage(block, out_channels=32, num_blocks=n, stride=2)
        self.stage3 = self._make_stage(block, out_channels=64, num_blocks=n, stride=2)

        # Global avg pool + linear classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Init weights (Kaiming for conv, default for linear)
        self._init_weights()

    def _make_stage(self, block, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Build one stage of `num_blocks` residual blocks.
        Only the first block in the stage may downsample (stride > 1).
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)))     # (B, 16, 32, 32)
        x = self.stage1(x)                      # (B, 16, 32, 32)
        x = self.stage2(x)                      # (B, 32, 16, 16)
        x = self.stage3(x)                      # (B, 64, 8, 8)
        x = self.pool(x)                        # (B, 64, 1, 1)
        x = x.flatten(start_dim=1)              # (B, 64)
        x = self.fc(x)                          # (B, num_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_resnet(
        depth: int = 20,
        num_classes: int = 10,
        device: Optional[torch.device] = None,
) -> ResNetCIFAR:
    """
    Factory function to build a CIFAR ResNet of the given depth.
    Allowed depths: 8, 14, 20, 26, 32, 38, 44, 50, 56, ... (formula: 6n+2)
    """
    if (depth - 2) % 6 != 0:
        raise ValueError(f"Invalid depth {depth}: must be 6n+2 (e.g. 8, 14, 20, 32, 56)")
    n = (depth - 2) // 6
    model = ResNetCIFAR(n=n, num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Building CIFAR ResNets")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for depth in [8, 20, 32]:
        model = build_resnet(depth=depth, device=device)
        n_params = count_parameters(model)
        print(f"\nResNet-{depth}: {n_params:,} parameters (~{n_params / 1e3:.1f} k)")

    # Forward pass test on ResNet-20
    print("\n" + "=" * 60)
    print("Forward pass test (ResNet-20)")
    print("=" * 60)
    model = build_resnet(depth=20, device=device)

    batch_size = 16
    fake_input = torch.randn(batch_size, 3, 32, 32, device=device)
    print(f"Input shape:  {tuple(fake_input.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(fake_input)
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (batch_size, 10)
    print("\n✓ Sanity check passed.")