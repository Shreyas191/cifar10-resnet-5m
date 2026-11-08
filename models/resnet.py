"""
Configurable ResNet for CIFAR-10 under a 5M-parameter budget.

Design variables (from the project spec):
  - C_i : channels in stage i            -> `channels`
  - N_i : residual blocks in stage i     -> `num_blocks`
  - F_i : conv kernel size in stage i    -> `kernel_size`
  - K_i : kernel size of skip connection -> `skip_kernel` (1 = 1x1 projection)
  - P   : average-pool size              -> `pool_size`  (adaptive avg pool to PxP)

Extras that cost almost no parameters but reliably help:
  - Squeeze-and-Excitation (SE) attention in each block (`use_se`)
  - Dropout before the classifier (`dropout`)

Adapted from the structure of kuangliu/pytorch-cifar (ResNet18), heavily modified.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]


class BasicBlock(nn.Module):
    """ReLU(S(x) + F(x)),  F = conv-BN-ReLU-conv-BN,  S = identity or 1x1-conv-BN."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=3, skip_kernel=1, use_se=False):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, skip_kernel, stride, skip_kernel // 2, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks=(2, 2, 2, 2),
        channels=(64, 128, 256, 512),
        kernel_size=3,
        skip_kernel=1,
        pool_size=1,
        use_se=False,
        dropout=0.0,
        num_classes=10,
        stem_channels=None,
    ):
        super().__init__()
        assert len(num_blocks) == len(channels)
        stem_channels = stem_channels or channels[0]
        self.conv1 = nn.Conv2d(3, stem_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_channels)

        layers = []
        in_planes = stem_channels
        for i, (n, c) in enumerate(zip(num_blocks, channels)):
            stride = 1 if i == 0 else 2
            for j in range(n):
                layers.append(
                    BasicBlock(in_planes, c, stride if j == 0 else 1, kernel_size, skip_kernel, use_se)
                )
                in_planes = c
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(in_planes * pool_size * pool_size, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layers(out)
        out = self.pool(out).flatten(1)
        return self.linear(self.dropout(out))


# ---------------------------------------------------------------------------
# Named configurations.  All are < 5,000,000 trainable parameters.
# ---------------------------------------------------------------------------
CONFIGS = {
    # Reference (NOT allowed: 11.17M params) -- kept only for comparison in the report.
    "resnet18": dict(num_blocks=(2, 2, 2, 2), channels=(64, 128, 256, 512)),
    # Three-stage ResNet18-style: ~2.78M params. Fast, good baseline (~94-95%).
    "resnet18_3stage": dict(num_blocks=(2, 2, 2), channels=(64, 128, 256)),
    # Four stages, last stage narrowed to fit budget: ~4.94M.
    "resnet_c368": dict(num_blocks=(2, 2, 2, 1), channels=(64, 128, 256, 368)),
    # Deeper 3-stage with SE, 3x3 projection shortcuts: ~4.99M.  Our main competition model.
    "resnet_se_343": dict(num_blocks=(3, 4, 3), channels=(64, 128, 256), use_se=True, skip_kernel=3, dropout=0.1),
    # Same depth, 1x1 shortcuts (fewer params, ~4.66M) -- ablation partner of the above.
    "resnet_se_343_k1": dict(num_blocks=(3, 4, 3), channels=(64, 128, 256), use_se=True, dropout=0.1),
    # Wider shallow 3-stage with SE: ~4.97M.
    "resnet_wide3": dict(num_blocks=(2, 3, 2), channels=(96, 160, 320), use_se=True, dropout=0.1),
    # N=(2,1,1,1) C=(64,128,256,512): ~4.98M -- classic "just under 5M" four-stage design.
    "resnet_2111": dict(num_blocks=(2, 1, 1, 1), channels=(64, 128, 256, 512)),
}


def build_model(name: str = "resnet_se_343", **overrides) -> ResNet:
    cfg = dict(CONFIGS[name])
    cfg.update(overrides)
    return ResNet(**cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in CONFIGS:
        m = build_model(name)
        n = count_parameters(m)
        flag = "OK " if n <= 5_000_000 else "OVER"
        print(f"{flag} {name:18s} {n:>12,d}")
