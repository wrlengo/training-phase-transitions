"""
Underparameterized ConvNet for CIFAR-10 (~19K parameters with base channels=8).

Three groups of two 3x3 conv+BN+ReLU layers (channels c, 2c, 4c), max-pool
after each group, global average pooling, linear classifier.
"""
import torch
import torch.nn as nn


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UnderparamConvNet(nn.Module):
    def __init__(self, num_channels=8, num_classes=10):
        c = num_channels
        super().__init__()
        self.stem = ConvBN(3, c)
        self.group1 = nn.Sequential(ConvBN(c, c), ConvBN(c, c))
        self.group2 = nn.Sequential(ConvBN(c, 2 * c), ConvBN(2 * c, 2 * c))
        self.group3 = nn.Sequential(ConvBN(2 * c, 4 * c), ConvBN(4 * c, 4 * c))
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 * c, num_classes)

    def features(self, x):
        """Penultimate features (after GAP, before the classifier)."""
        x = self.stem(x)
        x = self.pool(self.group1(x))     # 16x16
        x = self.pool(self.group2(x))     # 8x8
        x = self.pool(self.group3(x))     # 4x4
        return self.gap(x).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


def collect_weight_tensor(model):
    """All conv/fc weight matrices concatenated (BN params excluded) for the
    weight-distribution entropy S."""
    ws = []
    for name, p in model.named_parameters():
        if name.endswith("weight") and p.dim() >= 2:
            ws.append(p.detach().cpu().numpy().ravel())
    import numpy as np
    return np.concatenate(ws)


if __name__ == "__main__":
    m = UnderparamConvNet(8)
    print("ConvNet params:", sum(p.numel() for p in m.parameters()))
    print("weight tensor size:", collect_weight_tensor(m).shape)
