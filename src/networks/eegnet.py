import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, chunk_size=1750, n_channels=22, F1=8, F2=16, D=2, num_classes=4,
                kernel_1=64, kernel_2=16, dropout=0.5, max_norm=1.0):
        super(EEGNet, self).__init__()

        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.max_norm = max_norm

        # Block 1: Temporal convolution + Depthwise spatial convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1,
                      padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(self.F1, self.F1 * self.D, (self.n_channels, 1),
                      stride=1, padding=(0, 0), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01,
                           affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout)
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_2), stride=1,
                      padding=(0, self.kernel_2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1,
                      padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout)
        )

        # Classifier
        self.linear = nn.Linear(self._get_feature_dim(), num_classes, bias=False)

        # Apply max norm constraint
        self._apply_max_norm_constraint()

    def _get_feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.chunk_size)
            x = self.block1(x)
            x = self.block2(x)
        return self.F2 * x.shape[3]

    def _apply_max_norm_constraint(self):
        depthwise_layer = self.block1[2]
        if hasattr(depthwise_layer, 'weight'):
            with torch.no_grad():
                depthwise_layer.weight.data = torch.renorm(
                    depthwise_layer.weight.data, p=2, dim=0, maxnorm=self.max_norm
                )

    def _enforce_max_norm(self):
        depthwise_layer = self.block1[2]
        if hasattr(depthwise_layer, 'weight'):
            with torch.no_grad():
                depthwise_layer.weight.data = torch.renorm(
                    depthwise_layer.weight.data, p=2, dim=0, maxnorm=self.max_norm
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._enforce_max_norm()
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x