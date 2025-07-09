import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def decor_modules(model):
    return [m for m in model.modules() if isinstance(m, Decorrelation)]

def decor_update(modules):
    losses = np.zeros(len(modules))
    for i, m in enumerate(modules):
        loss_val = m.update().cpu().detach().numpy()
        losses[i] = loss_val if not np.isnan(loss_val) else 0.0
    return losses


class Decorrelation(nn.Module):
    def __init__(self, in_features, method='standard', decor_lr=0.01, kappa=0.5,
                full=True, downsample_perc=1.0, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.decor_lr = decor_lr
        self.downsample_perc = downsample_perc
        self.method = method
        self.kappa = kappa
        self.full = full
        self.device = device

        self.register_buffer("weight", torch.eye(in_features, device=device, dtype=dtype))
        self.decor_state = None

    def forward(self, input):
        if self.training:
            if self.downsample_perc != 1.0:
                self.decor_state = F.linear(self.downsample(
                    input).view(-1, np.prod(input.shape[1:])), self.weight)
                return self.decorrelate(input)
            else:
                self.decor_state = F.linear(
                    input.view(len(input), -1), self.weight)
                return self.decor_state.view(input.shape)
        else:
            return self.decorrelate(input)

    def decorrelate(self, input):
        return F.linear(input.view(len(input), -1), self.weight).view(input.shape)

    def update(self, loss_only=False, eps=1e-8):
        if self.decor_state is None or len(self.decor_state) == 0:
            return torch.tensor(0.0, device=self.weight.device)

        X = self.decor_state.T @ self.decor_state / len(self.decor_state)
        X = X + eps * torch.eye(X.size(0), device=X.device, dtype=X.dtype)
        
        if self.full:
            C = X - torch.diag(torch.diag(X))
        else:
            C = torch.sqrt(torch.arange(self.in_features).to(self.device)) * torch.tril(X.to(self.device), diagonal=-1)

        v = torch.mean(torch.clamp(self.decor_state ** 2 - 1.0, -10.0, 10.0), dim=0)

        match self.method:
            case 'standard':
                if not loss_only:
                    update_term = (1.0 - self.kappa) * C @ self.weight + self.kappa * v * self.weight
                    update_term = torch.clamp(update_term, -1.0, 1.0)
                    self.weight.data -= self.decor_lr * update_term
                loss = ((1-self.kappa) * torch.sum(C**2) + self.kappa * torch.sum(v**2)) / self.in_features**2
                if torch.isnan(loss) or torch.isinf(loss):
                    return torch.tensor(0.0, device=self.weight.device)
                return loss

            case 'normalized':
                if not loss_only:
                    update_term = ((1.0 - self.kappa)/(self.in_features-1)) * C @ self.weight + self.kappa * 2 * v * self.weight
                    update_term = torch.clamp(update_term, -1.0, 1.0)
                    self.weight.data -= self.decor_lr * update_term
                loss = (1/self.in_features) * (((1-self.kappa)/(self.in_features-1)) * torch.sum(C**2) + self.kappa * torch.sum(v**2))
                if torch.isnan(loss) or torch.isinf(loss):
                    return torch.tensor(0.0, device=self.weight.device)
                return loss

            case _:
                raise ValueError(f"Unknown method: {self.method}")

    def loss(self):
        return self.update(loss_only=True)

    def downsample(self, input):
        if self.downsample_perc < 1.0:
            num_samples = max(1, int(len(input) * self.downsample_perc))
            idx = np.random.choice(np.arange(len(input)), size=num_samples, replace=False)
            return input[idx]
        return input


class DecorLinear(Decorrelation):
    def __init__(self, in_features, out_features, bias=True, method='standard',
                decor_lr=0.01, kappa=0.5, full=True, downsample_perc=1.0, device=None, dtype=None):
        super().__init__(in_features, method, decor_lr, kappa, full, downsample_perc, device, dtype)
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, input):
        return self.linear(super().forward(input))


class DecorConv2d(Decorrelation):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                dilation=1, groups=1, bias=True, method='standard', decor_lr=0.01,
                kappa=0.5, full=True, downsample_perc=1.0, device=None, dtype=None):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        super().__init__(in_channels * np.prod(kernel_size), method, decor_lr, kappa, full, downsample_perc, device, dtype)

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                            dilation, groups, bias, device=device, dtype=dtype)

    def forward(self, input):
        if self.training:
            patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
            patches = patches.transpose(1, 2).reshape(-1, self.in_features)

            if patches.size(0) > 0:
                self.decor_state = self.downsample(patches)
                self.decor_state = F.linear(self.decor_state, self.weight)
            else:
                self.decor_state = None

        return self.conv(input)


class DecorrelatedEEGNet(nn.Module):
    def __init__(self, chunk_size=1750, n_channels=22, F1=8, F2=16, D=2, num_classes=4,
                kernel_1=64, kernel_2=16, dropout=0.5, max_norm=1.0,
                method='standard', decor_lr=0.01, kappa=0.5, full=True, downsample_perc=1.0,
                device=None, dtype=None):
        super(DecorrelatedEEGNet, self).__init__()

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
            DecorConv2d(1, self.F1, (1, self.kernel_1), stride=1,
                        padding=(0, self.kernel_1 // 2), bias=False,
                        method=method, decor_lr=decor_lr, kappa=kappa,
                        full=full, downsample_perc=downsample_perc,
                        device=device, dtype=dtype),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            DecorConv2d(self.F1, self.F1 * self.D, (self.n_channels, 1),
                        stride=1, padding=(0, 0), groups=self.F1, bias=False,
                        method=method, decor_lr=decor_lr, kappa=kappa,
                        full=full, downsample_perc=downsample_perc,
                        device=device, dtype=dtype),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01,
                           affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout)
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            DecorConv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_2), stride=1,
                        padding=(0, self.kernel_2 // 2), bias=False, groups=self.F1 * self.D,
                        method=method, decor_lr=decor_lr, kappa=kappa,
                        full=full, downsample_perc=downsample_perc,
                        device=device, dtype=dtype),
            DecorConv2d(self.F1 * self.D, self.F2, 1,
                        padding=(0, 0), groups=1, bias=False, stride=1,
                        method=method, decor_lr=decor_lr, kappa=kappa,
                        full=full, downsample_perc=downsample_perc,
                        device=device, dtype=dtype),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout)
        )

        # Classifier
        self.linear = DecorLinear(self._get_feature_dim(), num_classes, bias=False,
                                method=method, decor_lr=decor_lr, kappa=kappa,
                                full=full, downsample_perc=downsample_perc,
                                device=device, dtype=dtype)

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

    def forward(self, x):
        self._enforce_max_norm()
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x