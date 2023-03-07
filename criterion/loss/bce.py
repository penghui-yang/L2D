import torch
from torch import nn


class BCE(nn.Module):
    def __init__(self, reduction="sum", eps=1e-8):
        super(BCE, self).__init__()
        self.reduction = reduction
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.loss = None

    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        if self.reduction == "sum":
            return -self.loss.sum()
        elif self.reduction == "none":
            return -self.loss
        else:
            raise AttributeError
