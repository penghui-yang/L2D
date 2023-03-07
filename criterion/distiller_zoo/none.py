import torch
import torch.nn as nn


class NONE(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(*args):
        return torch.tensor(0)
