import torch
import torch.nn as nn
from torch.nn import KLDivLoss


class MLD(nn.Module):
    def __init__(self, reduction="batchmean", eps=1e-8):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.criterion = KLDivLoss(reduction="none")

    def forward(self, student, teacher, *args):
        N, C = student.shape
        student = torch.sigmoid(student)
        teacher = torch.sigmoid(teacher)
        student = torch.clamp(student, min=self.eps, max=1-self.eps)
        teacher = torch.clamp(teacher, min=self.eps, max=1-self.eps)
        loss = self.criterion(torch.log(student), teacher) + self.criterion(torch.log(1 - student), 1 - teacher)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "batchmean":
            loss = loss.sum() / N
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise AttributeError
        return loss
