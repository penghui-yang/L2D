import torch
import torch.nn as nn
import torch.nn.functional as F


def phi(e, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps).sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class HuberDist(nn.Module):
    @staticmethod
    def forward(student, teacher):
        N, C = student.shape

        with torch.no_grad():
            t_d = phi(teacher)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = phi(student)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction="sum") / N
        return loss
