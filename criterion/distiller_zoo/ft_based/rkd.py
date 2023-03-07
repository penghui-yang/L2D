import torch
import torch.nn as nn
import torch.nn.functional as F


class RkdAngle(nn.Module):
    @staticmethod
    def forward(student, teacher):
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
    @staticmethod
    def forward(student, teacher):
        N, C = student.shape

        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction="sum") / N
        return loss


class RKD(nn.Module):
    def __init__(self, x=10, y=10):
        super().__init__()
        self.alpha_1 = x
        self.alpha_2 = y
        self.rkd_dist = RkdDistance()
        self.rkd_angle = RkdAngle()

    def forward(self, f_s, f_t, out_student, out_teacher, targets):
        f_s, f_t = f_s[-1], f_t[-1]
        rkd_dist_loss = self.rkd_dist(f_s, f_t)
        rkd_angle_loss = self.rkd_angle(f_s, f_t)
        loss = rkd_dist_loss * self.alpha_1 + rkd_angle_loss * self.alpha_2
        return loss
