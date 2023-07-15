import torch
import torch.nn as nn


class PartialSoftmaxDistiller(nn.Module):
    
    def __init__(self, reduction="sum"):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, student, teacher, target):
        n_pos = torch.sum(target, dim=1)
        N, C = target.shape
        loss = 0.0
        for i in range(N):
            if n_pos[i] > 0:
                neg_s = torch.masked_select(student[i, :], target[i, :] == 0)
                pos_s = torch.masked_select(student[i, :], target[i, :] == 1)
                neg_t = torch.masked_select(teacher[i, :], target[i, :] == 0)
                pos_t = torch.masked_select(teacher[i, :], target[i, :] == 1)

                z_s = torch.hstack((neg_s.repeat(int(n_pos[i].item()), 1), torch.unsqueeze(pos_s, dim=1)))
                z_t = torch.hstack((neg_t.repeat(int(n_pos[i].item()), 1), torch.unsqueeze(pos_t, dim=1)))

                p_s = torch.softmax(z_s, dim=1)
                p_t = torch.softmax(z_t, dim=1)

                loss += self.criterion(torch.log(p_s), p_t)

        loss /= N
        return loss
