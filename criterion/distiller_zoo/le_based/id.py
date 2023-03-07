import torch.nn as nn

from criterion.distiller_zoo.le_based.huber_dist import HuberDist


class ID(nn.Module):  # instance-aware label-wise embedding distillation
    def __init__(self):
        super().__init__()
        self.le_distill_criterion = HuberDist()

    def forward(self, le_student, le_teacher, targets):
        N, C, le_length = le_student.shape
        le_mask = targets.unsqueeze(2).repeat(1, 1, le_length)
        le_student_pos = le_student * le_mask
        le_teacher_pos = le_teacher * le_mask
        n_pos_per_instance = targets.sum(dim=1)
        loss = 0.0

        for i in range(N):
            if n_pos_per_instance[i] > 1:
                le_s_i = le_student_pos[i, :, :]
                le_t_i = le_teacher_pos[i, :, :]
                le_s_pos_i = le_s_i[~(le_s_i == 0).all(1)]
                le_t_pos_i = le_t_i[~(le_t_i == 0).all(1)]
                delta_loss = self.le_distill_criterion(le_s_pos_i, le_t_pos_i)
                loss += delta_loss

        return loss
