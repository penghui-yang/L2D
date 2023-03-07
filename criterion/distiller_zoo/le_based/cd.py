import torch.nn as nn

from criterion.distiller_zoo.le_based.huber_dist import HuberDist


class CD(nn.Module):  # class-aware label-wise embedding distillation
    def __init__(self):
        super().__init__()
        self.le_distill_criterion = HuberDist()

    def forward(self, le_student, le_teacher, targets):
        N, C, le_length = le_student.shape
        le_mask = targets.unsqueeze(2).repeat(1, 1, le_length)
        le_student_pos = le_student * le_mask
        le_teacher_pos = le_teacher * le_mask
        n_pos_per_label = targets.sum(dim=0)
        loss = 0.0

        for c in range(C):
            if n_pos_per_label[c] > 1:
                le_s_c = le_student_pos[:, c, :]
                le_t_c = le_teacher_pos[:, c, :]
                le_s_pos_c = le_s_c[~(le_s_c == 0).all(1)]
                le_t_pos_c = le_t_c[~(le_t_c == 0).all(1)]
                delta_loss = self.le_distill_criterion(le_s_pos_c, le_t_pos_c)
                loss += delta_loss

        return loss
