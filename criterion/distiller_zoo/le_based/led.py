from torch import nn

from criterion.distiller_zoo.le_based.cd import CD
from criterion.distiller_zoo.le_based.id import ID


class LED(nn.Module):
    def __init__(self, lambda_cd=100.0, lambda_id=1000.0):
        super().__init__()
        self.cd_distiller = CD()
        self.id_distiller = ID()
        self.lambda_cd = lambda_cd
        self.lambda_id = lambda_id

    def forward(self, le_student, le_teacher, targets):
        loss_cd = self.cd_distiller(le_student, le_teacher, targets)
        loss_id = self.id_distiller(le_student, le_teacher, targets)
        loss = self.lambda_cd * loss_cd + self.lambda_id * loss_id
        return loss
