import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


class Learner(nn.Module):
    def __init__(self, model, criterion, optimizer, scheduler):
        super(Learner, self).__init__()
        self.model = model
        self.scaler = GradScaler()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

    def forward(self, x):
        return self.model.forward(x)

    def forward_with_criterion(self, inputs, targets):
        with autocast():
            out = self.forward(inputs).float()
        return self.criterion(out, targets), out

    def learn(self, inputs, targets):
        loss, out = self.forward_with_criterion(inputs, targets)
        self.model.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss, out


class Learner_KD(nn.Module):
    def __init__(self, model_t, model_s, criterion_s, criterion_t2s, optimizer, scheduler):
        super(Learner_KD, self).__init__()
        self.model_t = model_t
        self.model_s = model_s
        self.scaler = GradScaler()
        self.criterion_s = criterion_s
        self.criterion_t2s = criterion_t2s
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

    def forward_with_criterion(self, inputs, targets):
        with autocast():
            f_t, le_t, logits_t = self.model_t.forward(inputs, le=True, ft=True)
            f_s, le_s, logits_s = self.model_s.forward(inputs, le=True, ft=True)
        f_t, le_t, logits_t = f_t.float().detach(), le_t.float().detach(), logits_t.float().detach()
        f_s, le_s, logits_s = f_s.float(), le_s.float(), logits_s.float()
        targets = targets.to(torch.float)
        loss_hard = self.criterion_s(logits_s, targets)
        loss_soft = self.criterion_t2s(f_s, f_t, le_s, le_t, logits_s, logits_t, targets)
        loss = loss_hard + loss_soft
        return loss, logits_s

    def learn(self, inputs, targets):
        loss, out = self.forward_with_criterion(inputs, targets)
        self.model_s.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss, out

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.model_t.eval()
        return self
