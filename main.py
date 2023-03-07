import argparse
import os
import warnings

import torch
import torch.nn as nn
from mmcv import Config
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import criterion
import models
from data import dataloader
from evaluate import evaluate
from learner import Learner, Learner_KD
from tools.add_weight_decay import add_weight_decay
from tools.set_up_seed import setup_seed
from train import train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default=None, type=str, help="path of cfg file")
    parser.add_argument("--data_root", default=None, type=str, help="path of data files")
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile(args.cfg_file)
    print("\nDataset:%s T:%s (lr:%e %d in %d) ===%s===> S:%s (lr:%e %d in %d) | img size:%d  batch size:%d"
          % (cfg.dataset, cfg.model_t, cfg.lr_t, cfg.stop_epoch_t, cfg.max_epoch_t, cfg.criterion_t2s_para["name"],
             cfg.model_s, cfg.lr_s, cfg.stop_epoch_s, cfg.max_epoch_s, cfg.img_size, cfg.batch_size))
    torch.cuda.empty_cache()

    setup_seed(0)

    writer = SummaryWriter(comment=" %s T model:%s (lr:%e %din%d) =%s=> S model:%s (lr:%e %din%d)|%d %d"
                           % (cfg.dataset, cfg.model_t, cfg.lr_t, cfg.stop_epoch_t, cfg.max_epoch_t,
                              cfg.criterion_t2s_para["name"], cfg.model_s, cfg.lr_s, cfg.stop_epoch_s,
                              cfg.max_epoch_s, cfg.img_size, cfg.batch_size))

    train_loader, test_loader = dataloader.__dict__[cfg.dataset](cfg, args.data_root)

    weight_decay = 1e-4

    # teacher model & student model
    if "swin" not in cfg.model_t:
        model_teacher = models.__dict__[cfg.model_t](train_loader.num_classes, pretrained=True)
    else:
        model_teacher = models.__dict__[cfg.model_t](train_loader.num_classes, pretrained=True, img_size=cfg.img_size)
    model_teacher = nn.DataParallel(model_teacher)
    model_teacher = model_teacher.cuda()
    parameters_t = add_weight_decay(model_teacher, weight_decay)

    if "swin" not in cfg.model_s:
        model_student = models.__dict__[cfg.model_s](train_loader.num_classes, pretrained=True)
    else:
        model_student = models.__dict__[cfg.model_s](train_loader.num_classes, pretrained=True, img_size=cfg.img_size)
    model_student = nn.DataParallel(model_student)
    model_student = model_student.cuda()
    parameters_s = add_weight_decay(model_student, weight_decay)

    criterion_t = criterion.BCE()

    # teacher model training

    if not cfg.teacher_pretrained:
        optimizer_t = torch.optim.Adam(parameters_t, lr=cfg.lr_t, weight_decay=0)
        scheduler_t = lr_scheduler.OneCycleLR(optimizer_t, max_lr=cfg.lr_t, steps_per_epoch=len(train_loader),
                                              epochs=cfg.max_epoch_t, pct_start=0.2)
        learner_t = Learner(model_teacher, criterion_t, optimizer_t, scheduler_t)

        for epoch in range(cfg.max_epoch_t):
            if epoch >= cfg.stop_epoch_t:
                break
            train(epoch, train_loader, learner_t)
            AP, mAP, of1, cf1 = evaluate(test_loader, model_teacher)
            writer.add_scalar("Teacher mAP", mAP, epoch)
            writer.add_scalar("Teacher OF1", of1, epoch)
            writer.add_scalar("Teacher CF1", cf1, epoch)

        torch.save(model_teacher.state_dict(), "pretrained_models/model_teacher_%s_%s_%d.pth"
                   % (cfg.model_t, cfg.dataset, cfg.img_size))

    else:
        model_teacher.load_state_dict(torch.load("pretrained_models/model_teacher_%s_%s_%d.pth"
                                                 % (cfg.model_t, cfg.dataset, cfg.img_size)))

    model_teacher.eval()
    print("Before distillation, evaluate teacher model and student model firstly:")
    _, mAP_t, of1_t, cf1_t = evaluate(test_loader, model_teacher)
    evaluate(test_loader, model_student)
    print("Finished!\n")

    # student model training

    criterion_s = criterion.BCE()
    criterion_t2s = criterion.distiller_zoo.BaseDistiller(**cfg.criterion_t2s_para["para"])
    optimizer_s = torch.optim.Adam(parameters_s, lr=cfg.lr_s, weight_decay=0)
    scheduler_s = lr_scheduler.OneCycleLR(optimizer_s, max_lr=cfg.lr_s, steps_per_epoch=len(train_loader),
                                          epochs=cfg.max_epoch_s, pct_start=0.2)
    learner_s = Learner_KD(model_teacher, model_student, criterion_s, criterion_t2s, optimizer_s, scheduler_s)

    for epoch in range(cfg.max_epoch_s):
        if epoch >= cfg.stop_epoch_s:
            break
        train(epoch, train_loader, learner_s)
        AP, mAP, of1, cf1 = evaluate(test_loader, model_student)
        writer.add_scalar("Student mAP", mAP, epoch)
        writer.add_scalar("Student OF1", of1, epoch)
        writer.add_scalar("Student CF1", cf1, epoch)

    writer.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if not os.path.exists("runs"):
        os.mkdir("runs")
    args = get_args()
    main(args)
