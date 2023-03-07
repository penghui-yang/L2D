from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.channels = [in_channel, out_channel]
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))
        # output
        y = self.conv2(x)
        return y, x


def get_channel(net, img_size):
    net = net
    data = torch.randn(4, 3, img_size, img_size)
    net.eval()
    feat, _ = net(data, ft=True)
    channels = []
    for i in range(len(feat)):
        assert len(feat[i].shape) in [2, 3, 4]
        if len(feat[i].shape) == 3:  # swin transformer
            channels.append(feat[i].shape[-1])
        else:
            channels.append(feat[i].shape[1])
    return channels


class ReviewKDModel(nn.Module):
    def __init__(self, teacher, student, img_size=224):
        super(ReviewKDModel, self).__init__()
        self.student = student
        self.shapes = [1, 7, 14, 28, 56]
        in_channels = get_channel(student, img_size)
        out_channels = get_channel(teacher, img_size)
        print(in_channels, out_channels)
        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, 256, out_channels[idx], idx < len(in_channels) - 1))
        self.abfs = abfs[::-1]

    def forward(self, x, le: bool = False, ft: bool = False):
        student_features = self.student(x, le=le, ft=ft)
        if not ft:
            return student_features
        assert len(student_features) == 2
        logit = student_features[1]
        fstudent = student_features[0]
        for i in range(len(fstudent)):
            if len(fstudent[i].shape) == 3:
                n, hw, c = fstudent[i].shape
                h = w = int(sqrt(hw))
                fstudent[i] = fstudent[i].transpose(1, 2).reshape(n, c, h, w)
            elif len(fstudent[i].shape) == 2:
                fstudent[i] = fstudent[i].unsqueeze(2)
                fstudent[i] = fstudent[i].unsqueeze(3)
            # print("s", i, ":", fstudent[i].shape)

        x = fstudent[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        return results, logit


class ReviewKD(nn.Module):
    def __init__(self, review_kd_loss_weight=1.0):
        super(ReviewKD, self).__init__()
        self.review_kd_loss_weight = review_kd_loss_weight

    def forward(self, fstudent, fteacher, out_student, out_teacher, targets):
        fstudent = fstudent[::-1]
        fteacher = fteacher[::-1]
        for i in range(len(fteacher)):
            if len(fteacher[i].shape) == 3:
                n, hw, c = fteacher[i].shape
                h = w = int(sqrt(hw))
                fteacher[i] = fteacher[i].transpose(1, 2).reshape(n, c, h, w)
            elif len(fteacher[i].shape) == 2:
                fteacher[i] = fteacher[i].unsqueeze(2)
                fteacher[i] = fteacher[i].unsqueeze(3)
            # print("t", i, ":", fteacher[i].shape)

        # print("======================================")
        # for fs in fstudent:
        #     print(fs.shape)
        # print("--------------------------------------")
        # for ft in fteacher:
        #     print(ft.shape)
        # print("======================================")

        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            n, c, h, w = fs.shape
            loss = F.mse_loss(fs, ft, reduction="mean")
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                tmpft = F.adaptive_avg_pool2d(ft, (l, l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss

        loss_all = loss_all * self.review_kd_loss_weight
        return loss_all
