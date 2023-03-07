import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.cuda.amp import autocast

from models.repvgg import repvgg_model_convert


def compute_mAP(labels, outputs):
    APs = []
    for j in range(labels.shape[1]):
        new_AP = average_precision_score(labels[:, j], outputs[:, j])
        APs.append(new_AP)
    mAP = np.mean(APs)
    return APs, mAP


def test(outputs, targets):
    idxs = np.sum(targets == 1, axis=1).astype(int)
    sorted_outputs = np.sort(-outputs, axis=1)
    thr = -sorted_outputs[range(len(targets)), idxs].reshape(len(sorted_outputs), 1)

    preds = np.zeros(outputs.shape, dtype=np.int64)
    preds[outputs > thr] = 1

    APs, mAP = compute_mAP(targets, outputs)  # average precision & mean average precision
    of1 = f1_score(targets, preds, average="micro")  # overall f1 score
    cf1 = f1_score(targets, preds, average="macro")  # per-class f1 score

    print("  mAP: {:.2f}  OF1: {:.2f}  CF1: {:.2f}".format(mAP * 100, of1 * 100, cf1 * 100))

    return APs, mAP, of1, cf1


def evaluate(eval_loader, model):
    print(" Evaluation: ", end="")

    if hasattr(model, "repvgg_flag") and model.repvgg_flag:  # repvgg
        deploy_model = repvgg_model_convert(model)
    else:
        deploy_model = model

    deploy_model.eval()

    outputs = []
    targets = []
    for i, (input, target) in enumerate(eval_loader):
        input = input.cuda()
        target = target.cuda()

        with autocast():
            output = torch.sigmoid(deploy_model(input).detach())

        outputs.append(output.cpu().detach().numpy())
        targets.append(target.cpu().detach().numpy())

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    APs, mAP, of1, cf1 = test(outputs, targets)

    return APs, mAP, of1, cf1
