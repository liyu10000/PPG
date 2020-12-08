import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt

from config import cfg
classes = cfg.segdefect.classes


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        """
        :param input: C, H, W
        :param target: C, H, W
        """
        # self.save_for_backward(input, target)
        eps = 1.0

        # t = 0.0
        # for i, (ip, tg) in enumerate(zip(input, target)):
        #     inter = torch.dot(ip.view(-1), tg.view(-1))
        #     union = torch.sum(ip) + torch.sum(tg) + eps
        #     t += (2 * inter.float() + eps) / (union.float() + eps)
        # return t / (i + 1)

        inter = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * inter.float() + eps) / (union.float() + eps)
        return t 


def dice_loss(inputs, targets):
    """Dice loss for batches
    :param inputs: N, C, H, W
    :param targets: N, C, H, W
    """
    if inputs.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    if classes > 1:
        inputs = F.softmax(inputs, dim=1)
    else:
        inputs = torch.sigmoid(inputs)

    for i, (input, target) in enumerate(zip(inputs, targets)):
        s += DiceCoeff().forward(input, target)

    return 1. - s / (i + 1)

def bce_loss(inputs, targets, reduction='mean'):
    return nn.BCEWithLogitsLoss(reduction=reduction)(inputs, targets)

def bce_dice_loss(inputs, targets, reduction='mean'):
    return bce_loss(inputs, targets, reduction) + dice_loss(inputs, targets)
