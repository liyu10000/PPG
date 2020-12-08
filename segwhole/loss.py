import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt

from config import cfg

classes = cfg.segwhole.classes


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        """
        :param input: C, H, W
        :param target: C, H, W
        """
        # self.save_for_backward(input, target)
        eps = 1.0
        inter = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * inter.float() + eps) / (union.float() + eps)
        return t 

    # # This function has only a single output, so it gets only one gradient
    # def backward(self, grad_output):
    #     input, target = self.saved_variables
    #     grad_input = grad_target = None
    #     if self.needs_input_grad[0]:
    #         grad_input = grad_output * 2 * (target * self.union - self.inter) \
    #                      / (self.union * self.union)
    #     if self.needs_input_grad[1]:
    #         grad_target = None
    #     return grad_input, grad_target


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

def bce_loss(inputs, targets):
    return nn.BCEWithLogitsLoss()(inputs, targets)

def bce_dice_loss(inputs, targets):
    return bce_loss(inputs, targets) + dice_loss(inputs, targets)
