import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt

classes = 1

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


def parse_train_log(train_log):
    epochs = []
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    with open(train_log, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('Starting') and 'train' in line:  # train starts
                tokens = line.strip().split()
                epoch = int(tokens[2])
                epochs.append(epoch)  # number of epochs
                i += 1
                line = lines[i]
                tokens = line.strip().split()
                loss = float(tokens[1])
                iou = float(tokens[4])
                train_losses.append(loss)  # loss
                train_ious.append(iou)  # iou
            if line.startswith('Starting') and 'val' in line:  # val starts
                i += 1
                line = lines[i]
                tokens = line.strip().split()
                loss = float(tokens[1])
                iou = float(tokens[4])
                val_losses.append(loss)  # loss
                val_ious.append(iou)  # iou
            i += 1
    return epochs, train_losses, val_losses, train_ious, val_ious


def plot_train_log(epochs, train_metric, val_metric, save_name=None):
    # plot trend
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.grid()
    ax.scatter(epochs, train_metric, marker='.', color='black')
    ax.scatter(epochs, val_metric, marker='s', color='red')
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)


if __name__ == '__main__':
    train_log = '../datadefects/exps/exp2/bce_low.log'
    epochs, train_losses, val_losses, train_ious, val_ious = parse_train_log(train_log)
    plot_train_log(epochs, train_losses, val_losses, '../datadefects/exps/exp2/bce_low_loss.png')
    plot_train_log(epochs, train_ious, val_ious, '../datadefects/exps/exp2/bce_low_iou.png')