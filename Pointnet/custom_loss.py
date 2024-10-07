import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# special loss for segmentation Focal Loss + Dice Loss
class PointNetSegLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, size_average=True, dice=False):
        super(PointNetSegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.dice = dice

        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, (list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)

    def forward(self, predictions, targets, pred_choice=None):
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)

        predictions = predictions.contiguous().view(-1, predictions.size(2))
        pn = F.softmax(predictions, dim=1)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        loss = ((1 - pn) ** self.gamma * ce_loss)
        if self.size_average: loss = loss.mean()
        else: loss = loss.sum()

        if self.dice: return loss + self.dice_loss(targets, pred_choice, eps=1)
        else: return loss

    @staticmethod
    def dice_loss(predictions, targets, eps=1):
        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        cats = torch.unique(targets)

        top, bot = 0, 0
        for c in cats:
            locs = targets == c
            y_tru = targets[locs]
            y_hat = predictions[locs]

            top += torch.sum(y_hat == y_tru)
            bot += len(y_tru) + len(y_hat)

        return 1 - 2 * ((top + eps) / (bot + eps))
