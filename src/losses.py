import torch
import torch.nn as nn


class CenterfaceLoss(nn.Module):
    def __init__(self, loss_weights):
        super(CenterfaceLoss, self).__init__()
        self.loss_weights = loss_weights

    def forward(self, hmp, s, off, lmk, gt_boxes):
        pass