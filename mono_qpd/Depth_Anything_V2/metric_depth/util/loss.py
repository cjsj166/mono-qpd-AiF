import torch
from torch import nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        
        diff = torch.abs(pred[valid_mask] - target[valid_mask])
        loss = diff.mean()

        return loss