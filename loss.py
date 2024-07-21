import torch
import torch.nn as nn


class Loss_bce(nn.Module):
    def __init__(self, reduction='none'):
        super(Loss_bce, self).__init__()
        self.loss_func_bce = torch.nn.BCELoss(reduction=reduction)
    
    def forward(self, sources, targets, reweight):
        sources = sources.view(-1)
        targets = targets.view(-1)
        reweight = reweight.view(-1)

        loss = self.loss_func_bce(sources, targets)*reweight
        loss = loss.mean()
    
        return loss

