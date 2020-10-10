from src.losses import FocalLoss, SmoothL1Loss
import torch.nn as nn
import torch.nn.functional as F

class CenterfaceLoss(nn.Module):
    def __init__(self):
        super(CenterfaceLoss, self).__init__()
        self.foc_crit = FocalLoss()
        self.smooth_crit = SmoothL1Loss()


    def forward(self, output, batch):
        hm_loss, s_loss, off_loss = 0, 0, 0
        hm_loss += self.foc_crit(output['hm'], batch['hm'])
        s_loss += self.smooth_crit(output['scale'], batch['mask'], batch['ind'], batch['scale'])
        off_loss += self.smooth_crit(output['off'], batch['mask'], batch['ind'], batch['off'])

        return hm_loss, s_loss, off_loss