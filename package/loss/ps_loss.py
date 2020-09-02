from __future__ import print_function
import torch
import torch.nn.functional as F
from .loss import Loss
from ..utils.meter import RecentAverageMeter as Meter


def py_sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25):
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
    # loss = loss.mean()
    loss = loss.sum() / target.sum()
    return loss


class PSLoss(Loss):
    def __init__(self, cfg, tb_writer=None):
        super(PSLoss, self).__init__(cfg, tb_writer=tb_writer)
        # self.criterion = torch.nn.BCELoss()
        self.criterion = py_sigmoid_focal_loss

    # TODO: Pytorch newer versions support high-dimension CrossEntropyLoss, so no need to reshape pred and label.
    def __call__(self, batch, pred, step=0, **kwargs):
        cfg = self.cfg

        # Calculation
        ps_pred = pred['ps_pred']
        ps_label = batch['ps_label']
        num_parts = len(ps_pred)
        N, C, H, W = ps_pred[0].size()
        assert ps_label.size() == (N, H, W), "ps_label.size() is {}, ps_pred[0].size() is {}".format(ps_label.size(), ps_pred[0].size())
        loss = 0
        for i in range(num_parts):
            label = (ps_label == i+1).float()  # N, H, W
            loss += self.criterion(ps_pred[i].squeeze(1), label)

        # Meter
        if cfg.name not in self.meter_dict:
            self.meter_dict[cfg.name] = Meter(name=cfg.name)
        self.meter_dict[cfg.name].update(loss.item())

        # Tensorboard
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(cfg.name, {cfg.name: self.meter_dict[cfg.name].avg}, step)

        # Scale by loss weight
        loss *= cfg.weight

        return {'loss': loss}
