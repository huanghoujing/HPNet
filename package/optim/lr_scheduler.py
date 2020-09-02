import math
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

"""The get_lr method of _LRScheduler can be interpreted as a math function that 
takes parameter `self.last_epoch` and returns the current lr.
Note that `epoch` in pytorch lr scheduler is only local naming, which in fact means
one time of [calling **LR.step()] instead of [going over the whole dataset].
"""


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, warmup_epochs, warmup_begin_lrs, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.warmup_begin_lrs = warmup_begin_lrs
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lrs = [begin_lr + 1. * self.last_epoch * (final_lr - begin_lr) / self.warmup_epochs
                   for begin_lr, final_lr in zip(self.warmup_begin_lrs, self.base_lrs)]
        else:
            lrs = [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs]
        return lrs


class WarmupCosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, warmup_epochs, warmup_begin_lrs, T_max, eta_min=0, last_epoch=-1):
        self.warmup_begin_lrs = warmup_begin_lrs
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lrs = [begin_lr + 1. * self.last_epoch * (final_lr - begin_lr) / self.warmup_epochs
                   for begin_lr, final_lr in zip(self.warmup_begin_lrs, self.base_lrs)]
        else:
            lrs = [self.eta_min + (base_lr - self.eta_min) *
                   (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                   for base_lr in self.base_lrs]
        return lrs
