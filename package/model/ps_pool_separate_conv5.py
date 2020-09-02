import torch
import torch.nn.functional as F


def max_pool(in_dict):
    """Implement `local max pooling` as `masking + global max pooling`.
    Args:
        feat: pytorch tensor, with shape [N, C, H, W]
        mask: pytorch tensor, with shape [N, pC, pH, pW]
    Returns:
        feat_list: a list (length = pC) of pytorch tensors with shape [N, C]
        visible: pytorch tensor with shape [N, pC]
    NOTE:
        The implementation of `masking + global max pooling` is only equivalent
        to `local max pooling` when feature values are non-negative, which holds
        for ResNet that has ReLU as final operation of all blocks.
    """
    assert len(in_dict['feat']) == len(in_dict['ps_pred'])
    N = in_dict['feat'][0].shape[0]
    num_parts = len(in_dict['ps_pred'])
    feat_list = []
    visible_list = []
    for i in range(num_parts):
        feat = in_dict['feat'][i]
        # [N, 1, pH, pW]
        m = (in_dict['ps_pred'][i] > 0.5).float()
        visible_list.append((m.sum(-1).sum(-1) > 0).float())
        # [N, C, pH, pW]
        m = m.expand_as(feat)
        # [N, C]
        local_feat = F.adaptive_max_pool2d(feat * m, 1).view(N, -1)
        feat_list.append(local_feat)
    # [N, pC]
    visible = torch.cat(visible_list, 1)
    out_dict = {'feat_list': feat_list, 'visible': visible}
    return out_dict


class PSPoolSConv5(object):
    def __init__(self, cfg):
        if cfg.max_or_avg == 'max':
            self.pool = max_pool
        else:
            raise ValueError('Invalid Pool Type {}'.format(cfg.max_or_avg))

    def __call__(self, *args, **kwargs):
        return self.pool(*args, **kwargs)
