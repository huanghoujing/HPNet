import torch
import torch.nn as nn


def init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_embedding(in_dim=None, out_dim=None):
    layers = [
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True)
    ]
    return nn.Sequential(*layers)


#################
# For Fixing BN #
#################

fix_bn_types = ['buffer', 'buffer+param']


def fix_bn(model, type):
    """Fix running_mean, running_var and optionally weight and bias.
    This should be called before EVERY forward step during training."""
    assert type in fix_bn_types
    from torch.nn.modules.batchnorm import _BatchNorm
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
            if type == 'buffer+param':
                if m.weight is not None:
                    m.weight.requires_grad = False
                    m.weight.grad = None
                if m.bias is not None:
                    m.bias.requires_grad = False
                    m.bias.grad = None


def get_bn_buffers(model):
    """Copy BN buffer values."""
    from torch.nn.modules.batchnorm import _BatchNorm
    from copy import deepcopy
    ret = []
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            ret.append(deepcopy(m.running_mean.cpu()))
            ret.append(deepcopy(m.running_var.cpu()))
    return ret


def get_bn_params(model):
    """Copy BN Parameter data."""
    from torch.nn.modules.batchnorm import _BatchNorm
    from copy import deepcopy
    ret = []
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            if (m.weight is not None) and (m.bias is not None):
                ret.append(deepcopy(m.weight.data.cpu()))
                ret.append(deepcopy(m.bias.data.cpu()))
    return ret


def assert_bn_grad_is_None(model):
    from torch.nn.modules.batchnorm import _BatchNorm
    for n, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            if (m.weight is not None) and hasattr(m.weight, 'grad'):
                assert (m.weight.grad is None), "Gradient of {}.weight is not None!".format(n)
            if (m.bias is not None) and hasattr(m.bias, 'grad'):
                assert (m.bias.grad is None), "Gradient of {}.bias is not None!".format(n)


def assert_bn_grad_is_None_or_zero(model):
    from torch.nn.modules.batchnorm import _BatchNorm
    for n, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            if (m.weight is not None) and hasattr(m.weight, 'grad'):
                assert (m.weight.grad is None) or (m.weight.grad.data.abs().sum() == 0), \
                    "Gradient of {}.weight is not None or 0!".format(n)
            if (m.bias is not None) and hasattr(m.bias, 'grad'):
                assert (m.bias.grad is None) or (m.bias.grad.data.abs().sum() == 0), \
                    "Gradient of {}.bias is not None or 0!".format(n)


def get_bn_values(model, type):
    assert type in fix_bn_types
    return get_bn_buffers(model) if type == 'buffer' else get_bn_buffers(model) + get_bn_params(model)


def are_tensor_lists_equal(list1, list2):
    assert len(list1) == len(list2), "Two lists of tensors should have same length!"
    return all([torch.sum(l1 - l2).item() == 0 for l1, l2 in zip(list1, list2)])

#################
