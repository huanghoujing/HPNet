from .resnet import get_resnet
from .resnet_separate_conv5 import get_resnet as get_resnet_sep_conv5


backbone_factory = {
    'resnet18': get_resnet,
    'resnet34': get_resnet,
    'resnet50': get_resnet,
    'resnet101': get_resnet,
    'resnet152': get_resnet,
    'resnet_sep_conv5_50': get_resnet_sep_conv5,
}


def create_backbone(cfg):
    return backbone_factory[cfg.name](cfg)
