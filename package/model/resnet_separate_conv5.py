from __future__ import print_function
import os.path as osp
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import namedtuple
from ..utils.torch_utils import load_state_dict


NUM_PARTS = 4
model_urls = {
    'resnet_sep_conv5_50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_list = nn.ModuleList([self._make_layer(block, 512, layers[3], stride=cfg.last_conv_stride, restore_inplanes=True) for _ in range(NUM_PARTS)])
        self.out_c = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, restore_inplanes=False):
        inplanes_old = self.inplanes
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # Only used for duplicating a stage.
        if restore_inplanes:
            self.inplanes = inplanes_old

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = [layer4(x) for layer4 in self.layer4_list]

        return x


ArchCfg = namedtuple('ArchCfg', ['block', 'layers'])
arch_dict = {
    'resnet_sep_conv5_50': ArchCfg(Bottleneck, [3, 4, 6, 3]),
}


def duplicate_weight(state_dict, prefix, new_prefix):
    """Duplicate parameters in state_dict."""
    ret_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            ret_dict[new_prefix + k[len(prefix):]] = v
    ret_dict.update(state_dict)
    return ret_dict


def get_resnet(cfg):
    model = ResNet(arch_dict[cfg.name].block, arch_dict[cfg.name].layers, cfg)
    if cfg.pretrained:
        state_dict = model_zoo.load_url(model_urls[cfg.name], model_dir=cfg.pretrained_model_dir)
        for i in range(NUM_PARTS):
            state_dict = duplicate_weight(state_dict, 'layer4.', 'layer4_list.{}.'.format(i))
        state_dict = {k:v for k, v in state_dict.items() if not k.startswith('layer4.')}
        load_state_dict(model, state_dict)
        print('=> Loaded ImageNet Model: {}'.format(osp.join(cfg.pretrained_model_dir, osp.basename(model_urls[cfg.name]))))
    return model
