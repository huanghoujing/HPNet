from __future__ import print_function
from itertools import chain
import torch
import torch.nn as nn
from .base_model import BaseModel
from .backbone import create_backbone
from .ps_pool_separate_conv5 import PSPoolSConv5
from ..utils.model import init_classifier
from ..eval.torch_distance import normalize


class PartSegHead(nn.Module):
    def __init__(self, cfg):
        super(PartSegHead, self).__init__()
        self.mid_conv = nn.Conv2d(in_channels=cfg.in_c, out_channels=cfg.mid_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg.mid_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=cfg.mid_c, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        nn.init.normal_(self.mid_conv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.sigmoid(self.conv(self.relu(self.bn(self.mid_conv(x)))))
        return x


class HPNet(BaseModel):
    def __init__(self, cfg):
        super(HPNet, self).__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg.backbone)
        self.pool = PSPoolSConv5(cfg)
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(self.backbone.out_c) for _ in range(cfg.num_parts)])
        for bn in self.bn_list:
            bn.bias.requires_grad_(False)
        if hasattr(cfg, 'num_classes') and cfg.num_classes > 0:
            self.create_cls_list()
        if cfg.use_ps:
            cfg.ps_head.in_c = self.backbone.out_c
            self.ps_head_list = nn.ModuleList([PartSegHead(cfg.ps_head) for _ in range(cfg.num_parts)])
        print('Model Structure:\n{}'.format(self))

    def create_cls_list(self):
        self.cls_list = nn.ModuleList([nn.Linear(self.backbone.out_c, self.cfg.num_classes) for _ in range(self.cfg.num_parts)])
        self.cls_list.apply(init_classifier)

    def get_ft_and_new_params(self):
        ft_modules, new_modules = self.get_ft_and_new_modules()
        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules]))
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules]))
        return ft_params, new_params

    def get_ft_and_new_modules(self):
        ft_modules = [self.backbone]
        new_modules = [self.bn_list]
        if hasattr(self, 'cls_list'):
            new_modules += [self.cls_list]
        if hasattr(self, 'ps_head_list'):
            new_modules += [self.ps_head_list]
        return ft_modules, new_modules

    def set_train_mode(self):
        self.train()

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict['im'])

    def reid_forward(self, in_dict):
        # print('=====> in_dict.keys() entering reid_forward():', in_dict.keys())
        pool_out_dict = self.pool(in_dict)
        feat_list = [em(f) for em, f in zip(self.bn_list, pool_out_dict['feat_list'])]
        out_dict = {
            'feat_list': feat_list,
            'tri_loss_em_list': pool_out_dict['feat_list'],
        }
        if hasattr(self, 'cls_list'):
            logits_list = [cls(f) for cls, f in zip(self.cls_list, feat_list)]
            out_dict['logits_list'] = logits_list
        if 'visible' in pool_out_dict:
            out_dict['visible'] = pool_out_dict['visible']
        return out_dict

    def ps_forward(self, in_dict):
        return [ps_head(feat) for ps_head, feat in zip(self.ps_head_list, in_dict['feat'])]

    def forward(self, in_dict, forward_type='reid'):
        backbone_out = self.backbone_forward(in_dict)
        in_dict['feat'] = [backbone_out for _ in range(self.cfg.num_parts)] if self.cfg.share_conv5 else backbone_out
        out_dict = {}
        forward_type = forward_type.split('-')
        if 'ps' in forward_type:
            in_dict['ps_pred'] = out_dict['ps_pred'] = self.ps_forward(in_dict)
        if 'reid' in forward_type:
            out_dict.update(self.reid_forward(in_dict))
        return out_dict

    def extract_feat(self, in_dict):
        self.eval()
        with torch.no_grad():
            out_dict = self.forward(in_dict, forward_type=self.cfg.eval_forward_type)
            out_dict['feat_list'] = [normalize(f) for f in out_dict['feat_list']]
            feat = torch.cat(out_dict['feat_list'], 1)
            feat = feat.cpu().numpy()
            ret_dict = {
                'im_path': in_dict['im_path'],
                'feat': feat,
            }
            if 'label' in in_dict:
                ret_dict['label'] = in_dict['label'].cpu().numpy() if isinstance(in_dict['label'], torch.Tensor) else in_dict['label']
            if 'cam' in in_dict:
                ret_dict['cam'] = in_dict['cam'].cpu().numpy() if isinstance(in_dict['cam'], torch.Tensor) else in_dict['cam']
            if 'visible' in out_dict:
                ret_dict['visible'] = out_dict['visible'].cpu().numpy()
        return ret_dict