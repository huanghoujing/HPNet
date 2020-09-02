from __future__ import print_function
from itertools import chain
import torch
import torch.nn as nn
from .base_model import BaseModel
from .backbone import create_backbone
from .global_pool import GlobalPool
from ..utils.model import init_classifier
from ..eval.torch_distance import normalize


class GPBaseline(BaseModel):
    def __init__(self, cfg):
        super(GPBaseline, self).__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg.backbone)
        self.pool = GlobalPool(cfg)
        self.bn = nn.BatchNorm1d(self.backbone.out_c)
        self.bn.bias.requires_grad_(False)
        if hasattr(cfg, 'num_classes') and cfg.num_classes > 0:
            self.cls = nn.Linear(self.backbone.out_c, self.cfg.num_classes)
            self.cls.apply(init_classifier)
        print('Model Structure:\n{}'.format(self))

    def get_ft_and_new_params(self):
        ft_modules, new_modules = self.get_ft_and_new_modules()
        ft_params = list(chain.from_iterable([list(m.parameters()) for m in ft_modules]))
        new_params = list(chain.from_iterable([list(m.parameters()) for m in new_modules]))
        return ft_params, new_params

    def get_ft_and_new_modules(self):
        ft_modules = [self.backbone]
        new_modules = [self.bn]
        if hasattr(self, 'cls'):
            new_modules += [self.cls]
        return ft_modules, new_modules

    def set_train_mode(self):
        self.train()

    def backbone_forward(self, in_dict):
        return self.backbone(in_dict['im'])

    def reid_forward(self, in_dict):
        # print('=====> in_dict.keys() entering reid_forward():', in_dict.keys())
        pool_out_dict = self.pool(in_dict)
        feat_list = [self.bn(pool_out_dict['feat_list'][0])]
        out_dict = {
            'feat_list': feat_list,
            'tri_loss_em_list': pool_out_dict['feat_list'],
        }
        if hasattr(self, 'cls'):
            out_dict['logits_list'] = [self.cls(feat_list[0])]
        return out_dict

    def forward(self, in_dict, forward_type='reid'):
        backbone_out = self.backbone_forward(in_dict)
        in_dict['feat'] = [backbone_out]
        out_dict = self.reid_forward(in_dict)
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
        return ret_dict