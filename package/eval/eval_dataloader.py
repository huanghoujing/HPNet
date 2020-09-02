from __future__ import print_function
import os.path as osp
import numpy as np
from .extract_feat import extract_dataloader_feat
from .eval_feat import eval_feat
from ..utils.file import save_pickle, load_pickle


def _print_stat(dic):
    print('=> Eval Statistics:')
    print('\tdic.keys():', dic.keys())
    print("\tdic['q_feat'].shape:", dic['q_feat'].shape)
    print("\tdic['q_label'].shape:", dic['q_label'].shape)
    print("\tdic['q_cam'].shape:", dic['q_cam'].shape)
    if 'q_visible' in dic:
        print("\tdic['q_visible'].shape:", dic['q_visible'].shape)
    print("\tdic['g_feat'].shape:", dic['g_feat'].shape)
    print("\tdic['g_label'].shape:", dic['g_label'].shape)
    print("\tdic['g_cam'].shape:", dic['g_cam'].shape)
    if 'g_visible' in dic:
        print("\tdic['g_visible'].shape:", dic['g_visible'].shape)


def eval_dataloader(model, q_loader, g_loader, cfg):
    if osp.exists(cfg.test_feat_cache_file) and cfg.use_cache:
        q_feat_dict, g_feat_dict = load_pickle(cfg.test_feat_cache_file)
    else:
        q_feat_dict = extract_dataloader_feat(model, q_loader, cfg)
        g_feat_dict = extract_dataloader_feat(model, g_loader, cfg)
        save_pickle([q_feat_dict, g_feat_dict], cfg.test_feat_cache_file)
        # if cfg.use_cache:
        #     save_pickle([q_feat_dict, g_feat_dict], cfg.test_feat_cache_file)
    dic = {
        'q_feat': q_feat_dict['feat'],
        'q_label': np.array(q_feat_dict['label']),
        'q_cam': np.array(q_feat_dict['cam']),
        'g_feat': g_feat_dict['feat'],
        'g_label': np.array(g_feat_dict['label']),
        'g_cam': np.array(g_feat_dict['cam']),
    }
    which_part = cfg.which_part
    if which_part is not None:
        pfd = cfg.pfd
        print('=> Use Part Index {} for Evaluation. Each Part Has Feature Dimension {}.'.format(which_part, pfd))
        if not isinstance(which_part, (list, tuple)):
            which_part = [which_part]
        dic['q_feat'] = np.concatenate([dic['q_feat'][:, i*pfd:i*pfd+pfd] for i in which_part], 1)
        dic['g_feat'] = np.concatenate([dic['g_feat'][:, i*pfd:i*pfd+pfd] for i in which_part], 1)
    if 'visible' in q_feat_dict:
        dic['q_visible'] = q_feat_dict['visible']
        if which_part is not None:
            dic['q_visible'] = np.stack([dic['q_visible'][:, i] for i in which_part], 1)
    if 'visible' in g_feat_dict:
        dic['g_visible'] = g_feat_dict['visible']
        if which_part is not None:
            dic['g_visible'] = np.stack([dic['g_visible'][:, i] for i in which_part], 1)
    _print_stat(dic)
    return eval_feat(dic, cfg)
