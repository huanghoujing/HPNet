from __future__ import print_function
from tqdm import tqdm
import torch
from ..utils.misc import concat_dict_list
from ..utils.torch_utils import recursive_to_device


def extract_dataloader_feat(model, loader, cfg):
    model.eval()
    dict_list = []
    for batch in tqdm(loader, desc='Extract Feature', miniters=20, ncols=120, unit=' batches'):
        batch = recursive_to_device(batch, cfg.device)
        with torch.no_grad():
            feat_dict = model.extract_feat(batch)
        dict_list.append(feat_dict)
    ret_dict = concat_dict_list(dict_list)
    return ret_dict
