import torch
import torchvision.transforms.functional as F
from copy import deepcopy
import random
import numpy as np
from PIL import Image
import cv2
from .random_erasing import RandomErasing, RandomErasingWithPS

"""We expect a list `cfg.transform_list`. The types specified in this list 
will be applied sequentially. Each type name corresponds to a function name in 
this file, so you have to implement the function w.r.t. your custom type. 
The function head should be `FUNC_NAME(in_dict, cfg)`, and it should modify `in_dict`
in place.
The transform list allows us to apply optional transforms in any order, while custom
functions allow us to perform sync transformation for images and all labels.

Examples:
    transform_list = ['hflip', 'resize']
    transform_list = ['hflip', 'random_crop', 'resize']
    transform_list = ['hflip', 'resize', 'random_erase']
"""


def hflip(in_dict, cfg):
    if np.random.random() < 0.5:
        in_dict['im'] = F.hflip(in_dict['im'])
        if 'ps_label' in in_dict:
            in_dict['ps_label'] = F.hflip(in_dict['ps_label'])


def resize_3d_np_array(maps, resize_h_w, interpolation):
    """maps: np array with shape [C, H, W], dtype is not restricted"""
    return np.stack([cv2.resize(m, tuple(resize_h_w[::-1]), interpolation=interpolation) for m in maps])


def resize(in_dict, cfg):
    in_dict['im'] = Image.fromarray(cv2.resize(np.array(in_dict['im']), tuple(cfg.im.h_w[::-1]), interpolation=cv2.INTER_LINEAR))
    if 'ps_label' in in_dict:
        in_dict['ps_label_im_size'] = deepcopy(in_dict['ps_label'])
        in_dict['ps_label'] = F.resize(in_dict['ps_label'], tuple(cfg.ps_label.h_w), Image.NEAREST)  # TODO: TMP
        # print("in_dict['ps_label'].shape:", np.array(in_dict['ps_label']).shape)
        in_dict['ps_label_im_size'] = F.resize(in_dict['ps_label_im_size'], tuple(cfg.im.h_w), Image.NEAREST)  # TODO: TMP
        # in_dict['ps_label'] = Image.fromarray(cv2.resize(np.array(in_dict['ps_label']), tuple(cfg.ps_label.h_w[::-1]), cv2.INTER_NEAREST), mode='L')


# If called, it should be after `to_tensor`
def random_erase(in_dict, cfg):
    if 'ps_label' in in_dict:
        in_dict['im'], in_dict['ps_label'] = RandomErasingWithPS(probability=0.5, sl=0.1, sh=0.2, mean=[0, 0, 0])(in_dict['im'], in_dict['ps_label'])
    else:
        in_dict['im'] = RandomErasing(probability=0.5, sh=0.2, mean=[0, 0, 0])(in_dict['im'])


def random_crop(in_dict, cfg):
    def get_params(min_keep_ratio=0.85):
        x_keep_ratio = random.uniform(min_keep_ratio, 1)
        y_keep_ratio = random.uniform(min_keep_ratio, 1)
        x1 = random.uniform(0, 1 - x_keep_ratio)
        y1 = random.uniform(0, 1 - y_keep_ratio)
        x2 = x1 + x_keep_ratio
        y2 = y1 + y_keep_ratio
        return x1, y1, x2, y2

    def crop_kpt(kpt, x1, y1, x2, y2):
        """Borders x2 and y2 are exclusive.
        kpt: np array with shape [N, 3]
        """
        kpt[:, 2][kpt[:, 0] < x1] = 0
        kpt[:, 2][kpt[:, 0] >= x2] = 0
        kpt[:, 2][kpt[:, 1] < y1] = 0
        kpt[:, 2][kpt[:, 1] >= y2] = 0
        kpt[:, 0] = kpt[:, 0] - x1
        kpt[:, 1] = kpt[:, 1] - y1
        return kpt

    x1, y1, x2, y2 = get_params()
    im_w, im_h = in_dict['im'].size
    im_x1, im_y1, im_x2, im_y2 = int(im_w * x1), int(im_h * y1), int(im_w * x2), int(im_h * y2)
    in_dict['im'] = in_dict['im'].crop((im_x1, im_y1, im_x2, im_y2))
    if 'kpt' in in_dict:
        in_dict['kpt'] = crop_kpt(in_dict['kpt'], im_x1, im_y1, im_x2, im_y2)
        in_dict['kpt_im_h_w'] = (im_y2 - im_y1, im_x2 - im_x1)
    if 'ps_label' in in_dict:
        ps_w, ps_h = in_dict['ps_label'].size
        ps_x1, ps_y1, ps_x2, ps_y2 = int(ps_w * x1), int(ps_h * y1), int(ps_w * x2), int(ps_h * y2)
        in_dict['ps_label'] = in_dict['ps_label'].crop((ps_x1, ps_y1, ps_x2, ps_y2))
    return in_dict

############
# Fuse Parts

# Original 7 parts
#   'background': 0,
#   'head': 1,
#   'torso': 2,
#   'upper_arm': 3,
#   'lower_arm': 4,
#   'upper_leg': 5,
#   'lower_leg': 6,
#   'foot': 7,
fuse_mapping = {
    '4parts': [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2), (5, 3), (6, 4), (7, 4)],
    '2parts': [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2), (6, 2), (7, 2)],
    'fg': [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
}

def fuse_parts(in_dict, cfg):
    ori_ps_label = np.array(in_dict['ps_label'])
    ps_label = ori_ps_label.copy()
    for m in fuse_mapping[cfg.transform.fuse_type]:
        ps_label[ori_ps_label == m[0]] = m[1]
    in_dict['ps_label'] = Image.fromarray(ps_label, mode='L')

################
# Part Occlusion
# This should be called at the end of transformation
# This is only used for ps_pool model, not for pa_pool

def _occlude_one_part(in_dict, idx):
    in_dict['im'][:, in_dict['ps_label_im_size'] == idx] = 0
    in_dict['ps_label'][in_dict['ps_label'] == idx] = 0
    in_dict['ps_label_im_size'][in_dict['ps_label_im_size'] == idx] = 0

# def _occlude_one_part(in_dict, idx):
#     if np.random.random() > 0.5:
#         in_dict['im'][:, in_dict['ps_label_im_size'] == idx] = in_dict['im'][:, in_dict['ps_label_im_size'] == idx].normal_(0, 1)  # TODO: is this pytorch's bug?
#     else:
#         in_dict['im'][:, in_dict['ps_label_im_size'] == idx] = 0
#     in_dict['ps_label'][in_dict['ps_label'] == idx] = 0
#     in_dict['ps_label_im_size'][in_dict['ps_label_im_size'] == idx] = 0

def occlude_part(in_dict, cfg):
    if np.random.random() > 0.5:
    # if np.random.random() > 0.25:
        return
    nparts = cfg.transform.num_parts
    indices = list(range(1, nparts+1))
    for _ in range(np.random.randint(cfg.transform.max_occlude_parts)+1):
        idx = np.random.choice(indices)
        indices.remove(idx)
        _occlude_one_part(in_dict, idx)

# def occlude_part(in_dict, cfg):
#     if np.random.random() > 0.5:
#     # if np.random.random() > 0.25:
#         return
#     nparts = cfg.transform.num_parts
#     indices = list(range(1, nparts+1))
#     for _ in range(np.random.randint(cfg.transform.max_occlude_parts)+1):
#         idx = np.random.choice(indices, p=[0.1, 0.4, 0.4, 0.1])
#         indices.remove(idx)
#         _occlude_one_part(in_dict, idx)

##########################
# Enhancement Augmentation

from PIL import Image, ImageFilter

def blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

all_augs = [
    (blur, [1,2]),
    (F.adjust_brightness, [0.6, 1.4]),
    (F.adjust_contrast, [0.6, 1.4]),
    (F.adjust_saturation, [0.6, 1.4]),
    (F.adjust_gamma, [0.6, 1.4]),
]

def aug_im(im, func, param, prob):
    if np.random.rand() < prob:
        im = func(im, param)
    return im

def comb_aug_im(im, all_augs):
    if np.random.rand() < 0.5:
        return im
    random.shuffle(all_augs)
    for func, ran in all_augs:
        param = np.random.choice(ran) if func is blur else np.random.uniform(ran[0], ran[1])
        im = aug_im(im, func, param, 0.7)
    return im

def enhance_aug(in_dict, cfg):
    in_dict['im'] = comb_aug_im(in_dict['im'], all_augs)

##########################


def to_tensor(in_dict, cfg):
    in_dict['im'] = F.to_tensor(in_dict['im'])
    in_dict['im'] = F.normalize(in_dict['im'], cfg.im.mean, cfg.im.std)
    if 'ps_label' in in_dict:
        in_dict['ps_label'] = torch.from_numpy(np.array(in_dict['ps_label'])).long()
    if 'ps_label_im_size' in in_dict:
        in_dict['ps_label_im_size'] = torch.from_numpy(np.array(in_dict['ps_label_im_size'])).long()


def transform(in_dict, cfg):
    if 'fuse_parts' in cfg.transform_list:
        fuse_parts(in_dict, cfg)
    for t in cfg.transform_list:
        if t not in ['random_erase', 'occlude_part', 'fuse_parts']:
            eval('{}(in_dict, cfg)'.format(t))
    to_tensor(in_dict, cfg)
    if 'random_erase' in cfg.transform_list:
        random_erase(in_dict, cfg)
    if 'occlude_part' in cfg.transform_list:
        occlude_part(in_dict, cfg)
    return in_dict
