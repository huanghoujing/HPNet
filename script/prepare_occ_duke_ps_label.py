import sys
sys.path.insert(0, '.')
import os.path as osp
from package.utils.file import get_files_by_pattern, copy_to


# You have to generate Occluded_Duke following https://github.com/lightas/Occluded-DukeMTMC-Dataset
# The result should have following structure
# - Occluded_Duke
#   - bounding_box_train
#   - query
#   - bounding_box_test
im_root = 'occ_duke/Occluded_Duke'

# This can be downloaded from Baidu Cloud (https://pan.baidu.com/s/1Mm2gWO-Xg3wiyCd6SEAWaA)
# or Google Drive (https://drive.google.com/open?id=1BARSoobjTAPeOSOM-HnGzlOYTj1l9-Qs)
ori_ps_label_dir = 'occ_duke/DukeMTMC-reID_ps_label'

# The part segmentation label for Occluded_Duke would be saved here
save_ps_label_dir = 'occ_duke/Occluded_Duke_ps_label'

dirs = ['bounding_box_train', 'query', 'bounding_box_test']

ori_ps_label_paths = {}
for dir in dirs:
    ori_ps_label_paths[dir] = get_files_by_pattern(osp.join(ori_ps_label_dir, dir), pattern='*.png', strip_root=True)

def prepare(split):
    im_dir = osp.join(im_root, split)
    paths = get_files_by_pattern(im_dir, pattern='*.jpg', strip_root=True)
    paths = [p.replace('.jpg', '.png') for p in paths]
    for path in paths:
        found = False
        for dir in dirs:
            if path in ori_ps_label_paths[dir]:
                copy_to(osp.join(ori_ps_label_dir, dir, path), osp.join(save_ps_label_dir, split, path))
                found = True
                break
        if not found:
            raise RuntimeError('{} could not be found!'.format(path))
    print('Done for split {}'.format(split))


for dir in dirs:
    prepare(dir)