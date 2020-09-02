import os.path as osp
from .market1501 import Market1501


class OccDukeMTMCreID(Market1501):
    """Proposed in paper:
    Pose-guided feature alignment for occluded person re-identification, ICCV 2019
    """
    has_ps_label = True
    im_root = 'Occluded_Duke'
    split_spec = {
        'train': {'pattern': '{}/bounding_box_train/*.jpg'.format(im_root), 'map_label': True},
        'query': {'pattern': '{}/query/*.jpg'.format(im_root), 'map_label': False},
        'gallery': {'pattern': '{}/bounding_box_test/*.jpg'.format(im_root), 'map_label': False},
    }

    def _get_ps_label_path(self, im_path):
        path = im_path.replace(self.im_root, self.im_root + '_ps_label')
        path = path.replace('.jpg', '.png')
        path = osp.join(self.root, path)
        return path
