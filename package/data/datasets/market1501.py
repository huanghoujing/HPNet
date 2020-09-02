import os.path as osp
from PIL import Image
from ..dataset import Dataset
from ...utils.file import get_files_by_pattern
from ...utils.file import save_pickle


class Market1501(Dataset):
    has_ps_label = True
    im_root = 'Market-1501-v15.09.15'
    split_spec = {
            'train': {'pattern': '{}/bounding_box_train/*.jpg'.format(im_root), 'map_label': True},
            'query': {'pattern': '{}/query/*.jpg'.format(im_root), 'map_label': False},
            'gallery': {'pattern': '{}/bounding_box_test/*.jpg'.format(im_root), 'map_label': False},
    }

    def get_im(self, im_path):
        return Image.open(osp.join(self.root, im_path)).convert("RGB")

    def _get_ps_label_path(self, im_path):
        path = im_path.replace(self.im_root, self.im_root + '_ps_label')
        path = path.replace('.jpg', '.png')
        path = osp.join(self.root, path)
        return path

    @staticmethod
    def parse_im_path(im_path):
        im_name = osp.basename(im_path)
        id = -1 if im_name.startswith('-1') else int(im_name[:4])
        cam = int(im_name[4]) if im_name.startswith('-1') else int(im_name[6])
        return id, cam

    def save_split(self, spec, save_path):
        cfg = self.cfg
        im_paths = sorted(get_files_by_pattern(self.root, pattern=spec['pattern'], strip_root=True))
        assert len(im_paths) > 0, "There are {} images for split [{}] of dataset [{}]. Please place your dataset in right position."\
            .format(len(im_paths), cfg.split, self.__class__.__name__)
        ids, cams = zip(*[self.parse_im_path(p) for p in im_paths])
        # Filter out id -1 which is officially not used
        im_paths, ids, cams = zip(*[(im_path, id, cam) for im_path, id, cam in zip(im_paths, ids, cams) if id != -1])
        if spec['map_label']:
            unique_ids = sorted(list(set(ids)))
            ids2labels = dict(zip(unique_ids, range(len(unique_ids))))
            labels = [ids2labels[id] for id in ids]
        else:
            labels = ids
        samples = [{'im_path': im_path, 'label': label, 'cam': cam} for im_path, label, cam in zip(im_paths, labels, cams)]
        save_pickle(samples, save_path)
        return samples
