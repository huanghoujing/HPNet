from __future__ import print_function
import os.path as osp
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset as TorchDataset
from .transform import transform


class Dataset(TorchDataset):
    """Args:
        samples: None or a list of dicts; samples[i] has key 'im_path' and optional 'label', 'cam'.
    """
    has_ps_label = None
    im_root = None
    split_spec = None

    def __init__(self, cfg, samples=None):
        self.cfg = deepcopy(cfg)
        self.root = osp.join(cfg.root, cfg.name)
        if samples is None:
            self.samples = self.load_split()
        else:
            self.samples = samples
            cfg.split = 'None'
        print(self.summary)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cfg = self.cfg
        # Deepcopy to inherit all meta items
        sample = deepcopy(self.samples[index])
        im_path = sample['im_path']
        sample['im'] = self.get_im(im_path)
        if self.has_ps_label and cfg.use_ps_label:
            sample['ps_label'] = self.get_ps_label(im_path)
        transform(sample, cfg)
        return sample

    def save_split(self, spec, save_path):
        raise NotImplementedError

    def load_split(self):
        cfg = self.cfg
        save_path = osp.join(self.root, cfg.split + '.pkl')
        return self.save_split(self.split_spec[cfg.split], save_path)

    def get_im(self, im_path):
        return Image.open(osp.join(self.root, im_path)).convert("RGB")

    def _get_ps_label_path(self, im_path):
        raise NotImplementedError

    def get_ps_label(self, im_path):
        ps_label = Image.open(self._get_ps_label_path(im_path))
        return ps_label

    # Use property (instead of setting it in self.__init__) in case self.samples is changed after initialization.
    @property
    def num_samples(self):
        return len(self.samples)

    @property
    def num_ids(self):
        return len(set([s['label'] for s in self.samples])) if 'label' in self.samples[0] else -1

    @property
    def num_cams(self):
        return len(set([s['cam'] for s in self.samples])) if 'cam' in self.samples[0] else -1

    @property
    def summary(self):
        summary = ['=' * 25]
        summary += [self.__class__.__name__]
        summary += ['=' * 25]
        summary += ['   split: {}'.format(self.cfg.split)]
        summary += ['# images: {}'.format(self.num_samples)]
        summary += ['   # ids: {}'.format(self.num_ids)]
        summary += ['  # cams: {}'.format(self.num_cams)]
        summary = '\n'.join(summary) + '\n'
        return summary
