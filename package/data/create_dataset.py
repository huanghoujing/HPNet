from copy import deepcopy
from .datasets.market1501 import Market1501
from .datasets.coco import COCO
from .datasets.partial_reid import PartialREID
from .datasets.partial_ilids import PartialiLIDs
from .datasets.partial_ilids_fpr import PartialiLIDsFPR
from .datasets.occluded_reid import OccludedREID
from .datasets.occ_duke import OccDukeMTMCreID
from .combine_train_sets import CombinedTrainset


__factory = {
        'market1501': Market1501,
        'coco': COCO,
        'partial_reid': PartialREID,
        'partial_ilids': PartialiLIDs,
        'partial_ilids_fpr': PartialiLIDsFPR,
        'occluded_reid': OccludedREID,
        'occ_duke': OccDukeMTMCreID,
    }


dataset_shortcut = {
    'market1501': 'M',
    'partial_reid': 'PR',
    'partial_ilids': 'PI',
    'partial_ilids_fpr': 'PIF',
    'occluded_reid': 'O',
    'occ_duke': 'OD',
}


def create_dataset(cfg, samples=None):
    if isinstance(cfg.name, (list, tuple)):
        datasets = []
        for name, split in zip(cfg.name, cfg.split):
            cfg_tmp = deepcopy(cfg)
            cfg_tmp.name = name
            cfg_tmp.split = split
            datasets.append(__factory[name](cfg_tmp, samples=samples))
        if cfg.combine:
            return CombinedTrainset(datasets)
        else:
            return datasets
    else:
        return __factory[cfg.name](cfg, samples=samples)
