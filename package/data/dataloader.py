from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import SequentialSampler, RandomSampler
from .create_dataset import create_dataset
from .random_identity_sampler import RandomIdentitySampler


def create_dataloader(cfg, dataset_cfg, samples=None):
    def create_a_loader(dataset):
        if cfg.batch_type == 'seq':
            sampler = SequentialSampler(dataset)
        elif cfg.batch_type == 'random':
            sampler = RandomSampler(dataset)
        elif cfg.batch_type == 'pk':
            sampler = RandomIdentitySampler(dataset, cfg.pk.k)
        else:
            raise NotImplementedError
        loader = TorchDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=cfg.drop_last,
        )
        return loader

    dataset = create_dataset(dataset_cfg, samples=samples)
    if isinstance(dataset, (list, tuple)):
        loader = [create_a_loader(d) for d in dataset]
    else:
        loader = create_a_loader(dataset)
    return loader
