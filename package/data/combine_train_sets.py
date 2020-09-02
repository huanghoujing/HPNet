from torch.utils.data import ConcatDataset


class CombinedTrainset(ConcatDataset):
    def __init__(self, datasets):
        super(CombinedTrainset, self).__init__(datasets)
        print(self.summary)

    @property
    def num_ids(self):
        return sum([d.num_ids for d in self.datasets])

    @property
    def num_cams(self):
        return sum([d.num_cams for d in self.datasets])

    @property
    def summary(self):
        summary = ['=' * 25]
        summary += [self.__class__.__name__]
        summary += ['=' * 25]
        summary += ['datasets: {}'.format(', '.join([d.cfg.name for d in self.datasets]))]
        summary += ['  splits: {}'.format(', '.join([d.cfg.split for d in self.datasets]))]
        summary += ['# images: {} = {}'.format(' + '.join([str(len(d)) for d in self.datasets]), len(self))]
        summary += ['   # ids: {} = {}'.format(' + '.join([str(d.num_ids) for d in self.datasets]), self.num_ids)]
        summary += ['  # cams: {} = {}'.format(' + '.join([str(d.num_cams) for d in self.datasets]), self.num_cams)]
        summary = '\n'.join(summary) + '\n'
        return summary
