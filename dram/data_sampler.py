from torch.utils.data.sampler import Sampler
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight


class LobeChunkCLESampler(Sampler):

    def __init__(self, logger, data_source, batch_size, balance_label_count=None):
        super(LobeChunkCLESampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.balance_label_count = balance_label_count
        self.logger = logger
        # compute lobe-wise ctss
        ctsses = [int(float(self.data_source.meta[lobe_wise_uid]['cle']))
                  for lobe_wise_uid in self.data_source.uids]
        indices = list(range(len(ctsses)))
        self.logger.info("total {} instances to sample from.".format(len(indices)))
        ctss_labels, ctss_counts = np.unique(ctsses, return_counts=True)
        cws = class_weight.compute_class_weight('balanced',
                                                np.unique(ctsses),
                                                ctsses)
        cws = np.clip(cws / cws.sum(), a_min=0.2, a_max=0.8)
        self.class_weights = list(cws)
        self.ctss_frequency_map = {cl: cw / np.sum(ctss_counts)
                                   for cl, cw in zip(ctss_labels, ctss_counts)}
        for ctss_type in range(0, 6):
            if ctss_type not in ctss_labels:
                self.class_weights.insert(ctss_type, max(self.class_weights))
                self.ctss_frequency_map[ctss_type] = 1e-5
        # self.ctss_frequency_map = { cl: cw for cl, cw in zip(ctss_labels, class_weights)}
        self.logger.info("sampled CTSS distribution {}-{}.".format(ctss_labels, ctss_counts))
        self.logger.info("ctss_frequency_map {}.".format(self.ctss_frequency_map))
        if self.balance_label_count is None:
            self.balance_label_count = int(np.median(ctss_counts))
        sampling_ctsses = []
        sampling_indices = []
        for al in ctss_labels:
            al_locs = np.where(np.asarray(ctsses) == al)[0]
            s_indices = np.random.choice(al_locs, self.balance_label_count)
            sampling_ctsses.extend([ctsses[x] for x in s_indices])
            sampling_indices.extend(s_indices)

        self.logger.info("total {} instances to sample from after enforcing counts {} for each label."
                         .format(len(sampling_indices), self.balance_label_count))
        X = np.zeros((len(sampling_ctsses), 1))
        y = np.asarray(sampling_ctsses)
        test_size = max(int(self.batch_size * 2), len(np.unique(sampling_ctsses)))
        n_splits = len(sampling_ctsses) // test_size
        s = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        stratified_indices = np.asarray([test_index for _, test_index in s.split(X, y)]).flatten().tolist()

        self.indices = [sampling_indices[i] for i in stratified_indices]
        sampled_ctsss = [ctsses[x] for x in self.indices]
        k = min(20, len(sampled_ctsss))
        self.logger.info("sampled ctss {} at the first {} items.".format(sampled_ctsss[:k], k))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class LobeChunkCTSSSampler(Sampler):

    def __init__(self, logger, data_source, batch_size, balance_label_count=None):
        super(LobeChunkCTSSSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.balance_label_count = balance_label_count
        self.logger = logger
        # compute lobe-wise ctss
        self.ctsses = [int(float(self.data_source.all_metas[lobe_wise_uid]['ctss']))
                  for lobe_wise_uid in self.data_source.uids]
        self.logger.info("total {} instances to sample from.".format(len(self.ctsses)))
        self.ctss_labels, self.ctss_counts = np.unique(self.ctsses, return_counts=True)
        self.class_weights = list(class_weight.compute_class_weight('balanced',
                                                 np.unique(self.ctsses),
                                                 self.ctsses))
        self.ctss_frequency_map = {cl: cw / np.sum(self.ctss_counts)
                                   for cl, cw in zip(self.ctss_labels, self.ctss_counts)}
        for ctss_type in range(0, 6):
            if ctss_type not in self.ctss_labels:
                self.class_weights.insert(ctss_type, max(self.class_weights))
                self.ctss_frequency_map[ctss_type] = 1e-5
        # self.ctss_frequency_map = { cl: cw for cl, cw in zip(ctss_labels, class_weights)}
        self.logger.info("sampled CTSS distribution {}-{}.".format(self.ctss_labels, self.ctss_counts))
        self.logger.info("ctss_frequency_map {}.".format(self.ctss_frequency_map))
        if self.balance_label_count is None:
            self.balance_label_count = int(np.median(self.ctss_counts))

        self.total_n = self.balance_label_count * len(self.ctss_labels)
        self.logger.info("sampling total {} samples.".format(self.total_n))

        self.grouped_data = {int(label): np.where(self.ctsses == label)[0] for label in self.ctss_labels}

    def __iter__(self):
        all_sampled_indices = []
        for n in range(self.total_n):
            sampled_label = np.random.choice(self.ctss_labels, 1)[0]
            group_indices = self.grouped_data[sampled_label]

            sample_idx = np.random.choice(group_indices, 1)[0]
            all_sampled_indices.append(sample_idx)
        return iter(all_sampled_indices)

    def __len__(self):
        return self.total_n