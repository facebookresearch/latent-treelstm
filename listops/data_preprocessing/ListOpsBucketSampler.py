# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from collections import defaultdict


class ListOpsBucketSampler:
    random_seed = 42

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.data = defaultdict(list)
        data = dataset.data
        for i, e in enumerate(data):
            self.data[len(e["tokens"])].append(i)
        if shuffle:
            self.rng = np.random.RandomState(ListOpsBucketSampler.random_seed)
            ListOpsBucketSampler.random_seed += 1
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.length = len(data) // batch_size if drop_last else - (-len(data) // batch_size)

    def __iter__(self):
        if self.shuffle:
            for data in self.data.values():
                self.rng.shuffle(data)
            freqs = [len(e) for e in self.data.values()]
        available_lengths = list(self.data.keys())
        if self.shuffle:
            get_seq_len = lambda: self.rng.choice(available_lengths, p=freqs / np.sum(freqs))
        else:
            get_seq_len = lambda: available_lengths[0]
        batch = []
        b_size = self.batch_size
        k = get_seq_len()
        progress = defaultdict(int)
        i = 0
        while i < self.length:
            batch.extend(self.data[k][progress[k]:progress[k] + b_size])
            progress[k] += b_size
            if len(batch) == self.batch_size:
                yield batch
                i += 1
                batch = []
                b_size = self.batch_size
                k = get_seq_len()
            else:
                b_size = self.batch_size - len(batch)
                progress[k] = 0
                if self.shuffle:
                    self.rng.shuffle(self.data[k])
                else:
                    idx = available_lengths.index(k)
                    del available_lengths[idx]
                    try:
                        k = get_seq_len()
                    except IndexError:
                        break
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return self.length
