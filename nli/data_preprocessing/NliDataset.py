# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pickle
from torch.utils.data import Dataset


class NliDataset(Dataset):
    label_size = 3

    def __init__(self, data, max_len=float("inf"), is_mnli_test=False):
        super().__init__()
        self.data = [e for e in data if (len(e["premise"]) + len(e["hypothesis"]) < max_len)]
        self.is_mnli_test = is_mnli_test

    def __getitem__(self, index):
        e = self.data[index]
        if self.is_mnli_test:
            return int(e["pairID"]), e["premise"], e["hypothesis"]
        return e["label"], e["premise"], e["hypothesis"]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        labels, premises, hypotheses = zip(*data)
        labels = torch.tensor(labels, dtype=torch.long)
        p_max_len = max(len(e) for e in premises)
        h_max_len = max(len(e) for e in hypotheses)
        p_mask = torch.zeros((len(premises), p_max_len), dtype=torch.float32)
        h_mask = torch.zeros((len(hypotheses), h_max_len), dtype=torch.float32)
        for idx, e in enumerate(premises):
            p_mask[idx, :len(e)] = 1
        for idx, e in enumerate(hypotheses):
            h_mask[idx, :len(e)] = 1
        premises = [e + [0] * (p_max_len - len(e)) for e in premises]
        hypotheses = [e + [0] * (h_max_len - len(e)) for e in hypotheses]
        premises = torch.tensor(premises, dtype=torch.long)
        hypotheses = torch.tensor(hypotheses, dtype=torch.long)
        return labels, premises, p_mask, hypotheses, h_mask

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
