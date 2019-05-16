# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import Dataset
from listops.data_preprocessing import load_listops_data


class ListOpsDatasetTrees(Dataset):
    def __init__(self, data_path, vocab_path, max_len=float("inf")):
        super().__init__()
        self.data, idx_to_word, word_to_idx = load_listops_data(data_path, vocab_path, max_len)
        for e in self.data:
            e["transitions"] = ListOpsDatasetTrees._convert(e["transitions"])
        self.vocab_size = len(idx_to_word)
        self.label_size = 10

    def __getitem__(self, index):
        e = self.data[index]
        return e["label"], e["tokens"], e["transitions"]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        labels, tokens, _trees = zip(*data)
        labels = torch.tensor(labels, dtype=torch.long)
        max_len = max(len(e) for e in tokens)
        mask = torch.zeros((len(tokens), max_len), dtype=torch.float32)
        for idx, e in enumerate(tokens):
            mask[idx, :len(e)] = 1
        trees = []
        for i in range(max_len - 1):
            trees.append(torch.tensor([e[i] if i < len(e) else 0 for e in _trees], dtype=torch.long))
        tokens = [e + [0] * (max_len - len(e)) for e in tokens]
        tokens = torch.tensor(tokens, dtype=torch.long)
        return labels, tokens, trees, mask

    @staticmethod
    def _convert(sr_tree):
        merge_tree = []
        idx = -1
        for e in sr_tree:
            if e == 0:
                idx += 1
            else:
                idx -= 1
                merge_tree.append(idx)
        return merge_tree
