# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from listops.data_preprocessing import load_listops_data


data = load_listops_data("data/listops/interim/train.tsv")
vocab = set()
for e in data:
    vocab.update(e["tokens"])


if not os.path.exists("data/listops/processed"):
    os.makedirs("data/listops/processed")
with open("data/listops/processed/vocab.txt", 'w') as f:
    for word in sorted(list(vocab)):
        f.write(word + '\n')
