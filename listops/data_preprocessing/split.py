# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from shutil import copyfile


rnd = np.random.RandomState(42)
with open("data/listops/external/train_d20s.tsv") as f:
    lines = f.readlines()
    rnd.shuffle(lines)


if not os.path.exists("data/listops/interim"):
    os.makedirs("data/listops/interim")
with open("data/listops/interim/valid.tsv", 'w') as f:
    for line in lines[:1000]:
        f.write(line)
with open("data/listops/interim/train.tsv", 'w') as f:
    for line in lines[1000:]:
        f.write(line)

copyfile("data/listops/external/test_d20s.tsv", "data/listops/interim/test.tsv")
