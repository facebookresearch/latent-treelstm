# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from listops.models import PpoModel
from utils import AverageMeter
from torch.utils.data import DataLoader
from listops.data_preprocessing import ListOpsDataset


def to_tree(seq, argmax):
    if argmax:
        return to_tree(seq[:argmax[0]] + [f"( {seq[argmax[0]]} {seq[argmax[0]+1]} )"] + seq[argmax[0]+2:], argmax[1:])
    return seq[0]


device = 0
with torch.cuda.device(device):
    test_dataset = ListOpsDataset("data/listops/interim/test.tsv", "data/listops/processed/vocab.txt")
    test_data = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False,
                           collate_fn=ListOpsDataset.collate_fn, pin_memory=True)
    print(len(test_dataset))

    model = PpoModel(vocab_size=test_dataset.vocab_size,
                     word_dim=128,
                     hidden_dim=128,
                     label_dim=test_dataset.label_size,
                     parser_leaf_transformation="lstm_transformation",
                     parser_trans_hidden_dim=128).cuda(device)
    checkpoint = torch.load("data/listops/ppo/models/exp0/m177699120722052117/13-52.mdl")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open("data/listops/processed/vocab.txt") as f:
        idx_to_word = [word.strip() for word in f.readlines()]
        word_to_idx = dict()
        for idx, word in enumerate(idx_to_word):
            word_to_idx[word] = idx

    accuracy_meter = AverageMeter()
    for batch_idx, (labels, tokens, mask) in enumerate(test_data):
        print(batch_idx)
        labels = labels.to(device=device, non_blocking=True)
        tokens = tokens.to(device=device, non_blocking=True)
        mask = mask.to(device=device, non_blocking=True)

        with torch.no_grad():
            pred_labels, ce_loss, rewards, actions, actions_log_prob, entropy, normalized_entropy = \
                model(tokens, mask, labels)
            accuracy = (labels == pred_labels).to(dtype=torch.float32)

        for kkk, e in enumerate(accuracy.data.cpu().numpy()):
            accuracy_meter.update(e)
            # if e == 0:
            #     length = int(mask.sum(dim=1).cpu().numpy()[kkk])
            #     qqq = [idx_to_word[e] for e in tokens.data[kkk].cpu().numpy()]
            #     argmax = [np.argmax(e.data[kkk].cpu().numpy()) for e in actions]
            #     print(to_tree(qqq[:length], argmax[:length-1]))

    print(accuracy_meter.avg)
    print(accuracy_meter.sum)
    print(accuracy_meter.count)
