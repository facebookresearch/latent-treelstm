# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

T_SHIFT = 0
T_REDUCE = 1


def load_listops_data(data_path, vocab_path=None, max_len=float("inf")):
    if vocab_path:
        with open(vocab_path) as f:
            idx_to_word = [word.strip() for word in f.readlines()]
            word_to_idx = dict()
            for idx, word in enumerate(idx_to_word):
                word_to_idx[word] = idx
    data = []
    with open(data_path) as f:
        too_long = 0
        too_short = 0
        for e_id, line in enumerate(f):
            label, seq = line.strip().split('\t')
            e = dict()
            e["label"] = int(label)
            e["sentence"] = seq
            e["tokens"], e["transitions"] = convert_bracketed_sequence(seq.split(' '))
            if len(e["tokens"]) > max_len:
                too_long += 1
                continue
            if len(e["tokens"]) == 1:
                too_short += 1
                continue
            if vocab_path:
                e["tokens"] = [word_to_idx[e] for e in e["tokens"]]
            e["id"] = str(e_id)
            data.append(e)
    print(f"file path: {data_path}")
    print(f"number of skipped sentences due to length > {max_len}: {too_long}")
    print(f"number of skipped sentences due to length < 2: {too_short}")
    if vocab_path:
        return data, idx_to_word, word_to_idx
    else:
        return data


def convert_bracketed_sequence(seq):
    tokens, transitions = [], []
    if len(seq) == 1:
        return seq, []
    for item in seq:
        if item == "(":
            continue
        if item == ")":
            transitions.append(T_REDUCE)
        else:
            tokens.append(item)
            transitions.append(T_SHIFT)
    return tokens, transitions
