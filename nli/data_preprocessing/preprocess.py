# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import h5py
import pickle
import numpy as np


def load_glove(file_path):
    glove = {}
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.split(' ')
            glove[line[0]] = np.array([float(e) for e in line[1:]], dtype=np.float32)
    return glove


def tokenize_tree(tree, lower):
    return [(word.lower() if lower else word) for word in tree.split() if word not in ('(', ')')]


def load_data(file_path, lower):
    data = []
    with open(file_path) as f:
        for line in f:
            e = json.loads(line)
            new_e = {}
            if e["gold_label"] == '-':
                continue
            new_e["pairID"] = e["pairID"]
            new_e["label"] = e["gold_label"]
            new_e["premise"] = tokenize_tree(e["sentence1_binary_parse"], lower)
            new_e["hypothesis"] = tokenize_tree(e["sentence2_binary_parse"], lower)
            data.append(new_e)
    return data


def update_vocab(data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove, is_train=False):
    for e in data:
        if is_train and e["label"] not in label_to_idx:
            label_to_idx[e["label"]] = len(idx_to_label)
            idx_to_label.append(e["label"])
        for key in ["premise", "hypothesis"]:
            for word in e[key]:
                if word not in word_to_idx and (is_train or word in glove):
                    word_to_idx[word] = len(idx_to_word)
                    idx_to_word.append(word)


def preprocess_data(data, label_to_idx, word_to_idx, unk_idx):
    for e in data:
        if e["label"] in label_to_idx:
            e["label"] = label_to_idx[e["label"]]
        for key in ["premise", "hypothesis"]:
            e[key] = [word_to_idx.get(word, unk_idx) for word in e[key]]


def save_data(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def main(glove, lower):
    # ================ load data and tokenize sentences ================
    train_mnli_data = load_data("data/nli/multinli_1.0/multinli_1.0_train.jsonl", lower)
    valid_mnli_matched_data = load_data("data/nli/multinli_1.0/multinli_1.0_dev_matched.jsonl", lower)
    valid_mnli_mismatched_data = load_data("data/nli/multinli_1.0/multinli_1.0_dev_mismatched.jsonl", lower)
    test_mnli_matched_data = load_data("data/nli/multinli_1.0/multinli_0.9_test_matched_unlabeled.jsonl", lower)
    test_mnli_mismatched_data = load_data("data/nli/multinli_1.0/multinli_0.9_test_mismatched_unlabeled.jsonl", lower)

    train_snli_data = load_data("data/nli/snli_1.0/snli_1.0_train.jsonl", lower)
    valid_snli_data = load_data("data/nli/snli_1.0/snli_1.0_dev.jsonl", lower)
    test_snli_data = load_data("data/nli/snli_1.0/snli_1.0_test.jsonl", lower)
    test_snli_hard_data = load_data("data/nli/snli_1.0/snli_1.0_test_hard.jsonl", lower)

    # ============= create vocab from train_data and save it ============
    label_to_idx = {}
    idx_to_label = []
    word_to_idx = {}
    idx_to_word = []

    unk_idx = 0
    word_to_idx["<UNK>"] = unk_idx
    idx_to_word.append("<UNK>")

    update_vocab(train_mnli_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove, is_train=True)
    update_vocab(valid_mnli_matched_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove)
    update_vocab(valid_mnli_mismatched_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove)
    update_vocab(test_mnli_matched_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove)
    update_vocab(test_mnli_mismatched_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove)

    update_vocab(train_snli_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove, is_train=True)
    update_vocab(valid_snli_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove)
    update_vocab(test_snli_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove)
    update_vocab(test_snli_hard_data, word_to_idx, idx_to_word, label_to_idx, idx_to_label, glove)

    with open(f"data/nli/vocab_lower={lower}.pckl", "wb") as f:
        pickle.dump({"label_to_idx": label_to_idx,
                     "idx_to_label": idx_to_label,
                     "word_to_idx": word_to_idx,
                     "idx_to_word": idx_to_word,
                     "unk_idx": unk_idx}, f)

    # =============== generate embedding matrix and save it ===============
    rnd = np.random.RandomState(42)
    std = np.std([glove[word] for word in idx_to_word if word in glove], axis=0)
    np_glove = rnd.normal(scale=std, size=(len(word_to_idx), 300)).astype(np.float32)
    for i, word in enumerate(idx_to_word):
        if word in glove:
            np_glove[i] = glove[word]
    with h5py.File(f"data/nli/glove_lower={lower}.h5", 'w') as h5_file:
        h5_file["glove"] = np_glove

    # ==== preprocess data according to vocab and save it =====

    preprocess_data(train_mnli_data, label_to_idx, word_to_idx, unk_idx)
    preprocess_data(valid_mnli_matched_data, label_to_idx, word_to_idx, unk_idx)
    preprocess_data(valid_mnli_mismatched_data, label_to_idx, word_to_idx, unk_idx)
    preprocess_data(test_mnli_matched_data, label_to_idx, word_to_idx, unk_idx)
    preprocess_data(test_mnli_mismatched_data, label_to_idx, word_to_idx, unk_idx)

    save_data(train_mnli_data, f"data/nli/multinli_1.0/train_lower={lower}.pckl")
    save_data(valid_mnli_matched_data, f"data/nli/multinli_1.0/valid_matched_lower={lower}.pckl")
    save_data(valid_mnli_mismatched_data, f"data/nli/multinli_1.0/valid_mismatched_lower={lower}.pckl")
    save_data(test_mnli_matched_data, f"data/nli/multinli_1.0/test_matched_lower={lower}.pckl")
    save_data(test_mnli_mismatched_data, f"data/nli/multinli_1.0/test_mismatched_lower={lower}.pckl")

    preprocess_data(train_snli_data, label_to_idx, word_to_idx, unk_idx)
    preprocess_data(valid_snli_data, label_to_idx, word_to_idx, unk_idx)
    preprocess_data(test_snli_data, label_to_idx, word_to_idx, unk_idx)
    preprocess_data(test_snli_hard_data, label_to_idx, word_to_idx, unk_idx)

    save_data(train_snli_data, f"data/nli/snli_1.0/train_lower={lower}.pckl")
    save_data(valid_snli_data, f"data/nli/snli_1.0/valid_lower={lower}.pckl")
    save_data(test_snli_data, f"data/nli/snli_1.0/test_lower={lower}.pckl")
    save_data(test_snli_hard_data, f"data/nli/snli_1.0/test_hard_lower={lower}.pckl")


if __name__ == '__main__':
    glove = load_glove("data/vectors_cache/glove.840B.300d.txt")
    main(glove, lower=True)
    main(glove, lower=False)
