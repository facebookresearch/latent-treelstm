# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from modules import BinaryTreeLstmRnn
from modules import BinaryTreeBasedModule


class IdealTreeModel(nn.Module):
    def __init__(self, vocab_size, word_dim, hidden_dim, label_dim,
                 leaf_transformation=BinaryTreeBasedModule.no_transformation, trans_hidden_dim=None):
        super().__init__()
        self.embd_tree = nn.Embedding(vocab_size, word_dim)
        self.tree_lstm_rnn = BinaryTreeLstmRnn(word_dim, hidden_dim, leaf_transformation, trans_hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=label_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embd_tree.weight, 0.0, 0.01)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val=0)
        self.tree_lstm_rnn.reset_parameters()

    def forward(self, x, tree, mask, labels):
        h = self.tree_lstm_rnn(self.embd_tree(x), tree, mask)
        logits = self.linear(h)
        ce_loss = self.criterion(input=logits, target=labels)
        pred_labels = logits.argmax(dim=1)
        return ce_loss, pred_labels

    def reset_memory_managers(self):
        if hasattr(self.tree_lstm_rnn, "bt_reducer"):
            self.tree_lstm_rnn.bt_reducer.reset()
