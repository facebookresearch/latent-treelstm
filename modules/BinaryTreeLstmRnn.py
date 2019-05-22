# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from modules import BinaryTreeBasedModule


class BinaryTreeLstmRnn(BinaryTreeBasedModule):
    def __init__(self, input_dim, hidden_dim,
                 leaf_transformation=BinaryTreeBasedModule.no_transformation, trans_hidden_dim=None, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)

    def forward(self, x, parse_tree, mask):
        h, c = self._transform_leafs(x, mask)
        for i in range(x.shape[1] - 1):
            h_l, c_l = h[:, :-1], c[:, :-1]
            h_r, c_r = h[:, 1:], c[:, 1:]
            h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)
            h, c = self._merge(parse_tree[i], h_l, c_l, h_r, c_r, h_p, c_p, mask[:, i + 1:])
        return h.squeeze(dim=1)
