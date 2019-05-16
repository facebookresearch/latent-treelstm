# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from sst.models import ReinforceModel


class PpoModel(ReinforceModel):
    def evaluate_actions(self, x, mask, actions):
        embd_parser_x = self.dropout(self.embd_parser(x))
        normalized_entropy, _, actions_log_prob = self.parser(embd_parser_x, mask, eval_actions=actions)[2:]
        return normalized_entropy, actions_log_prob
