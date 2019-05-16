# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from nli.models import ReinforceModel


class PpoModel(ReinforceModel):
    def evaluate_actions(self, premises, p_mask, p_actions, hypotheses, h_mask, h_actions):
        p_embd_parser = self.dropout(self.embd_parser(premises))
        h_embd_parser = self.dropout(self.embd_parser(hypotheses))
        p_normalized_entropy, _, p_actions_log_prob = self.parser(p_embd_parser, p_mask, eval_actions=p_actions)[2:]
        h_normalized_entropy, _, h_actions_log_prob = self.parser(h_embd_parser, h_mask, eval_actions=h_actions)[2:]

        actions_log_prob = p_actions_log_prob + h_actions_log_prob
        normalized_entropy = (p_normalized_entropy + h_normalized_entropy) / 2.0

        return normalized_entropy, actions_log_prob
