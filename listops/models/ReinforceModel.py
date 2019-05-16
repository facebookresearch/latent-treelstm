# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from itertools import chain
from modules import BinaryTreeLstmRnn
from modules import BinaryTreeBasedModule
from modules import BottomUpTreeLstmParser
# from relaxed_modules import BinaryTreeLstmRnn
# from relaxed_modules import BinaryTreeBasedModule
# from relaxed_modules import BottomUpTreeLstmParser


class ReinforceModel(nn.Module):
    no_baseline = "no_baseline"
    ema = "ema"
    self_critical = "self_critical"

    def __init__(self, vocab_size, word_dim, hidden_dim, label_dim,
                 parser_leaf_transformation=BinaryTreeBasedModule.no_transformation, parser_trans_hidden_dim=None,
                 tree_leaf_transformation=BinaryTreeBasedModule.no_transformation, tree_trans_hidden_dim=None,
                 baseline_type=no_baseline, var_normalization=False):
        super().__init__()
        self.embd_parser = nn.Embedding(vocab_size, word_dim)
        self.parser = BottomUpTreeLstmParser(word_dim, hidden_dim, parser_leaf_transformation, parser_trans_hidden_dim)
        self.embd_tree = nn.Embedding(vocab_size, word_dim)
        self.tree_lstm_rnn = BinaryTreeLstmRnn(word_dim, hidden_dim, tree_leaf_transformation, tree_trans_hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=label_dim)

        self.baseline_params = ReinforceModel.get_baseline_dict(baseline_type)
        self.var_norm_params = {"var_normalization": var_normalization, "var": 1.0, "alpha": 0.9}
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embd_parser.weight, 0.0, 0.01)
        nn.init.normal_(self.embd_tree.weight, 0.0, 0.01)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val=0)
        self.parser.reset_parameters()
        self.tree_lstm_rnn.reset_parameters()

    def get_policy_parameters(self):
        return list(chain(self.embd_parser.parameters(), self.parser.parameters()))

    def get_environment_parameters(self):
        return list(chain(self.embd_tree.parameters(), self.tree_lstm_rnn.parameters(), self.linear.parameters()))

    def forward(self, x, mask, labels):
        entropy, normalized_entropy, actions, actions_log_prob, logits, rewards = self._forward(x, mask, labels)
        ce_loss = rewards.mean()
        if self.training:
            baseline = self._get_baseline(rewards, x, mask, labels)
            rewards = self._normalize(rewards - baseline)
        pred_labels = logits.argmax(dim=1)
        return pred_labels, ce_loss, rewards.detach(), actions, actions_log_prob, entropy, normalized_entropy

    def _forward(self, x, mask, labels):
        entropy, normalized_entropy, actions, actions_log_prob = self.parser(self.embd_parser(x), mask)[1:]
        h = self.tree_lstm_rnn(self.embd_tree(x), actions, mask)
        logits = self.linear(h)
        rewards = self.criterion(input=logits, target=labels)
        return entropy, normalized_entropy, actions, actions_log_prob, logits, rewards

    def _get_baseline(self, rewards, x, mask, labels):
        with torch.no_grad():
            if self.baseline_params["type"] == ReinforceModel.no_baseline:
                return 0.0
            if self.baseline_params["type"] == ReinforceModel.ema:
                # If we use updated mean then it seems the estimator is biased because the baseline is a function of
                # the sampled actions. Nevertheless, the updated mean was used in the original paper
                # [https://arxiv.org/pdf/1402.0030.pdf] Appendix A.
                mean = self.baseline_params["mean"]
                alpha = self.baseline_params["alpha"]
                self.baseline_params["mean"] = self.baseline_params["mean"] * alpha + rewards.mean() * (1.0 - alpha)
                return mean
            elif self.baseline_params["type"] == ReinforceModel.self_critical:
                self.eval()
                rewards = self._forward(x, mask, labels)[-1]
                self.train()
                return rewards

    def _normalize(self, rewards):
        if self.var_norm_params["var_normalization"]:
            with torch.no_grad():
                alpha = self.var_norm_params["alpha"]
                self.var_norm_params["var"] = self.var_norm_params["var"] * alpha + rewards.var() * (1.0 - alpha)
                return rewards / self.var_norm_params["var"].sqrt().clamp(min=1.0)
        return rewards

    @staticmethod
    def get_baseline_dict(baseline_type):
        if baseline_type == ReinforceModel.no_baseline:
            return {"type": baseline_type}
        if baseline_type == ReinforceModel.ema:
            return {"type": baseline_type, "mean": 2.3, "alpha": 0.9}  # 2.3 ~= -np.log(1./10)
        if baseline_type == ReinforceModel.self_critical:
            return {"type": baseline_type}
        raise ValueError(f"There is no {baseline_type} baseline!")
