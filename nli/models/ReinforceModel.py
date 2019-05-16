# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from itertools import chain
from torch.nn import functional as F
from modules import BinaryTreeLstmRnn
from modules import BinaryTreeBasedModule
from modules import BottomUpTreeLstmParser


class ReinforceModel(nn.Module):
    no_baseline = "no_baseline"
    ema = "ema"
    self_critical = "self_critical"

    def __init__(self, vocab_size, word_dim, hidden_dim, mlp_hidden_dim, label_dim, dropout_prob,
                 parser_leaf_transformation=BinaryTreeBasedModule.no_transformation, parser_trans_hidden_dim=None,
                 tree_leaf_transformation=BinaryTreeBasedModule.no_transformation, tree_trans_hidden_dim=None,
                 baseline_type=no_baseline, var_normalization=False, use_batchnorm=False):
        super().__init__()
        self.embd_parser = nn.Embedding(vocab_size, word_dim)
        self.parser = BottomUpTreeLstmParser(word_dim, hidden_dim, parser_leaf_transformation, parser_trans_hidden_dim)
        self.embd_tree = nn.Embedding(vocab_size, word_dim)
        self.tree_lstm_rnn = BinaryTreeLstmRnn(word_dim, hidden_dim, tree_leaf_transformation, tree_trans_hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(in_features=4 * hidden_dim, out_features=mlp_hidden_dim)
        self.linear2 = nn.Linear(in_features=mlp_hidden_dim, out_features=label_dim)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(num_features=4 * hidden_dim)
            self.bn2 = nn.BatchNorm1d(num_features=mlp_hidden_dim)

        self.baseline_params = ReinforceModel.get_baseline_dict(baseline_type)
        self.var_norm_params = {"var_normalization": var_normalization, "var": 1.0, "alpha": 0.9}
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embd_parser.weight, 0.0, 0.01)
        nn.init.normal_(self.embd_tree.weight, 0.0, 0.01)
        self.parser.reset_parameters()
        self.tree_lstm_rnn.reset_parameters()

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, val=0)
        nn.init.uniform_(self.linear2.weight, -0.005, 0.005)
        nn.init.constant_(self.linear2.bias, val=0)
        if self.use_batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

    def get_policy_parameters(self):
        if self.embd_parser.weight.requires_grad:
            return list(chain(self.embd_parser.parameters(), self.parser.parameters()))
        return list(self.parser.parameters())

    def get_environment_parameters(self):
        if self.embd_tree.weight.requires_grad:
            return list(chain(self.embd_tree.parameters(), self.tree_lstm_rnn.parameters(),
                              self.linear1.parameters(), self.linear2.parameters()))
        return list(chain(self.tree_lstm_rnn.parameters(), self.linear1.parameters(), self.linear2.parameters()))

    def forward(self, premises, p_mask, hypotheses, h_mask, labels):
        entropy, normalized_entropy, actions, actions_log_prob, logits, rewards = \
            self._forward(premises, p_mask, hypotheses, h_mask, labels)
        ce_loss = rewards.mean()
        if self.training:
            baseline = self._get_baseline(rewards, premises, p_mask, hypotheses, h_mask, labels)
            rewards = self._normalize(rewards - baseline)
        pred_labels = logits.argmax(dim=1)
        return pred_labels, ce_loss, rewards.detach(), actions, actions_log_prob, entropy, normalized_entropy

    def _forward(self, premises, p_mask, hypotheses, h_mask, labels):
        p_embd_parser = self.dropout(self.embd_parser(premises))
        h_embd_parser = self.dropout(self.embd_parser(hypotheses))
        p_entropy, p_normalized_entropy, p_actions, p_actions_log_prob = self.parser(p_embd_parser, p_mask)[1:]
        h_entropy, h_normalized_entropy, h_actions, h_actions_log_prob = self.parser(h_embd_parser, h_mask)[1:]

        p_embd_tree = self.dropout(self.embd_tree(premises))
        h_embd_tree = self.dropout(self.embd_tree(hypotheses))
        prem_h = self.tree_lstm_rnn(p_embd_tree, p_actions, p_mask)
        hyp_h = self.tree_lstm_rnn(h_embd_tree, h_actions, h_mask)
        h = torch.cat([prem_h, hyp_h, (prem_h - hyp_h).abs(), prem_h * hyp_h], dim=1)

        if self.use_batchnorm:
            h = self.bn1(h)
        h = self.dropout(h)

        h = self.linear1(h)
        h = F.relu(h)
        if self.use_batchnorm:
            h = self.bn2(h)
        h = self.dropout(h)

        logits = self.linear2(h)
        rewards = self.criterion(input=logits, target=labels)

        actions = {"p_actions": p_actions, "h_actions": h_actions}
        actions_log_prob = p_actions_log_prob + h_actions_log_prob
        entropy = (p_entropy + h_entropy) / 2.0
        normalized_entropy = (p_normalized_entropy + h_normalized_entropy) / 2.0

        return entropy, normalized_entropy, actions, actions_log_prob, logits, rewards

    def _get_baseline(self, rewards, premises, p_mask, hypotheses, h_mask, labels):
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
                self.dropout.train()
                rewards = self._forward(premises, p_mask, hypotheses, h_mask, labels)[-1]
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
            return {"type": baseline_type, "mean": 0.48, "alpha": 0.9}  # 0.48 ~= -np.log(1./3)
        if baseline_type == ReinforceModel.self_critical:
            return {"type": baseline_type}
        raise ValueError(f"There is no {baseline_type} baseline!")
