# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from utils import clamp_grad
from utils import Categorical
from modules import BinaryTreeBasedModule


class BottomUpTreeLstmParser(BinaryTreeBasedModule):
    def __init__(self, input_dim, hidden_dim,
                 leaf_transformation=BinaryTreeBasedModule.no_transformation, trans_hidden_dim=None, dropout_prob=None):
        super().__init__(input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob)
        self.q = nn.Parameter(torch.empty(size=(hidden_dim, ), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.q, mean=0, std=0.01)

    def forward(self, x, mask, relaxed=False, tau_weights=None, straight_through=False, noise=None, ev_actions=None):
        if relaxed is False and (straight_through is True or tau_weights is not None):
            raise ValueError("(straight_through == True or tau_weights is not None) and relaxed == False are mutually "
                             "exclusive values!")
        if relaxed is True and tau_weights is None:
            raise ValueError("tau_weights == None and relaxed == True are mutually exclusive values!")
        probs = []
        gumbel_noise = []
        actions = []
        entropy = []
        normalized_entropy = []
        log_prob = []
        h, c = self._transform_leafs(x, mask)
        for i in range(1, x.shape[1]):
            noise_i = None if noise is None else noise[i - 1]
            ev_actions_i = None if ev_actions is None else ev_actions[i - 1]
            cat_distr, gumbel_noise_i, actions_i, h, c = self._make_step(h, c, mask[:, i:], relaxed, tau_weights,
                                                                         straight_through, noise_i, ev_actions_i)
            probs.append(cat_distr.probs)
            gumbel_noise.append(gumbel_noise_i)
            actions.append(actions_i)
            entropy.append(cat_distr.entropy)
            normalized_entropy.append(cat_distr.normalized_entropy)
            log_prob.append(cat_distr.log_prob(actions_i))
        log_prob = None if relaxed else sum(log_prob)
        entropy = sum(entropy)
        # normalize by the number of layers - 1.
        # -1 because the last layer contains only one possible action and the entropy is zero anyway.
        normalized_entropy = sum(normalized_entropy) / (torch.sum(mask[:, 2:], dim=-1) + 1e-17)

        if relaxed:
            return probs, gumbel_noise, entropy, normalized_entropy, actions, log_prob
        else:
            return probs, entropy, normalized_entropy, actions, log_prob

    def _make_step(self, h, c, mask, relaxed, tau_weights, straight_through, gumbel_noise, ev_actions):
        # ==== calculate the prob distribution over the merge actions and sample one ====
        h_l, c_l = h[:, :-1], c[:, :-1]
        h_r, c_r = h[:, 1:], c[:, 1:]
        h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)
        score = torch.matmul(h_p, self.q)  # (N x L x d, d) -> (N x L)
        cat_distr = Categorical(score, mask)
        if ev_actions is None:
            actions, gumbel_noise = self._sample_action(cat_distr, mask, relaxed, tau_weights, straight_through,
                                                        gumbel_noise)
        else:
            actions = ev_actions
        # ==== incorporate sampled action into the agent's representation of the environment state ====
        h_p, c_p = BinaryTreeBasedModule._merge(actions, h_l, c_l, h_r, c_r, h_p, c_p, mask)
        return cat_distr, gumbel_noise, actions, h_p, c_p

    def _sample_action(self, cat_distr, mask, relaxed, tau_weights, straight_through, gumbel_noise):
        if self.training:
            if relaxed:
                N = mask.sum(dim=-1, keepdim=True)
                tau = tau_weights[0] + tau_weights[1].exp() * torch.log(N + 1) + tau_weights[2].exp() * N
                actions, gumbel_noise = cat_distr.rsample(temperature=tau, gumbel_noise=gumbel_noise)
                if straight_through:
                    actions_hard = torch.zeros_like(actions)
                    actions_hard.scatter_(-1, actions.argmax(dim=-1, keepdim=True), 1.0)
                    actions = (actions_hard - actions).detach() + actions
                actions = clamp_grad(actions, -0.5, 0.5)
            else:
                actions, gumbel_noise = cat_distr.rsample(gumbel_noise=gumbel_noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise
