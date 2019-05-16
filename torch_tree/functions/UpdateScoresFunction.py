# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch_tree.cpp import update_scores_forward
from torch_tree.cpp import update_scores_backward


class UpdateScoresFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, new_scores, actions):
        ctx.save_for_backward(actions)
        return update_scores_forward(scores, new_scores, actions)

    @staticmethod
    def backward(ctx, grad_updated_scores):
        actions = ctx.saved_variables[0]
        grad_scores, grad_new_scores = update_scores_backward(grad_updated_scores, actions)
        return grad_scores, grad_new_scores, None
