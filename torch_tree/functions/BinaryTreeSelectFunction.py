# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch_tree.cpp import select_forward
from torch_tree.cpp import select_backward


class BinaryTreeSelectFunction(torch.autograd.Function):
    """
    This function selects vectors from the provided tensors according to the parser's actions.
    It takes as an input parser's `actions`, `index_mapping` and `is_leaf` for performing "logical" to "physical" index
    mapping and most importantly the `h_and_grad_h` variable that contains feature tensors from all lower levels of
    reduction and tensors where gradient with respect to those feature will be stored.
    This function treats all tensors in `h` except the last one as if it is not differentiable with respect to them,
    i.e. returning `None` gradient for each tensor in the backward function.
    That being said, there is a substantial performance gain if one indicates this fact to pytorch explicitly by setting
    `requires_grad` variable to False in the corresponding tensors (practically, this corresponds to using `.detach()`).
    It is implemented in this way to reduce memory overhead. Otherwise, one has to return very sparse true gradient
    for each tensor.
    Meanwhile, during the backward pass list of grad tensors which were manually created during the forward pass will
    be gradually filled in, and correct gradients eventually will be calculated and returned.
    """

    @staticmethod
    def forward(ctx, actions, index_mapping, is_leaf, h_cpu_p, h_gpu_p, grad_h_cpu_p, grad_h_gpu_p, *args):
        requires_grad = grad_h_cpu_p is not None
        if requires_grad:
            h = args[:len(args) // 2 + 1]
            grad_h = args[len(args) // 2 + 1:]
            # TODO(serhii): This can be a source of potential bugs if the last hidden state is not involved in the loss.
            #  If it is involved, current_level_grad_h will be always filled in correctly.
            current_level_grad_h = torch.empty_like(h[-1])
            ctx.mark_non_differentiable(current_level_grad_h)
            ctx.save_for_backward(actions, index_mapping, is_leaf, *grad_h, current_level_grad_h)
            ctx.grad_h_cpu_p = grad_h_cpu_p
            ctx.grad_h_gpu_p = grad_h_gpu_p
        else:
            h = args

        h_l, h_r = select_forward(h, h_cpu_p, h_gpu_p, actions, index_mapping, is_leaf)

        if requires_grad:
            return h_l, h_r, current_level_grad_h
        else:
            return h_l, h_r

    @staticmethod
    def backward(ctx, grad_h_l, grad_h_r, _):
        # _ should be a None but it is not https://github.com/pytorch/pytorch/issues/12631
        actions, index_mapping, is_leaf, *grad_h = ctx.saved_variables
        select_backward(grad_h, ctx.grad_h_cpu_p, ctx.grad_h_gpu_p, grad_h_l.contiguous(), grad_h_r.contiguous(),
                        actions, index_mapping, is_leaf)
        none_list = (len(grad_h) - 1) * [None]
        return (None, None, None, None, None, None, None, *none_list, grad_h[-1], *none_list)
