# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch_tree.cpp import select_parser_forward
from torch_tree.cpp import select_parser_backward


class BinaryTreeParserSelectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, actions, index_mapping, index_mapping_p, h_p_cpu, h_p_gpu, grad_h_p_cpu, grad_h_p_gpu, *args):
        requires_grad = grad_h_p_cpu is not None
        if requires_grad:
            if len(args) == 2:
                h, h_p = args[0], [args[1]]
                grad_h = torch.zeros_like(h)
                grad_h_p = []
                current_level_grad_h_p = torch.zeros_like(h_p[-1])
                ctx.mark_non_differentiable(current_level_grad_h_p, grad_h)
            else:
                h, grad_h = args[:2]
                args = args[2:]
                h_p = args[:len(args) // 2 + 1]
                grad_h_p = args[len(args) // 2 + 1:]
                current_level_grad_h_p = torch.zeros_like(h_p[-1])
                ctx.mark_non_differentiable(current_level_grad_h_p)
            ctx.save_for_backward(actions, index_mapping, index_mapping_p, grad_h, *grad_h_p, current_level_grad_h_p)
            ctx.grad_h_p_cpu = grad_h_p_cpu
            ctx.grad_h_p_gpu = grad_h_p_gpu
        else:
            h, *h_p = args

        h_l, h_c, h_r = select_parser_forward(h, h_p, h_p_cpu, h_p_gpu, actions, index_mapping, index_mapping_p)

        result = [h_l, h_c, h_r]
        if requires_grad:
            result.append(current_level_grad_h_p)
            if len(args) == 2:
                result.append(grad_h)
        return tuple(result)

    @staticmethod
    def backward(ctx, grad_h_l, grad_h_c, grad_h_r, *_):
        # _ should be a list of None but it is not https://github.com/pytorch/pytorch/issues/12631
        actions, index_mapping, index_mapping_p, grad_h, *grad_h_p = ctx.saved_variables
        select_parser_backward(grad_h_l.contiguous(), grad_h_c.contiguous(), grad_h_r.contiguous(), grad_h,
                               grad_h_p, ctx.grad_h_p_cpu, ctx.grad_h_p_gpu, actions, index_mapping, index_mapping_p)
        none_list = (len(grad_h_p) - 1) * [None]
        if len(grad_h_p) == 1:
            return (None, None, None, None, None, None, None, grad_h, *none_list, grad_h_p[-1], *none_list)
        else:
            return (None, None, None, None, None, None, None, None, None, *none_list, grad_h_p[-1], *none_list)

