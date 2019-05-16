# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch_tree import MemoryManager
from torch_tree.cpp import get_indexes_mapping
from torch_tree.functions import BinaryTreeSelectFunction


class BinaryTreeReducer(MemoryManager):
    def __call__(self, *tensors, binary_tree, reduce, mask=None):
        batch_size = tensors[0].shape[0]
        seq_len = tensors[0].shape[1]
        device = tensors[0].device

        if mask is not None:
            n = mask.sum(dim=-1)
            ones = torch.ones_like(n)

        nodes = [[e.contiguous()] for e in tensors]
        grad_nodes = [[] for _ in nodes]
        nodes_cpu_p, nodes_gpu_p, grad_nodes_cpu_p, grad_nodes_gpu_p = self.get_pointers(tensors)
        for i in range(seq_len - 1):
            if i == 0:
                idx_mapping, is_leaf = BinaryTreeReducer.get_indexes_mapping(batch_size, seq_len, device)
            else:
                idx_mapping, is_leaf = BinaryTreeReducer.get_indexes_mapping(binary_tree[i-1], idx_mapping, is_leaf, i)

            nodes_l = []
            nodes_r = []
            for e, e_cpu_p, e_gpu_p, grad_e, grad_e_cpu_p, grad_e_gpu_p in \
                    zip(nodes, nodes_cpu_p, nodes_gpu_p, grad_nodes, grad_nodes_cpu_p, grad_nodes_gpu_p):
                e = [_.detach() for _ in e[:-1]] + [e[-1]]  # a motivation for this is explained in select function doc
                result = BinaryTreeSelectFunction.apply(binary_tree[i], idx_mapping, is_leaf,
                                                        e_cpu_p, e_gpu_p, grad_e_cpu_p, grad_e_gpu_p, *e, *grad_e)
                nodes_l.append(result[0])
                nodes_r.append(result[1])
                if len(result) == 3:
                    grad_e.append(result[2])

            nodes_p = reduce(*nodes_l, *nodes_r)
            if mask is not None:
                n = n - ones
                _mask_i = (n > 0).to(dtype=n.dtype)
                for k, (node_p, node_l) in enumerate(zip(nodes_p, nodes_l)):
                    mask_i = _mask_i.view([batch_size] + (node_p.dim() - 1) * [1])
                    nodes[k].append(node_p * mask_i + node_l * (1.0 - mask_i))
            else:
                for k, node_p in enumerate(nodes_p):
                    nodes[k].append(node_p.contiguous())

        return [e[1:] for e in nodes]

    @staticmethod
    def get_indexes_mapping(*args):
        if len(args) == 4:
            return get_indexes_mapping(*args)
        batch_size, seq_len, device = args
        index_mapping = torch.arange(0, seq_len, dtype=torch.long, device=device).repeat(batch_size, 1)
        is_leaf = torch.ones(batch_size, seq_len, dtype=torch.uint8, device=device)
        return index_mapping, is_leaf
