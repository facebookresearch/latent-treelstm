# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils import Categorical
from torch_tree import MemoryManager
from torch_tree.cpp import get_indexes_mapping_parser
from torch_tree.functions import UpdateScoresFunction
from torch_tree.functions import BinaryTreeParserSelectFunction


class BinaryTreeParser(MemoryManager):
    def __call__(self, *tensors, reduce, calculate_score, sample_parser_action, mask=None):
        batch_size = tensors[0].shape[0]
        seq_len = tensors[0].shape[1]
        device = tensors[0].device

        if mask is not None:
            n = mask.sum(dim=-1)
            ones = torch.ones_like(n)

        nodes = [e.contiguous() for e in tensors]
        grad_nodes = [None for _ in nodes]
        nodes_p_cpu, nodes_p_gpu, grad_nodes_p_cpu, grad_nodes_p_gpu = self.get_pointers(tensors)
        sampled_nodes = []
        cat_distrs = []
        actions = []
        for i in range(seq_len - 1):
            if i == 0:
                # ===================== select =====================
                nodes_l, nodes_r = zip(*((node[:, :-1], node[:, 1:]) for node in nodes))
                # ===================== reduce =====================
                nodes_p = reduce(*nodes_l, *nodes_r)
                # =============== calculate scores =================
                scores = calculate_score(*nodes_p)
                idx_mapping, idx_mapping_p = BinaryTreeParser.get_indexes_mapping_parser(batch_size, seq_len, device)
                nodes_p = [[node_p] for node_p in nodes_p]
                grad_nodes_p = [[] for _ in nodes_p]
                if mask is not None:
                    n = n - ones
            else:
                # ===================== select =====================
                nodes_l = []
                nodes_c = []
                nodes_r = []
                for k, (e, grad_e, e_p, e_p_cpu, e_p_gpu, grad_e_p, grad_e_p_cpu, grad_e_p_gpu) \
                        in enumerate(zip(nodes, grad_nodes, nodes_p, nodes_p_cpu, nodes_p_gpu,
                                         grad_nodes_p, grad_nodes_p_cpu, grad_nodes_p_gpu)):
                    args = [e if grad_e is None else e.detach()]
                    if grad_e is not None:
                        args.append(grad_e)
                    args.extend(_.detach() for _ in e_p[:-1])  # a motivation for this is explained in select function doc
                    args.append(e_p[-1])
                    args.extend(grad_e_p)
                    result = BinaryTreeParserSelectFunction.apply(actions[-1], idx_mapping, idx_mapping_p,
                                                                  e_p_cpu, e_p_gpu, grad_e_p_cpu, grad_e_p_gpu, *args)
                    nodes_l.append(result[0])
                    nodes_c.append(result[1])
                    nodes_r.append(result[2])
                    if len(result) > 3:
                        grad_e_p.append(result[3])
                        if grad_e is None:
                            grad_nodes[k] = result[4]
                sampled_nodes.append(nodes_c)
                # ===================== reduce =====================
                nodes_p_lc = [e.contiguous() for e in reduce(*nodes_l, *nodes_c)]
                nodes_p_cr = [e.contiguous() for e in reduce(*nodes_c, *nodes_r)]
                if mask is not None:
                    n = n - ones
                    _mask_i = (n > 0).to(dtype=n.dtype)
                    _a_mask = (actions[-1] != 0).to(dtype=mask.dtype)
                    for k, (node_p_lc, node_p_cr, node_l, node_c) in \
                            enumerate(zip(nodes_p_lc, nodes_p_cr, nodes_l, nodes_c)):
                        mask_i = _mask_i.view([batch_size] + (node_p_lc.dim() - 1) * [1])
                        a_mask = _a_mask.view([batch_size] + (node_p_lc.dim() - 1) * [1])
                        # do not use calculated composition, instead propagate current one to the top
                        candidate_node = (node_l * a_mask + node_c * (1.0 - a_mask)) * (1.0 - mask_i)
                        node_p_lc = node_p_lc * mask_i + candidate_node
                        node_p_cr = node_p_cr * mask_i + candidate_node
                        nodes_p[k].append(torch.stack([node_p_lc, node_p_cr], dim=1))
                else:
                    for k, (node_p_lc, node_p_cr) in enumerate(zip(nodes_p_lc, nodes_p_cr)):
                        nodes_p[k].append(torch.stack([node_p_lc, node_p_cr], dim=1))

                # =============== calculate scores =================
                new_scores = calculate_score(*(e[-1] for e in nodes_p))
                scores = UpdateScoresFunction.apply(scores, new_scores, actions[-1])
                idx_mapping, idx_mapping_p = \
                    BinaryTreeParser.get_indexes_mapping_parser(actions[-1], idx_mapping, idx_mapping_p, i)

            cat_distrs.append(Categorical(scores, None if mask is None else mask[:, i + 1:]))
            actions.append(sample_parser_action(i, cat_distrs[i]))

            if i == seq_len - 2:
                row_idxs = torch.arange(batch_size, device=device)
                col_idxs = idx_mapping_p[:batch_size, 0]
                sampled_nodes.append(list(e[-1][row_idxs, col_idxs] for e in nodes_p))

        result = [cat_distrs, actions]
        result.extend(zip(*sampled_nodes))
        return result

    @staticmethod
    def get_indexes_mapping_parser(*args):
        if len(args) == 4:
            return get_indexes_mapping_parser(*args)
        batch_size, seq_len, device = args
        index_mapping = torch.cat([torch.arange(0, seq_len, dtype=torch.long, device=device).repeat(batch_size, 1),
                                   torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)], dim=0)
        index_mapping_p = torch.cat(
            [torch.arange(0, seq_len - 1, dtype=torch.long, device=device).repeat(batch_size, 1),
             torch.zeros(batch_size, seq_len - 1, dtype=torch.long, device=device)], dim=0)
        return index_mapping, index_mapping_p
