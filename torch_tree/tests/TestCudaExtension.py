# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from unittest import TestCase
from torch_tree import MemoryManager
from torch_tree.cpp import select_forward
from torch_tree.cpp import get_indexes_mapping
from torch_tree.cpp import select_parser_forward
from torch_tree.cpp import update_scores_forward
from torch_tree.cpp import get_indexes_mapping_parser


class TestCudaExtension(TestCase):
    def test_index_mapping(self):
        for device_id in range(torch.cuda.device_count()):
            results = {"index_mapping": [torch.LongTensor([[0, 1, 2, 1],
                                                           [1, 2, 3, 4]]).cuda(device_id),
                                         torch.LongTensor([[0, 2, 1],
                                                           [1, 2, 2]]).cuda(device_id),
                                         torch.LongTensor([[0, 3],
                                                           [3, 2]]).cuda(device_id),
                                         torch.LongTensor([[4],
                                                     [4]]).cuda(device_id)],
                       "is_leaf": [torch.ByteTensor([[1, 1, 1, 0],
                                                     [0, 1, 1, 1]]).cuda(device_id),
                                   torch.ByteTensor([[1, 0, 0],
                                                     [0, 1, 0]]).cuda(device_id),
                                   torch.ByteTensor([[1, 0],
                                                     [0, 0]]).cuda(device_id),
                                   torch.ByteTensor([[0],
                                                     [0]]).cuda(device_id)]}
            actions = [torch.LongTensor([3, 0]).cuda(device_id),
                       torch.LongTensor([1, 2]).cuda(device_id),
                       torch.LongTensor([1, 0]).cuda(device_id),
                       torch.LongTensor([0, 0]).cuda(device_id)]
            prev_is_leaf = torch.ones(2, 5, dtype=torch.uint8).cuda(device_id)
            prev_index_mapping = torch.arange(0, 5, dtype=torch.long, device=device_id).repeat(2, 1)

            for step in range(4):
                prev_index_mapping, prev_is_leaf = \
                    get_indexes_mapping(actions[step], prev_index_mapping, prev_is_leaf, step + 1)

                self.assertTrue(torch.all(torch.eq(results["index_mapping"][step], prev_index_mapping)))
                self.assertTrue(torch.all(torch.eq(results["is_leaf"][step], prev_is_leaf)))

    def test_index_mapping_speed(self):
        bs = 1024
        seq_len = 10000

        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(device_id)
            total_t = 0
            warmup = 10
            N = 10000
            for i in range(N + warmup):
                actions = torch.LongTensor(bs).random_(0, seq_len - 1).cuda(device_id)
                prev_index_mapping = torch.arange(0, seq_len, dtype=torch.long, device=device_id).repeat(bs, 1)
                prev_is_leaf = torch.ones(bs, seq_len, dtype=torch.uint8).cuda(device_id)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                index_mapping, is_leaf = get_indexes_mapping(actions, prev_index_mapping, prev_is_leaf, 0)
                end_event.record()
                if i > warmup:
                    torch.cuda.synchronize()
                    total_t += start_event.elapsed_time(end_event)
                del index_mapping
                del is_leaf
            print(total_t / N)
            self.assertTrue(total_t / N > 0)

    def test_select_forward_speed(self):
        bs = 256
        seq_len = 100
        vector_dim = 4096
        mm = MemoryManager()

        for dtype in [torch.float, torch.double]:
            for device_id in range(torch.cuda.device_count()):
                torch.cuda.set_device(device_id)
                total_t = 0
                N = 20
                warmup = 5
                for i in range(N + warmup):
                    h = [torch.rand(bs, seq_len, vector_dim, dtype=dtype).cuda(device_id)]
                    index_mapping = torch.arange(0, seq_len, dtype=torch.long, device=device_id).repeat(bs, 1)
                    is_leaf = torch.ones(bs, seq_len, dtype=torch.uint8).cuda(device_id)
                    mm.reset()
                    h_cpu_p, h_gpu_p, _, _ = mm.get_pointers(h)
                    for step in range(seq_len - 1):
                        actions = torch.LongTensor(bs).random_(0, seq_len - step - 1).cuda(device_id)
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        h_l, h_r = select_forward(h, h_cpu_p[0], h_gpu_p[0], actions, index_mapping, is_leaf)
                        end_event.record()
                        if i > warmup:
                            torch.cuda.synchronize()
                            total_t += start_event.elapsed_time(end_event)
                        h.append(h_l + h_r)
                        index_mapping, is_leaf = get_indexes_mapping(actions, index_mapping, is_leaf, step)
                    del h
                    del index_mapping
                    del is_leaf
                print(total_t / (N * (seq_len - 1)))
                self.assertTrue(total_t / N > 0)

    def test_get_indexes_mapping_parser(self):
        for device_id in range(torch.cuda.device_count()):
            results = {"index_mapping": [torch.LongTensor([[0, 1, 2, 3, 5],
                                                           [0, 2, 3, 4, 5],
                                                           [-1, -1, -1, 0, -1],
                                                           [0, -1, -1, -1, -1]]).cuda(device_id),
                                         torch.LongTensor([[0, 2, 3, 5],
                                                           [0, 2, 3, 5],
                                                           [0, -1, 0, -1],
                                                           [0, -1, 0, -1]]).cuda(device_id),
                                         torch.LongTensor([[0, 0, 5],
                                                           [0, 0, 5],
                                                           [0, 1, -1],
                                                           [0, 2, -1]]).cuda(device_id),
                                         torch.LongTensor([[0, 5],
                                                           [0, 1],
                                                           [3, -1],
                                                           [0, 3]]).cuda(device_id)],
                       "index_mapping_p": [torch.LongTensor([[0, 1, 0,  1],
                                                             [1, 2, 3,  4],
                                                             [0, 0, 1,  1],
                                                             [1, 0, 0,  0]]).cuda(device_id),
                                           torch.LongTensor([[1, 0,  1],
                                                             [1, 0,  1],
                                                             [2, 1,  1],
                                                             [1, 2,  2]]).cuda(device_id),
                                           torch.LongTensor([[0, 1],
                                                             [0, 1],
                                                             [3, 3],
                                                             [3, 3]]).cuda(device_id),
                                           torch.LongTensor([[1],
                                                             [0],
                                                             [4],
                                                             [4]]).cuda(device_id)]}
            actions = [torch.LongTensor([3, 0]).cuda(device_id),
                       torch.LongTensor([0, 2]).cuda(device_id),
                       torch.LongTensor([1, 1]).cuda(device_id),
                       torch.LongTensor([0, 1]).cuda(device_id)]
            seq_len = 6
            batch_size = 2
            im = torch.cat([torch.arange(0, seq_len, dtype=torch.long, device=device_id).repeat(batch_size, 1),
                            -torch.ones(batch_size, seq_len, dtype=torch.long, device=device_id)], dim=0)
            im_p = torch.cat([torch.arange(0, seq_len - 1, dtype=torch.long, device=device_id).repeat(batch_size, 1),
                              torch.zeros(batch_size, seq_len - 1, dtype=torch.long, device=device_id)], dim=0)

            for step in range(1, seq_len - 1):
                im, im_p = get_indexes_mapping_parser(actions[step - 1], im, im_p, step)
                self.assertTrue(torch.all(torch.eq(results["index_mapping"][step - 1], im)))
                self.assertTrue(torch.all(torch.eq(results["index_mapping_p"][step - 1], im_p)))

    def test_get_indexes_mapping_parser_speed(self):
        bs = 1024
        seq_len = 10000

        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(device_id)
            total_t = 0
            warmup = 5
            N = 10000
            for i in range(N + warmup):
                actions = torch.LongTensor(bs).random_(0, seq_len - 1).cuda(device_id)
                im = torch.cat([torch.arange(0, seq_len, dtype=torch.long, device=device_id).repeat(bs, 1),
                                -torch.ones(bs, seq_len, dtype=torch.long, device=device_id)], dim=0)
                im_p = torch.cat([torch.arange(0, seq_len - 1, dtype=torch.long, device=device_id).repeat(bs, 1),
                                  torch.zeros(bs, seq_len - 1, dtype=torch.long, device=device_id)], dim=0)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                im, im_p = get_indexes_mapping_parser(actions, im, im_p, 0)
                end_event.record()
                if i > warmup:
                    torch.cuda.synchronize()
                    total_t += start_event.elapsed_time(end_event)
                del im
                del im_p
            print(total_t / N)
            self.assertTrue(total_t / N > 0)

    def test_select_forward_parser_speed(self):
        bs = 256
        seq_len = 100
        vector_dim = 4096
        mm = MemoryManager()

        for dtype in [torch.float, torch.double]:
            for device_id in range(torch.cuda.device_count()):
                torch.cuda.set_device(device_id)
                total_t = 0
                N = 20
                warmup = 5
                for i in range(N + warmup):
                    h = torch.rand(bs, seq_len, vector_dim, dtype=dtype).cuda(device_id)
                    h_p = [torch.rand(bs, seq_len - 1, vector_dim, dtype=dtype).cuda(device_id)]
                    im = torch.cat([torch.arange(0, seq_len, dtype=torch.long, device=device_id).repeat(bs, 1),
                                    -torch.ones(bs, seq_len, dtype=torch.long, device=device_id)], dim=0)
                    im_p = torch.cat([torch.arange(0, seq_len - 1, dtype=torch.long, device=device_id).repeat(bs, 1),
                                      torch.zeros(bs, seq_len - 1, dtype=torch.long, device=device_id)], dim=0)
                    mm.reset()
                    h_p_cpu, h_p_gpu, _, _ = mm.get_pointers([h])
                    for step in range(seq_len - 1):
                        actions = torch.LongTensor(bs).random_(0, seq_len - step - 1).cuda(device_id)
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        h_l, h_c, h_r = select_parser_forward(h, h_p, h_p_cpu[0], h_p_gpu[0], actions, im, im_p)
                        end_event.record()
                        if i > warmup:
                            torch.cuda.synchronize()
                            total_t += start_event.elapsed_time(end_event)
                        h_p.append(torch.stack([h_l + h_c, h_c + h_r], dim=1))
                        if step != seq_len - 2:
                            im, im_p = get_indexes_mapping_parser(actions, im, im_p, step)
                    del h
                    del h_p
                    del im
                    del im_p
                print(total_t / (N * (seq_len - 1)))
                self.assertTrue(total_t / N > 0)

    def test_update_scores_forward_speed(self):
        bs = 512
        seq_len = 10000

        for dtype in [torch.float, torch.double]:
            for device_id in range(torch.cuda.device_count()):
                torch.cuda.set_device(device_id)
                total_t = 0
                N = 20
                warmup = 5
                for i in range(N + warmup):
                    scores = torch.rand(bs, seq_len - 1, dtype=dtype).cuda(device_id)
                    new_scores = torch.rand(bs, 2, dtype=dtype).cuda(device_id)
                    actions = torch.LongTensor(bs).random_(0, seq_len - 1).cuda(device_id)

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    updated_scores = update_scores_forward(scores, new_scores, actions)
                    end_event.record()
                    if i > warmup:
                        torch.cuda.synchronize()
                        total_t += start_event.elapsed_time(end_event)
                    del scores
                    del new_scores
                    del updated_scores

                print(total_t / N)
                self.assertTrue(total_t / N > 0)
