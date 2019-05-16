# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import ctypes
import warnings
from torch_tree.cpp import mem_free
from torch_tree.cpp import allocate_gpu_double_pointer_array
from torch_tree.cpp import allocate_gpu_float_pointer_array


class MemoryManager:
    def __init__(self, buffer_max_size=10_000_000):
        self.buffer_max_size = buffer_max_size
        self.offset = dict()
        self.nodes_cpu = dict()
        self.nodes_gpu = dict()

    def get_pointers(self, tensors):
        seq_len = tensors[0].shape[1]
        device = tensors[0].device
        dtype = tensors[0].dtype
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError
        for e in tensors:
            if e.dtype != dtype or e.device != device:
                raise ValueError

        buffer_size = (sum(e.requires_grad for e in tensors) + len(tensors)) * (seq_len - 1)
        if buffer_size > self.buffer_max_size:
            raise ValueError
        elif self.buffer_max_size / buffer_size < 1000:
            warnings.warn("Input sequence is quite long. "
                          "Check the documentation to make sure you use this class correctly!")
            # TODO(serhii): We assume that time that it takes to initiate 1000 forward passes is long enough for the
            #               first initiated computations that use allocated memory to be finished. As a result we can
            #               reuse the pointers. PyTorch is asynchronous w.r.t GPU that is why if we don't wait long
            #               enough the pointer may be used again before the initiated computation on GPU used the
            #               correct content of the pointer.

        if (dtype, device) not in self.offset:
            self.offset[dtype, device] = 0
            self.nodes_cpu[dtype, device], self.nodes_gpu[dtype, device] = \
                MemoryManager._allocate(self.buffer_max_size, dtype, device)
        if self.buffer_max_size - self.offset[dtype, device] < buffer_size:
            self.offset[dtype, device] = 0

        offset = self.offset[dtype, device]
        self.offset[dtype, device] += buffer_size
        return MemoryManager._unpack_pointers(self.nodes_cpu[dtype, device], self.nodes_gpu[dtype, device],
                                              offset, tensors)

    @staticmethod
    def _unpack_pointers(_nodes_cpu, _nodes_gpu, offset, tensors):
        """
        This method just implements pointer arithmetic.
        """
        seq_len = tensors[0].shape[1]
        dtype = tensors[0].dtype

        nodes_cpu = []
        nodes_gpu = []
        grad_nodes_cpu = []
        grad_nodes_gpu = []

        value_type = ctypes.POINTER(ctypes.c_float) if dtype == torch.float32 else ctypes.POINTER(ctypes.c_double)
        offset = offset * ctypes.sizeof(value_type)
        for i in range(len(tensors)):
            nodes_cpu.append(_nodes_cpu + offset)
            nodes_gpu.append(_nodes_gpu + offset)
            offset += ctypes.sizeof(value_type) * (seq_len - 1)
            if tensors[i].requires_grad:
                grad_nodes_cpu.append(_nodes_cpu + offset)
                grad_nodes_gpu.append(_nodes_gpu + offset)
                offset += ctypes.sizeof(value_type) * (seq_len - 1)
            else:
                grad_nodes_cpu.append(None)
                grad_nodes_gpu.append(None)
        return nodes_cpu, nodes_gpu, grad_nodes_cpu, grad_nodes_gpu

    @staticmethod
    def _allocate(buffer_size, dtype, device):
        if dtype == torch.float32:
            return allocate_gpu_float_pointer_array(buffer_size, device.index)
        else:
            return allocate_gpu_double_pointer_array(buffer_size, device.index)

    def __del__(self):
        for key in self.offset.keys():
            mem_free(self.nodes_cpu[key], self.nodes_gpu[key])
