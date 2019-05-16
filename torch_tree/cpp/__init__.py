# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils import cpp_extension


cuda_module = cpp_extension.load(name="cuda_module", sources=["torch_tree/cpp/extension.cu",
                                                              "torch_tree/cpp/extension.cpp"])
allocate_gpu_double_pointer_array = cuda_module.allocate_gpu_double_pointer_array
allocate_gpu_float_pointer_array = cuda_module.allocate_gpu_float_pointer_array
mem_free = cuda_module.mem_free
get_indexes_mapping = cuda_module.get_indexes_mapping
get_indexes_mapping_parser = cuda_module.get_indexes_mapping_parser
select_forward = cuda_module.select_forward
select_backward = cuda_module.select_backward
select_parser_forward = cuda_module.select_parser_forward
select_parser_backward = cuda_module.select_parser_backward
update_scores_forward = cuda_module.update_scores_forward
update_scores_backward = cuda_module.update_scores_backward
