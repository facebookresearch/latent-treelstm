# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#include <vector>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/Exceptions.h>


#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_BYTE_TYPE(x) AT_CHECK(x.type().scalarType() == at::ScalarType::Byte, #x " must be a ByteTensor")
#define CHECK_LONG_TYPE(x) AT_CHECK(x.type().scalarType() == at::ScalarType::Long, #x " must be a LongTensor")

#define CUDA_DEVICE_CONTEXT_ENTER(device_id)             \
    int device_index;                                    \
    AT_CUDA_CHECK(cudaGetDevice(&device_index));         \
    AT_CUDA_CHECK(cudaSetDevice(device_id))              \

#define CUDA_DEVICE_CONTEXT_EXIT() AT_CUDA_CHECK(cudaSetDevice(device_index))


cudaError_t get_indexes_mapping_wrapper(const int64_t* __restrict__ actions,
                                        const int64_t* __restrict__ prev_index_mapping,
                                        const uint8_t* __restrict__ prev_is_leaf,
                                        int64_t* __restrict__ index_mapping,
                                        uint8_t* __restrict__ is_leaf,
                                        int64_t batch_size,
                                        int64_t sequence_length,
                                        int64_t step);

template <typename scalar_t>
cudaError_t select_forward_wrapper(const scalar_t* const * const  __restrict__ h,
                                   const int64_t* __restrict__ actions,
                                   const int64_t* __restrict__ index_mapping,
                                   const uint8_t* __restrict__ is_leaf,
                                   scalar_t* __restrict__ node_l,
                                   scalar_t* __restrict__ node_r,
                                   int64_t batch_size,
                                   int64_t sequence_length,
                                   int64_t node_dim,
                                   int64_t max_sequence_length);

template <typename scalar_t>
cudaError_t select_backward_wrapper(const scalar_t* __restrict__ grad_node_l,
                                    const scalar_t* __restrict__ grad_node_r,
                                    const int64_t* __restrict__ actions,
                                    const int64_t* __restrict__ index_mapping,
                                    const uint8_t* __restrict__ is_leaf,
                                    scalar_t** __restrict__ grad_h,
                                    int64_t batch_size,
                                    int64_t sequence_length,
                                    int64_t node_dim,
                                    int64_t max_sequence_length);

cudaError_t get_indexes_mapping_parser_wrapper(const int64_t* __restrict__ actions,
                                               const int64_t* __restrict__ prev_index_mapping,
                                               const int64_t* __restrict__ prev_index_mapping_p,
                                               int64_t* __restrict__ index_mapping,
                                               int64_t* __restrict__ index_mapping_p,
                                               int64_t batch_size,
                                               int64_t sequence_length,
                                               int64_t step);

template <typename scalar_t>
cudaError_t select_parser_forward_wrapper(const scalar_t* __restrict__ h,
                                          const scalar_t* const * const  __restrict__ h_p,
                                          const int64_t* __restrict__ actions,
                                          const int64_t* __restrict__ index_mapping,
                                          const int64_t* __restrict__ index_mapping_p,
                                          scalar_t* __restrict__ node_l,
                                          scalar_t* __restrict__ node_c,
                                          scalar_t* __restrict__ node_r,
                                          int64_t batch_size,
                                          int64_t sequence_length,
                                          int64_t node_dim,
                                          int64_t max_sequence_length);

template <typename scalar_t>
cudaError_t select_parser_backward_wrapper(const scalar_t* __restrict__ grad_node_l,
                                           const scalar_t* __restrict__ grad_node_c,
                                           const scalar_t* __restrict__ grad_node_r,
                                           const int64_t* __restrict__ actions,
                                           const int64_t* __restrict__ index_mapping,
                                           const int64_t* __restrict__ index_mapping_p,
                                           scalar_t* __restrict__ grad_h,
                                           scalar_t** __restrict__ grad_h_p,
                                           int64_t batch_size,
                                           int64_t sequence_length,
                                           int64_t node_dim,
                                           int64_t max_sequence_length);

template <typename scalar_t>
cudaError_t update_scores_forward_wrapper(const scalar_t* __restrict__ scores,
                                          const scalar_t* __restrict__ new_scores,
                                          const int64_t* __restrict__ actions,
                                          scalar_t* __restrict__ updated_scores,
                                          int64_t batch_size,
                                          int64_t sequence_length);

template <typename scalar_t>
cudaError_t update_scores_backward_wrapper(const scalar_t* __restrict__ grad_updated_scores,
                                           const int64_t* __restrict__ actions,
                                           scalar_t* __restrict__ grad_scores,
                                           scalar_t* __restrict__ grad_new_scores,
                                           int64_t batch_size,
                                           int64_t sequence_length);


template <typename scalar_t>
std::vector<uint64_t> allocate_gpu_pointer_array(int buffer_size, int device) {
    scalar_t** h_cpu;
    scalar_t** h_gpu;
    AT_CUDA_CHECK(cudaMallocHost(&h_cpu, buffer_size * sizeof(scalar_t*)));
    CUDA_DEVICE_CONTEXT_ENTER(device);
    AT_CUDA_CHECK(cudaMalloc(&h_gpu, buffer_size * sizeof(scalar_t*)));
    CUDA_DEVICE_CONTEXT_EXIT();
    return {reinterpret_cast<uint64_t>(h_cpu), reinterpret_cast<uint64_t>(h_gpu)};
}


void mem_free(uint64_t h_cpu, uint64_t h_gpu) {
    AT_CUDA_CHECK(cudaFreeHost(reinterpret_cast<void**>(h_cpu)));
    AT_CUDA_CHECK(cudaFree(reinterpret_cast<void**>(h_gpu)));
}


template <typename scalar_t>
void transfer_last_to_gpu(std::vector<torch::Tensor> &h, scalar_t** h_cpu, scalar_t** h_gpu) {
    auto size = h.size();
    h_cpu[size - 1] = h.back().data<scalar_t>();
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_gpu[size - 1], &h_cpu[size - 1], sizeof(scalar_t*), cudaMemcpyHostToDevice));
}


template <typename scalar_t>
void transfer_all_to_gpu(std::vector<torch::Tensor> &h, scalar_t** h_cpu, scalar_t** h_gpu) {
    for(size_t i = 0; i != h.size(); i++) {
        h_cpu[i] = h[i].data<scalar_t>();
    }
    AT_CUDA_CHECK(cudaMemcpyAsync(h_gpu, h_cpu, h.size() * sizeof(scalar_t*), cudaMemcpyHostToDevice));
}


std::vector<torch::Tensor> get_indexes_mapping(torch::Tensor actions,
                                               torch::Tensor prev_index_mapping,
                                               torch::Tensor prev_is_leaf,
                                               int64_t step) {
    // TODO(serhii): Probably these checks should be disabled after debugging.
    CHECK_INPUT(actions);
    CHECK_LONG_TYPE(actions)
    CHECK_INPUT(prev_index_mapping);
    CHECK_LONG_TYPE(prev_index_mapping)
    CHECK_INPUT(prev_is_leaf);
    CHECK_BYTE_TYPE(prev_is_leaf);

    CUDA_DEVICE_CONTEXT_ENTER(actions.get_device());

    const auto batch_size = prev_is_leaf.size(0);
    const auto sequence_length = prev_is_leaf.size(1) - 1;
    auto index_mapping = torch::empty({batch_size, sequence_length}, prev_index_mapping.options());
    auto is_leaf = torch::empty({batch_size, sequence_length}, prev_is_leaf.options());

    AT_CUDA_CHECK(get_indexes_mapping_wrapper(actions.data<int64_t>(),
                                              prev_index_mapping.data<int64_t>(),
                                              prev_is_leaf.data<uint8_t>(),
                                              index_mapping.data<int64_t>(),
                                              is_leaf.data<uint8_t>(),
                                              batch_size,
                                              sequence_length,
                                              step));

    CUDA_DEVICE_CONTEXT_EXIT();
    return {index_mapping, is_leaf};
}


std::vector<torch::Tensor> select_forward(std::vector<torch::Tensor> h,
                                          uint64_t _h_cpu,
                                          uint64_t _h_gpu,
                                          torch::Tensor actions,
                                          torch::Tensor index_mapping,
                                          torch::Tensor is_leaf) {
    // TODO(serhii): Probably these checks should be disabled after debugging.
    for(size_t i = 0; i != h.size(); i++) {
        CHECK_INPUT(h[i]);
    }
    CHECK_INPUT(actions);
    CHECK_LONG_TYPE(actions)
    CHECK_INPUT(index_mapping);
    CHECK_LONG_TYPE(index_mapping)
    CHECK_INPUT(is_leaf);
    CHECK_BYTE_TYPE(is_leaf);

    CUDA_DEVICE_CONTEXT_ENTER(actions.get_device());

    auto node_shape = h.front().sizes().vec();
    node_shape.erase(node_shape.begin() + 1);  //remove sequence_length dimension
    const auto batch_size = h.front().size(0);
    const auto sequence_length = is_leaf.size(1);
    const auto max_sequence_length = h.front().size(1);
    const auto node_dim = h.front().numel() / (max_sequence_length * batch_size);

    auto node_l = torch::empty(node_shape, h.front().options());
    auto node_r = torch::empty(node_shape, h.front().options());
    AT_DISPATCH_FLOATING_TYPES(h.front().type(), "select_forward", ([&] {
        scalar_t** h_cpu = reinterpret_cast<scalar_t**>(_h_cpu);
        scalar_t** h_gpu = reinterpret_cast<scalar_t**>(_h_gpu);
        transfer_last_to_gpu<scalar_t>(h, h_cpu, h_gpu);
        AT_CUDA_CHECK(select_forward_wrapper<scalar_t>(h_gpu,
                                                       actions.data<int64_t>(),
                                                       index_mapping.data<int64_t>(),
                                                       is_leaf.data<uint8_t>(),
                                                       node_l.data<scalar_t>(),
                                                       node_r.data<scalar_t>(),
                                                       batch_size, sequence_length, node_dim, max_sequence_length));
    }));

    CUDA_DEVICE_CONTEXT_EXIT();
    return {node_l, node_r};
}


void select_backward(std::vector<torch::Tensor> grad_h,
                     uint64_t _grad_h_cpu,
                     uint64_t _grad_h_gpu,
                     torch::Tensor grad_node_l,
                     torch::Tensor grad_node_r,
                     torch::Tensor actions,
                     torch::Tensor index_mapping,
                     torch::Tensor is_leaf) {
    // TODO(serhii): Probably these checks should be disabled after debugging.
    for(size_t i = 0; i != grad_h.size(); i++) {
        CHECK_INPUT(grad_h[i]);
    }
    CHECK_INPUT(actions);
    CHECK_LONG_TYPE(actions)
    CHECK_INPUT(index_mapping);
    CHECK_LONG_TYPE(index_mapping)
    CHECK_INPUT(is_leaf);
    CHECK_BYTE_TYPE(is_leaf);
    CHECK_INPUT(grad_node_l);
    CHECK_INPUT(grad_node_r);

    CUDA_DEVICE_CONTEXT_ENTER(actions.get_device());

    const auto batch_size = is_leaf.size(0);
    const auto sequence_length = is_leaf.size(1);
    const auto node_dim = grad_h[0].size(2);
    const auto max_sequence_length = grad_h[0].size(1);

    AT_DISPATCH_FLOATING_TYPES(grad_h.front().type(), "select_backward", ([&] {
        scalar_t** grad_h_cpu = reinterpret_cast<scalar_t**>(_grad_h_cpu);
        scalar_t** grad_h_gpu = reinterpret_cast<scalar_t**>(_grad_h_gpu);
        //  TODO(serhii): This is a source of potential bugs if the last hidden state is not involved in the loss
        if (index_mapping.size(1) == 2) {
            transfer_all_to_gpu<scalar_t>(grad_h, grad_h_cpu, grad_h_gpu);
        }
        AT_CUDA_CHECK(select_backward_wrapper<scalar_t>(grad_node_l.data<scalar_t>(),
                                                        grad_node_r.data<scalar_t>(),
                                                        actions.data<int64_t>(),
                                                        index_mapping.data<int64_t>(),
                                                        is_leaf.data<uint8_t>(),
                                                        grad_h_gpu,
                                                        batch_size, sequence_length, node_dim, max_sequence_length));
    }));

    CUDA_DEVICE_CONTEXT_EXIT();
    return;
}


std::vector<torch::Tensor> get_indexes_mapping_parser(torch::Tensor actions,
                                                      torch::Tensor prev_index_mapping,
                                                      torch::Tensor prev_index_mapping_p,
                                                      int64_t step) {
    // TODO(serhii): Probably these checks should be disabled after debugging.
    CHECK_INPUT(actions);
    CHECK_LONG_TYPE(actions)
    CHECK_INPUT(prev_index_mapping);
    CHECK_LONG_TYPE(prev_index_mapping)
    CHECK_INPUT(prev_index_mapping_p);
    CHECK_LONG_TYPE(prev_index_mapping_p);

    CUDA_DEVICE_CONTEXT_ENTER(actions.get_device());

    const auto batch_size = actions.size(0);
    const auto sequence_length = prev_index_mapping.size(1) - 1;
    auto index_mapping = torch::empty({2 * batch_size, sequence_length}, prev_index_mapping.options());
    auto index_mapping_p = torch::empty({2 * batch_size, sequence_length - 1}, prev_index_mapping.options());

    AT_CUDA_CHECK(get_indexes_mapping_parser_wrapper(actions.data<int64_t>(),
                                                     prev_index_mapping.data<int64_t>(),
                                                     prev_index_mapping_p.data<int64_t>(),
                                                     index_mapping.data<int64_t>(),
                                                     index_mapping_p.data<int64_t>(),
                                                     batch_size,
                                                     sequence_length,
                                                     step));

    CUDA_DEVICE_CONTEXT_EXIT();
    return {index_mapping, index_mapping_p};
}


std::vector<torch::Tensor> select_parser_forward(torch::Tensor h,
                                                 std::vector<torch::Tensor> h_p,
                                                 uint64_t _h_p_cpu,
                                                 uint64_t _h_p_gpu,
                                                 torch::Tensor actions,
                                                 torch::Tensor index_mapping,
                                                 torch::Tensor index_mapping_p) {
    for(size_t i = 0; i != h_p.size(); i++) {
        CHECK_INPUT(h_p[i]);
    }
    CHECK_INPUT(h);
    CHECK_INPUT(actions);
    CHECK_LONG_TYPE(actions)
    CHECK_INPUT(index_mapping);
    CHECK_LONG_TYPE(index_mapping)
    CHECK_INPUT(index_mapping_p);
    CHECK_LONG_TYPE(index_mapping_p);

    CUDA_DEVICE_CONTEXT_ENTER(actions.get_device());

    auto node_shape = h.sizes().vec();
    node_shape.erase(node_shape.begin() + 1);  //remove sequence_length dimension
    const auto batch_size = h.size(0);
    const auto sequence_length = index_mapping.size(1);
    const auto max_sequence_length = h.size(1);
    const auto node_dim = h.numel() / (max_sequence_length * batch_size);

    //  It may be that empty tensors can cause NaN. So, using masking won't suffice, hence we use zeros method.
    auto node_l = torch::zeros(node_shape, h.options());
    auto node_c = torch::zeros(node_shape, h.options());
    auto node_r = torch::zeros(node_shape, h.options());
    AT_DISPATCH_FLOATING_TYPES(h.type(), "select_parser_forward", ([&] {
        scalar_t** h_p_cpu = reinterpret_cast<scalar_t**>(_h_p_cpu);
        scalar_t** h_p_gpu = reinterpret_cast<scalar_t**>(_h_p_gpu);
        transfer_last_to_gpu<scalar_t>(h_p, h_p_cpu, h_p_gpu);
        AT_CUDA_CHECK(select_parser_forward_wrapper<scalar_t>(h.data<scalar_t>(),
                                                              h_p_gpu,
                                                              actions.data<int64_t>(),
                                                              index_mapping.data<int64_t>(),
                                                              index_mapping_p.data<int64_t>(),
                                                              node_l.data<scalar_t>(),
                                                              node_c.data<scalar_t>(),
                                                              node_r.data<scalar_t>(),
                                                              batch_size, sequence_length,
                                                              node_dim, max_sequence_length));
    }));

    CUDA_DEVICE_CONTEXT_EXIT();
    return {node_l, node_c, node_r};
}

void select_parser_backward(torch::Tensor grad_node_l,
                            torch::Tensor grad_node_c,
                            torch::Tensor grad_node_r,
                            torch::Tensor grad_h,
                            std::vector<torch::Tensor> grad_h_p,
                            uint64_t _grad_h_p_cpu,
                            uint64_t _grad_h_p_gpu,
                            torch::Tensor actions,
                            torch::Tensor index_mapping,
                            torch::Tensor index_mapping_p) {
    CHECK_INPUT(grad_node_l);
    CHECK_INPUT(grad_node_c);
    CHECK_INPUT(grad_node_r);
    CHECK_INPUT(grad_h);
    for(size_t i = 0; i != grad_h_p.size(); i++) {
        CHECK_INPUT(grad_h_p[i]);
    }
    CHECK_INPUT(actions);
    CHECK_LONG_TYPE(actions)
    CHECK_INPUT(index_mapping);
    CHECK_LONG_TYPE(index_mapping)
    CHECK_INPUT(index_mapping_p);
    CHECK_LONG_TYPE(index_mapping_p);

    CUDA_DEVICE_CONTEXT_ENTER(actions.get_device());

    const auto batch_size = grad_h.size(0);
    const auto sequence_length = index_mapping.size(1);
    const auto max_sequence_length = grad_h.size(1);
    const auto node_dim = grad_h.numel() / (max_sequence_length * batch_size);


    AT_DISPATCH_FLOATING_TYPES(grad_h.type(), "select_parser_backward", ([&] {
        scalar_t** grad_h_p_cpu = reinterpret_cast<scalar_t**>(_grad_h_p_cpu);
        scalar_t** grad_h_p_gpu = reinterpret_cast<scalar_t**>(_grad_h_p_gpu);
        //  TODO(serhii): This is a source of potential bugs if the last hidden state is not involved in loss
        if (index_mapping_p.size(1) == 2) {
            transfer_all_to_gpu<scalar_t>(grad_h_p, grad_h_p_cpu, grad_h_p_gpu);
        }
        AT_CUDA_CHECK(select_parser_backward_wrapper<scalar_t>(grad_node_l.data<scalar_t>(),
                                                               grad_node_c.data<scalar_t>(),
                                                               grad_node_r.data<scalar_t>(),
                                                               actions.data<int64_t>(),
                                                               index_mapping.data<int64_t>(),
                                                               index_mapping_p.data<int64_t>(),
                                                               grad_h.data<scalar_t>(),
                                                               grad_h_p_gpu,
                                                               batch_size, sequence_length,
                                                               node_dim, max_sequence_length));
    }));

    CUDA_DEVICE_CONTEXT_EXIT();
    return;
}


torch::Tensor update_scores_forward(torch::Tensor scores,
                                    torch::Tensor new_scores,
                                    torch::Tensor actions) {
    CHECK_INPUT(scores);
    CHECK_INPUT(new_scores);
    CHECK_INPUT(actions);
    CHECK_LONG_TYPE(actions)

    CUDA_DEVICE_CONTEXT_ENTER(actions.get_device());

    const auto batch_size = scores.size(0);
    const auto sequence_length = scores.size(1) - 1;
    auto updated_scores = torch::empty({batch_size, sequence_length}, scores.options());

    AT_DISPATCH_FLOATING_TYPES(scores.type(), "update_scores_forward", ([&] {
        AT_CUDA_CHECK(update_scores_forward_wrapper<scalar_t>(scores.data<scalar_t>(),
                                                              new_scores.data<scalar_t>(),
                                                              actions.data<int64_t>(),
                                                              updated_scores.data<scalar_t>(),
                                                              batch_size, sequence_length));
    }));

    CUDA_DEVICE_CONTEXT_EXIT();
    return updated_scores;
}


std::vector<torch::Tensor> update_scores_backward(torch::Tensor grad_updated_scores,
                                                  torch::Tensor actions) {
    CHECK_INPUT(grad_updated_scores);
    CHECK_INPUT(actions);
    CHECK_LONG_TYPE(actions)

    CUDA_DEVICE_CONTEXT_ENTER(actions.get_device());

    const auto batch_size = grad_updated_scores.size(0);
    const auto sequence_length = grad_updated_scores.size(1);
    auto grad_scores = torch::zeros({batch_size, sequence_length + 1}, grad_updated_scores.options());
    auto grad_new_scores = torch::zeros({batch_size, 2}, grad_updated_scores.options());

    AT_DISPATCH_FLOATING_TYPES(grad_updated_scores.type(), "update_scores_backward", ([&] {
        AT_CUDA_CHECK(update_scores_backward_wrapper<scalar_t>(grad_updated_scores.data<scalar_t>(),
                                                               actions.data<int64_t>(),
                                                               grad_scores.data<scalar_t>(),
                                                               grad_new_scores.data<scalar_t>(),
                                                               batch_size, sequence_length));
    }));

    CUDA_DEVICE_CONTEXT_EXIT();
    return {grad_scores, grad_new_scores};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allocate_gpu_double_pointer_array", &allocate_gpu_pointer_array<double>);
  m.def("allocate_gpu_float_pointer_array", &allocate_gpu_pointer_array<float>);
  m.def("mem_free", &mem_free);
  m.def("get_indexes_mapping", &get_indexes_mapping, "Calculates new indexes mapping (CUDA)");
  m.def("select_forward", &select_forward, "Selects the left and the right nodes (CUDA)");
  m.def("select_backward", &select_backward, "Backward function of select function (CUDA)");
  m.def("get_indexes_mapping_parser", &get_indexes_mapping_parser, "Calculates new indexes mapping (CUDA)");
  m.def("select_parser_forward", &select_parser_forward, "Selects nodes for recalculating parser's scores (CUDA)");
  m.def("select_parser_backward", &select_parser_backward, "Backward function of parser's select function (CUDA)");
  m.def("update_scores_forward", &update_scores_forward, "Updates scores by inserting new scores (CUDA)");
  m.def("update_scores_backward", &update_scores_backward, "Backward function of update_scores function (CUDA)");
}
