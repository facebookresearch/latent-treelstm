# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cuda_runtime.h>


#define MAX_NUM_THREADS_PER_BLOCK 256
// TODO(serhii): test speed while increasing/decreasing number of blocks
#define MAX_NUM_BLOCKS_PER_KERNEL 256 // 22 (SMs) * 2048 (Ths) / 256 (Max num of Ths) = 176 (blocks)

#define GRID_STRIDE_1D_LOOP(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define SELECT_KERNEL(assignment_expr_l, assignment_expr_r)                                                        \
    int64_t batch_idx;                                                                                             \
    int64_t element_idx;                                                                                           \
    int64_t action;                                                                                                \
    int64_t is_leaf_l;                                                                                             \
    int64_t is_leaf_r;                                                                                             \
    int64_t idx_l;                                                                                                 \
    int64_t idx_r;                                                                                                 \
    int64_t offset_l;                                                                                              \
    int64_t offset_r;                                                                                              \
                                                                                                                   \
    GRID_STRIDE_1D_LOOP(i, batch_size * node_dim) {                                                                \
        batch_idx = i / node_dim;                                                                                  \
        element_idx = i % node_dim;                                                                                \
        action = actions[batch_idx];                                                                               \
        is_leaf_l = is_leaf[batch_idx * sequence_length + action];                                                 \
        is_leaf_r = is_leaf[batch_idx * sequence_length + action + 1];                                             \
                                                                                                                   \
        /* `idx` is the corresponding coordinate of the sequence dimension if the left/right vector belongs to
            the initial feature tensor. Otherwise, `idx` is an index of the left/right vector in the list of
            reduced tensors. */                                                                                    \
        idx_l = index_mapping[batch_idx * sequence_length + action];                                               \
        idx_r = index_mapping[batch_idx * sequence_length + action + 1];                                           \
                                                                                                                   \
        /* if the source is the initial tensor an additional sequence dimension has to be taken into account */    \
        offset_l = is_leaf_l ? (batch_idx * max_sequence_length + idx_l) * node_dim + element_idx : i;             \
        offset_r = is_leaf_r ? (batch_idx * max_sequence_length + idx_r) * node_dim + element_idx : i;             \
        /* I think there is no difference(performance-wise) between if-else and ternary operator because there is
           no branch divergence within warps as long as node_dim is evenly divisible by warp size (32). */         \
        assignment_expr_l;                                                                                         \
        assignment_expr_r;                                                                                         \
    }

#define SELECT_PARSER_KERNEL(type_modif, name, select_expr_lr, select_expr_c, asgmt_expr_l, asgmt_expr_c, asgmt_expr_r)\
    int64_t batch_idx;                                                                                                 \
    int64_t element_idx;                                                                                               \
    int64_t action;                                                                                                    \
    int64_t column_index;                                                                                              \
    int64_t list_index;                                                                                                \
    int64_t num_cols_in_batch;                                                                                         \
    int64_t offset;                                                                                                    \
    type_modif scalar_t* name;                                                                                         \
    const int64_t NUM_EL = batch_size * sequence_length;                                                               \
    const int64_t NUM_EL_p = batch_size * (sequence_length - 1);                                                       \
                                                                                                                       \
    GRID_STRIDE_1D_LOOP(i, batch_size * node_dim) {                                                                    \
        batch_idx = i / node_dim;                                                                                      \
        element_idx = i % node_dim;                                                                                    \
        action = actions[batch_idx];                                                                                   \
                                                                                                                       \
        /* ============= left node ============= */                                                                    \
        if (action > 0) {                                                                                              \
            column_index = index_mapping[batch_idx * sequence_length + action - 1];                                    \
            list_index = index_mapping[NUM_EL + batch_idx * sequence_length + action - 1];                             \
            num_cols_in_batch = (list_index == -1 || list_index == 0) ? (max_sequence_length - 1 - list_index) : 2;    \
            offset = (batch_idx * num_cols_in_batch + column_index) * node_dim + element_idx;                          \
            select_expr_lr;                                                                                            \
            asgmt_expr_l;                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        /* ============= centre node ============= */                                                                  \
        column_index = index_mapping_p[batch_idx * (sequence_length - 1) + action];                                    \
        list_index = index_mapping_p[NUM_EL_p + batch_idx * (sequence_length - 1) + action];                           \
        num_cols_in_batch = list_index == 0 ? max_sequence_length - 1 : 2;                                             \
        offset = (batch_idx * num_cols_in_batch + column_index) * node_dim + element_idx;                              \
        select_expr_c;                                                                                                 \
        asgmt_expr_c;                                                                                                  \
                                                                                                                       \
        /* ============= right node ============= */                                                                   \
        if (action + 2 < sequence_length) {                                                                            \
            column_index = index_mapping[batch_idx * sequence_length + action + 2];                                    \
            list_index = index_mapping[NUM_EL + batch_idx * sequence_length + action + 2];                             \
            num_cols_in_batch = (list_index == -1 || list_index == 0) ? (max_sequence_length - 1 - list_index) : 2;    \
            offset = (batch_idx * num_cols_in_batch + column_index) * node_dim + element_idx;                          \
            select_expr_lr;                                                                                            \
            asgmt_expr_r;                                                                                              \
        }                                                                                                              \
    }


namespace {
    /*
        Calculates new index mapping (index_mapping, is_leaf) after performing reduction actions.
        Resulting arrays map indexes of the "logical" feature tensor after reduction steps to the "physical" indexes.

        example:
            index_mapping:   0 1 2 3 4           0 1 2 1            0 2 1            0 3            4
                             0 1 2 3 4           1 2 3 4            1 2 2            3 2            4
                                           3                 1                1              0
            actions:                   -------->         --------->       --------->     --------->
                                           0                 2                0              0
            is_leaf:         1 1 1 1 1           1 1 1 0            1 0 0            1 0            0
                             1 1 1 1 1           0 1 1 1            0 1 0            0 0            0

        @param actions stores sampled merging actions. It has Long (int64_t) type
               (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h#L88).
        @param prev_index_mapping contains column index of the initial feature tensor or the index of the reduced tensor
               depending on the prev_is_leaf variable.
        @param prev_is_leaf indicates whether the corresponding node is a leaf or not.
    */
    __global__ void get_indexes_mapping_kernel(const int64_t* __restrict__ actions,
                                               const int64_t* __restrict__ prev_index_mapping,
                                               const uint8_t* __restrict__ prev_is_leaf,
                                               int64_t* __restrict__ index_mapping,
                                               uint8_t* __restrict__ is_leaf,
                                               int64_t batch_size,
                                               int64_t sequence_length,
                                               int64_t step) {
        int64_t batch_idx;
        int64_t node_idx;
        int64_t action;
        int64_t prev_offset;

        GRID_STRIDE_1D_LOOP(offset, batch_size * sequence_length) {
            batch_idx = offset / sequence_length;
            node_idx = offset % sequence_length;

            action = actions[batch_idx];
            prev_offset = offset + batch_idx + (node_idx > action);

            index_mapping[offset] = (action == node_idx) ? step : prev_index_mapping[prev_offset];
            is_leaf[offset] = (action == node_idx) ? 0 : prev_is_leaf[prev_offset];
        }
    }

    /*
        Selects the left and the right nodes.

        @param h is a collection of the tensors to select from.
        @param actions stores sampled merging actions.
        @param index_mapping and is_leaf map "logical" indexes into "physical" ones.
    */
    template <typename scalar_t>
    __global__ void select_forward_kernel(const scalar_t* const* const __restrict__ h,
                                          const int64_t* __restrict__ actions,
                                          const int64_t* __restrict__ index_mapping,
                                          const uint8_t* __restrict__ is_leaf,
                                          scalar_t* __restrict__ node_l,
                                          scalar_t* __restrict__ node_r,
                                          int64_t batch_size,
                                          int64_t sequence_length,
                                          int64_t node_dim,
                                          int64_t max_sequence_length) {
        SELECT_KERNEL(node_l[i] = h[is_leaf_l ? 0 : idx_l][offset_l],
                      node_r[i] = h[is_leaf_r ? 0 : idx_r][offset_r]);
    }

    /*
        Back propagates gradients from the left and the right nodes.

        @param grad_h_l, grad_h_r gradients of the loss function with respect to left and right nodes.
        @param grad_h is a collection of the gradients for feature tensors.
        @param actions stores sampled merging actions.
        @param index_mapping and is_leaf map "logical" indexes into "physical" ones.
    */
    template <typename scalar_t>
    __global__ void select_backward_kernel(const scalar_t* __restrict__ grad_node_l,
                                           const scalar_t* __restrict__ grad_node_r,
                                           const int64_t* __restrict__ actions,
                                           const int64_t* __restrict__ index_mapping,
                                           const uint8_t* __restrict__ is_leaf,
                                           scalar_t** __restrict__ grad_h,
                                           int64_t batch_size,
                                           int64_t sequence_length,
                                           int64_t node_dim,
                                           int64_t max_sequence_length) {
        SELECT_KERNEL(grad_h[is_leaf_l ? 0 : idx_l][offset_l] = grad_node_l[i],
                      grad_h[is_leaf_r ? 0 : idx_r][offset_r] = grad_node_r[i]);
    }

     /*
        Calculates new index mapping after performing parser's step.
        Resulting arrays map "logical" indexes of the feature tensors to the "physical" indexes.

        example:
            index_mapping:     0  1  2  3  4  5       0  1  2  3  5      0  2  3  5      0  L  5      L  5      R
                               0  1  2  3  4  5       0  2  3  4  5      0  2  3  5      0  L  5      0  R      L
                              -1 -1 -1 -1 -1 -1      -1 -1 -1  0 -1      0 -1  0 -1      0  1 -1      3 -1      4
                              -1 -1 -1 -1 -1 -1       0 -1 -1 -1 -1      0 -1  0 -1      0  2 -1      0  3      4

                                                 3                   0               1            0         0
            actions:                           ----->              ----->          ----->       ----->    ----->
                                                 0                   2               1            1         0

            index_mapping_p:    0  1  2  3  4          0  1  L  R         R  L  R         L  R          R        -
                                0  1  2  3  4          R  2  3  4         R  L  R         L  R          L        -
                                0  0  0  0  0          0  0  1  1         2  1  1         3  3          4        -
                                0  0  0  0  0          1  0  0  0         1  2  2         3  3          4        -

        @param actions stores sampled parser's actions.
        @param prev_index_mapping contains node index of the feature tensor and its index in the list.
    */
    __global__ void get_indexes_mapping_parser_kernel(const int64_t* __restrict__ actions,
                                                      const int64_t* __restrict__ prev_index_mapping,
                                                      const int64_t* __restrict__ prev_index_mapping_p,
                                                      int64_t* __restrict__ index_mapping,
                                                      int64_t* __restrict__ index_mapping_p,
                                                      int64_t batch_size,
                                                      int64_t sequence_length,
                                                      int64_t step) {
        int64_t batch_idx;
        int64_t node_idx;
        int64_t action;
        int64_t prev_offset;
        int64_t offset_p;
        int64_t diff;
        const int64_t NUM_EL = batch_size * sequence_length;
        const int64_t NUM_EL_p = batch_size * (sequence_length - 1);

        GRID_STRIDE_1D_LOOP(offset, NUM_EL) {
            batch_idx = offset / sequence_length;
            node_idx = offset % sequence_length;
            action = actions[batch_idx];
            prev_offset = offset + batch_idx + (node_idx > action);

            index_mapping[offset] = (action == node_idx) ? prev_index_mapping_p[offset] :
                                                           prev_index_mapping[prev_offset];
            index_mapping[NUM_EL + offset] = (action == node_idx) ? prev_index_mapping_p[NUM_EL + offset] :
                                                                    prev_index_mapping[NUM_EL + batch_size + prev_offset];
            // TODO(serhii): the performance difference between these two is minuscule (0.02 ms according to nvvp)
//            if (action == node_idx) {
//                index_mapping[offset] = prev_index_mapping_p[offset];
//                index_mapping[NUM_EL + offset] = prev_index_mapping_p[NUM_EL + offset];
//            } else {
//                index_mapping[offset] = prev_index_mapping[prev_offset];
//                index_mapping[NUM_EL + offset] = prev_index_mapping[NUM_EL + batch_size + prev_offset];
//            }

            // the number of threads is bigger than number of elements in this matrix,
            // thus several operations will be repeated twice
            offset_p = offset % NUM_EL_p;
            batch_idx = offset_p / (sequence_length - 1);
            node_idx = offset_p % (sequence_length - 1);
            action = actions[batch_idx];
            prev_offset = offset_p + batch_idx + (node_idx > action);

            diff = action - node_idx;
            index_mapping_p[offset_p] = (diff == 0 || diff == 1) ? 1 - diff : prev_index_mapping_p[prev_offset];
            index_mapping_p[NUM_EL_p + offset_p] = (diff == 0 || diff == 1) ? step  : prev_index_mapping_p[NUM_EL_p + batch_size + prev_offset];
//            if (diff == 0 || diff == 1) {
//                index_mapping_p[offset_p] = 1 - diff; // "L" = 0, "R" = 1
//                index_mapping_p[NUM_EL_p + offset_p] = step;
//            } else {
//                index_mapping_p[offset_p] = prev_index_mapping_p[prev_offset];
//                index_mapping_p[NUM_EL_p + offset_p] = prev_index_mapping_p[NUM_EL_p + batch_size + prev_offset];
//            }
        }
    }

    template <typename scalar_t>
    __global__ void select_parser_forward_kernel(const scalar_t* __restrict__ h,
                                                 const scalar_t* const * const __restrict__ h_p,
                                                 const int64_t* __restrict__ actions,
                                                 const int64_t* __restrict__ index_mapping,
                                                 const int64_t* __restrict__ index_mapping_p,
                                                 scalar_t* __restrict__ node_l,
                                                 scalar_t* __restrict__ node_c,
                                                 scalar_t* __restrict__ node_r,
                                                 int64_t batch_size,
                                                 int64_t sequence_length,
                                                 int64_t node_dim,
                                                 int64_t max_sequence_length) {
        SELECT_PARSER_KERNEL(const, source,
                             source = list_index == -1 ? h : h_p[list_index],
                             source = h_p[list_index],
                             node_l[i] = source[offset],
                             node_c[i] = source[offset],
                             node_r[i] = source[offset]);
    }

    template <typename scalar_t>
    __global__ void select_parser_backward_kernel(const scalar_t* __restrict__ grad_node_l,
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
                                                  int64_t max_sequence_length) {
        SELECT_PARSER_KERNEL(, target,
                             target = list_index == -1 ? grad_h : grad_h_p[list_index],
                             target = grad_h_p[list_index],
                             target[offset] += grad_node_l[i],
                             target[offset] += grad_node_c[i],
                             target[offset] += grad_node_r[i]);
    }

    template <typename scalar_t>
    __global__ void update_scores_forward_kernel(const scalar_t* __restrict__ scores,
                                                 const scalar_t* __restrict__ new_scores,
                                                 const int64_t* __restrict__ actions,
                                                 scalar_t* __restrict__ updated_scores,
                                                 int64_t batch_size,
                                                 int64_t sequence_length) {
        int64_t batch_idx;
        int64_t pair_idx;
        int64_t diff;

        GRID_STRIDE_1D_LOOP(offset, batch_size * sequence_length) {
            batch_idx = offset / sequence_length;
            pair_idx = offset % sequence_length;

            diff = actions[batch_idx] - pair_idx;
            // when diff == 1 assign left/[:, 0] score if diff == 0 assign right/[:, 1] score
            updated_scores[offset] = (diff == 0 || diff == 1) ? new_scores[batch_idx * 2 + 1 - diff] :
                                                                scores[offset + batch_idx + (diff < 0)];
        }
    }

    template <typename scalar_t>
    __global__ void update_scores_backward_kernel(const scalar_t* __restrict__ grad_updated_scores,
                                                  const int64_t* __restrict__ actions,
                                                  scalar_t* __restrict__ grad_scores,
                                                  scalar_t* __restrict__ grad_new_scores,
                                                  int64_t batch_size,
                                                  int64_t sequence_length) {
        int64_t batch_idx;
        int64_t pair_idx;
        int64_t diff;
        int64_t grad_offset;
        scalar_t* grad;

        GRID_STRIDE_1D_LOOP(offset, batch_size * sequence_length) {
            batch_idx = offset / sequence_length;
            pair_idx = offset % sequence_length;

            diff = actions[batch_idx] - pair_idx;
            grad = (diff == 0 || diff == 1) ? grad_new_scores : grad_scores;
            grad_offset = (diff == 0 || diff == 1) ? batch_idx * 2 + 1 - diff : offset + batch_idx + (diff < 0);
            grad[grad_offset] = grad_updated_scores[offset];

//            TODO(serhii): check the speed
//            if (diff == 0 || diff == 1) {
//                grad_new_scores[batch_idx * 2 + 1 - diff] = grad_updated_scores[offset];
//            } else {
//                grad_scores[offset + batch_idx + (diff < 0)] = grad_updated_scores[offset];
//            }
        }
    }
}


cudaError_t get_indexes_mapping_wrapper(const int64_t* __restrict__ actions,
                                        const int64_t* __restrict__ prev_index_mapping,
                                        const uint8_t* __restrict__ prev_is_leaf,
                                        int64_t* __restrict__ index_mapping,
                                        uint8_t* __restrict__ is_leaf,
                                        int64_t batch_size,
                                        int64_t sequence_length,
                                        int64_t step) {
    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL,
                              (int)(batch_size * sequence_length - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
    get_indexes_mapping_kernel<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK>>>(actions,
                                                                          prev_index_mapping,
                                                                          prev_is_leaf,
                                                                          index_mapping,
                                                                          is_leaf,
                                                                          batch_size,
                                                                          sequence_length,
                                                                          step);
    return cudaPeekAtLastError();
}

template <typename scalar_t>
cudaError_t select_forward_wrapper(const scalar_t* const * const __restrict__ h,
                                   const int64_t* __restrict__ actions,
                                   const int64_t* __restrict__ index_mapping,
                                   const uint8_t* __restrict__ is_leaf,
                                   scalar_t* __restrict__ node_l,
                                   scalar_t* __restrict__ node_r,
                                   int64_t batch_size,
                                   int64_t sequence_length,
                                   int64_t node_dim,
                                   int64_t max_sequence_length) {
    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL,
                              (int)(batch_size * node_dim - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
    select_forward_kernel<scalar_t><<<num_blocks, MAX_NUM_THREADS_PER_BLOCK>>>(h,
                                                                               actions,
                                                                               index_mapping,
                                                                               is_leaf,
                                                                               node_l,
                                                                               node_r,
                                                                               batch_size,
                                                                               sequence_length,
                                                                               node_dim,
                                                                               max_sequence_length);
    return cudaPeekAtLastError();
}

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
                                    int64_t max_sequence_length) {
    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL,
                              (int)(batch_size * node_dim - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
    select_backward_kernel<scalar_t><<<num_blocks, MAX_NUM_THREADS_PER_BLOCK>>>(grad_node_l,
                                                                                grad_node_r,
                                                                                actions,
                                                                                index_mapping,
                                                                                is_leaf,
                                                                                grad_h,
                                                                                batch_size,
                                                                                sequence_length,
                                                                                node_dim,
                                                                                max_sequence_length);
    return cudaPeekAtLastError();
}

cudaError_t get_indexes_mapping_parser_wrapper(const int64_t* __restrict__ actions,
                                               const int64_t* __restrict__ prev_index_mapping,
                                               const int64_t* __restrict__ prev_index_mapping_p,
                                               int64_t* __restrict__ index_mapping,
                                               int64_t* __restrict__ index_mapping_p,
                                               int64_t batch_size,
                                               int64_t sequence_length,
                                               int64_t step) {
    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL,
                              (int)(batch_size * sequence_length - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
    get_indexes_mapping_parser_kernel<<<num_blocks, MAX_NUM_THREADS_PER_BLOCK>>>(actions,
                                                                                 prev_index_mapping,
                                                                                 prev_index_mapping_p,
                                                                                 index_mapping,
                                                                                 index_mapping_p,
                                                                                 batch_size,
                                                                                 sequence_length,
                                                                                 step);
    return cudaPeekAtLastError();
}


template <typename scalar_t>
cudaError_t select_parser_forward_wrapper(const scalar_t* __restrict__ h,
                                          const scalar_t* const * const __restrict__ h_p,
                                          const int64_t* __restrict__ actions,
                                          const int64_t* __restrict__ index_mapping,
                                          const int64_t* __restrict__ index_mapping_p,
                                          scalar_t* __restrict__ node_l,
                                          scalar_t* __restrict__ node_c,
                                          scalar_t* __restrict__ node_r,
                                          int64_t batch_size,
                                          int64_t sequence_length,
                                          int64_t node_dim,
                                          int64_t max_sequence_length) {
    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL,
                              (int)(batch_size * node_dim - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
    select_parser_forward_kernel<scalar_t> <<<num_blocks, MAX_NUM_THREADS_PER_BLOCK>>>(h, h_p, actions,
                                                                                       index_mapping,
                                                                                       index_mapping_p,
                                                                                       node_l, node_c, node_r,
                                                                                       batch_size, sequence_length,
                                                                                       node_dim, max_sequence_length);
    return cudaPeekAtLastError();
}


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
                                           int64_t max_sequence_length) {
    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL,
                              (int)(batch_size * node_dim - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
    select_parser_backward_kernel<scalar_t> <<<num_blocks, MAX_NUM_THREADS_PER_BLOCK>>>(grad_node_l,
                                                                                        grad_node_c,
                                                                                        grad_node_r,
                                                                                        actions,
                                                                                        index_mapping, index_mapping_p,
                                                                                        grad_h, grad_h_p,
                                                                                        batch_size, sequence_length,
                                                                                        node_dim, max_sequence_length);
    return cudaPeekAtLastError();
}



template <typename scalar_t>
cudaError_t update_scores_forward_wrapper(const scalar_t* __restrict__ scores,
                                          const scalar_t* __restrict__ new_scores,
                                          const int64_t* __restrict__ actions,
                                          scalar_t* __restrict__ updated_scores,
                                          int64_t batch_size,
                                          int64_t sequence_length) {
    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL,
                              (int)(batch_size * sequence_length - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
    update_scores_forward_kernel<scalar_t><<<num_blocks, MAX_NUM_THREADS_PER_BLOCK>>>(scores,
                                                                                      new_scores,
                                                                                      actions,
                                                                                      updated_scores,
                                                                                      batch_size,
                                                                                      sequence_length);
    return cudaPeekAtLastError();
}


template <typename scalar_t>
cudaError_t update_scores_backward_wrapper(const scalar_t* __restrict__ grad_updated_scores,
                                           const int64_t* __restrict__ actions,
                                           scalar_t* __restrict__ grad_scores,
                                           scalar_t* __restrict__ grad_new_scores,
                                           int64_t batch_size,
                                           int64_t sequence_length) {
    int num_blocks = std::min(MAX_NUM_BLOCKS_PER_KERNEL,
                              (int)(batch_size * sequence_length - 1) / MAX_NUM_THREADS_PER_BLOCK + 1);
    update_scores_backward_kernel<scalar_t><<<num_blocks, MAX_NUM_THREADS_PER_BLOCK>>>(grad_updated_scores,
                                                                                       actions,
                                                                                       grad_scores,
                                                                                       grad_new_scores,
                                                                                       batch_size,
                                                                                       sequence_length);
    return cudaPeekAtLastError();
}

template cudaError_t select_forward_wrapper<double>(const double * const * const __restrict__, const int64_t * __restrict__, const int64_t * __restrict__, const uint8_t * __restrict__, double  * __restrict__, double  * __restrict__, int64_t, int64_t, int64_t, int64_t);
template cudaError_t select_forward_wrapper<float>(const float * const * const __restrict__, const int64_t * __restrict__, const int64_t * __restrict__, const uint8_t * __restrict__, float * __restrict__, float * __restrict__, int64_t, int64_t, int64_t, int64_t);
template cudaError_t select_backward_wrapper<double>(const double  * __restrict__, const double  * __restrict__, const int64_t * __restrict__, const int64_t * __restrict__, const uint8_t * __restrict__, double ** __restrict__, int64_t, int64_t, int64_t, int64_t);
template cudaError_t select_backward_wrapper<float>(const float * __restrict__, const float * __restrict__, const int64_t * __restrict__, const int64_t * __restrict__, const uint8_t * __restrict__, float **  __restrict__, int64_t, int64_t, int64_t, int64_t);
template cudaError_t select_parser_forward_wrapper<double>(const double* __restrict__, const double* const * const __restrict__, const int64_t* __restrict__, const int64_t* __restrict__, const int64_t* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__, int64_t, int64_t, int64_t, int64_t);
template cudaError_t select_parser_forward_wrapper<float>(const float* __restrict__, const float* const * const __restrict__, const int64_t* __restrict__, const int64_t* __restrict__, const int64_t* __restrict__, float* __restrict__, float* __restrict__, float* __restrict__, int64_t, int64_t, int64_t, int64_t);
template cudaError_t select_parser_backward_wrapper<double>(const double* __restrict__, const double* __restrict__, const double* __restrict__, const int64_t* __restrict__, const int64_t* __restrict__, const int64_t* __restrict__, double* __restrict__, double** __restrict__, int64_t, int64_t, int64_t, int64_t);
template cudaError_t select_parser_backward_wrapper<float>(const float* __restrict__, const float* __restrict__, const float* __restrict__, const int64_t* __restrict__, const int64_t* __restrict__, const int64_t* __restrict__, float* __restrict__, float** __restrict__, int64_t, int64_t, int64_t, int64_t);
template cudaError_t update_scores_forward_wrapper<double>(const double* __restrict__, const double* __restrict__, const int64_t* __restrict__, double* __restrict__, int64_t, int64_t);
template cudaError_t update_scores_forward_wrapper<float>(const float* __restrict__, const float* __restrict__, const int64_t* __restrict__, float* __restrict__, int64_t, int64_t);
template cudaError_t update_scores_backward_wrapper<double>(const double* __restrict__, const int64_t* __restrict__, double* __restrict__, double* __restrict__, int64_t, int64_t);
template cudaError_t update_scores_backward_wrapper<float>(const float* __restrict__, const int64_t* __restrict__, float* __restrict__, float* __restrict__, int64_t, int64_t);


//            printf("i: %d, batch_idx: %d, node_idx: %d, action: %d, is_leaf_l: %d, is_leaf_r: %d, idx_l: %d, idx_r: %d, offset_l: %d offset_r: %d node_dim: %d sequence_length %d, value: %f\n",
//            (int)i, (int)batch_idx, (int)node_idx, (int)action, (int)is_leaf_l, (int)is_leaf_r, (int)idx_l, (int)idx_r, (int)offset_l, (int)offset_r, (int)node_dim, (int)sequence_length, (float)h[is_leaf_l ? 0 : idx_l][offset_l]);
