#pragma once

#include "util.cuh"
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include "../common/reduced_math.cuh"
#include "../common/index.cuh"

namespace inplace {
namespace detail {

    template<typename F, typename T>
    void rotate(F f, int m, int n, T* data);

}
}


namespace inplace {
namespace detail {

__device__ __forceinline__
unsigned int ctz(unsigned int x) {
    return __ffs(x) - 1;
}

__device__ __forceinline__
unsigned int gcd(unsigned int x, unsigned int y) {
    if (x == 0) return y;
    if (y == 0) return x;
    unsigned int cf2 = ctz(x | y);
    x >>= ctz(x);
    while (true) {
        y >>= ctz(y);
        if (x == y) break;
        if (x > y) {
            unsigned int t = x; x = y; y = t;
        }
        if (x == 1) break;
        y -= x;
    }
    return x << cf2;
}

template<typename F, typename T>
__global__ void coarse_col_rotate(F fn, reduced_divisor m, int n, T* d) {
    int warp_id = threadIdx.x & 0x1f;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int rotation_amount = fn(fn.master(global_index, warp_id, 32));
    int col = global_index;

    __shared__ T smem[32 * 16];
    
    if ((col < n) && (rotation_amount > 0)) {
        row_major_index rm(m, n);
        int c = gcd(rotation_amount, m.get());
        int l = m.get() / c;
        int inc = m.get() - rotation_amount;
        int smem_write_idx = threadIdx.y * 32 + threadIdx.x;
        int max_col = (l > 16) ? 15 : l - 1;
        int smem_read_col = (threadIdx.y == 0) ? max_col : (threadIdx.y - 1);
        int smem_read_idx = smem_read_col * 32 + threadIdx.x;
        
        for(int b = 0; b < c; b++) {
            int x = threadIdx.y;
            int pos = m.mod(b + x * inc);            
            smem[smem_write_idx] = d[rm(pos, col)];
            __syncthreads();
            T prior = smem[smem_read_idx];
            if (x < l) d[rm(pos, col)] = prior;
            __syncthreads();
            int n_rounds = l / 16;
            for(int i = 1; i < n_rounds; i++) {
                x += blockDim.y;
                int pos = m.mod(b + x * inc);            
                if (x < l) smem[smem_write_idx] = d[rm(pos, col)];
                __syncthreads();
                T incoming = smem[smem_read_idx];
                T outgoing = (threadIdx.y == 0) ? prior : incoming;
                if (x < l) d[rm(pos, col)] = outgoing;
                prior = incoming;
                __syncthreads();
            }
            //Last round/cleanup
            x += blockDim.y;
            pos = m.mod(b + x * inc);
            if (x <= l) smem[smem_write_idx] = d[rm(pos, col)];
            __syncthreads();
            int remainder_length = (l % 16);
            int fin_smem_read_col = (threadIdx.y == 0) ? remainder_length : threadIdx.y - 1;
            int fin_smem_read_idx = fin_smem_read_col * 32 + threadIdx.x;
            T incoming = smem[fin_smem_read_idx];
            T outgoing = (threadIdx.y == 0) ? prior : incoming;
            if (x <= l) d[rm(pos, col)] = outgoing;
            
        }
    }
}



template<typename F, typename T>
__global__ void fine_col_rotate(F fn, int m, int n, T* d) {
    __shared__ T smem[32 * 32]; 

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < n) {
        int warp_id = threadIdx.x & 0x1f;
        int coarse_rotation_amount = fn(fn.master(col, warp_id, 32));
        int overall_rotation_amount = fn(col);
        int fine_rotation_amount = overall_rotation_amount - coarse_rotation_amount;
        if (fine_rotation_amount < 0) fine_rotation_amount += m;
        //If the whole warp is rotating by 0, early exit
        unsigned int warp_vote =  __ballot_sync(0xffffffff, fine_rotation_amount > 0);
        if (warp_vote > 0) {
            int row = threadIdx.y;
            int idx = row * n + col;
            T* read_ptr = d + idx;
        
            int smem_idx = threadIdx.y * 32 + threadIdx.x;

            T first; // = -2;
            if (row < m) first = *read_ptr;

            bool first_phase = (threadIdx.y >= fine_rotation_amount);
            int smem_row = threadIdx.y - fine_rotation_amount;
            if (!first_phase) smem_row += 32;

            int smem_write_idx = smem_row * 32 + threadIdx.x;

            if (first_phase) smem[smem_write_idx] = first;

            T* write_ptr = read_ptr;
            int ptr_inc = 32 * n;
            read_ptr += ptr_inc;
            //Loop over blocks that are guaranteed not to fall off the edge
            for(int i = 0; i < (m / 32) - 1; i++) {
                T tmp = *read_ptr;
                if (!first_phase) smem[smem_write_idx] = tmp;
                __syncthreads();
                *write_ptr = smem[smem_idx];
                __syncthreads();
                if (first_phase) smem[smem_write_idx] = tmp;
                write_ptr = read_ptr;
                read_ptr += ptr_inc;
            }

            //Final block (read_ptr may have fallen off the edge)
            int remainder = m % 32;
            T tmp; // = -3;
            if (threadIdx.y < remainder) tmp = *read_ptr;
            int tmp_dest_row = 32 - fine_rotation_amount + threadIdx.y;
            if ((tmp_dest_row >= 0) && (tmp_dest_row < 32))
                smem[tmp_dest_row * 32 + threadIdx.x] = tmp;
            __syncthreads();
            int first_dest_row = 32 + remainder - fine_rotation_amount + threadIdx.y;
            if ((first_dest_row >= 0) && (first_dest_row < 32))
                smem[first_dest_row * 32 + threadIdx.x] = first;
        
            __syncthreads();
            *write_ptr = smem[smem_idx];
            write_ptr = read_ptr;
            __syncthreads();
            //Final incomplete block
            tmp_dest_row -= 32; first_dest_row -= 32;
            if ((tmp_dest_row >= 0) && (tmp_dest_row < 32))
                smem[tmp_dest_row * 32 + threadIdx.x] = tmp;
            __syncthreads();
            if ((first_dest_row >= 0) && (first_dest_row < 32))
                smem[first_dest_row * 32 + threadIdx.x] = first;
            __syncthreads();
            if (threadIdx.y < remainder) *write_ptr = smem[smem_idx];
        }
    }
}

template<typename F, typename T>
void rotate(F fn, int m, int n, T* data) {
    int n_blocks = div_up(n, 32);
    dim3 block_dim(32, 32);
    if (fn.fine()) {
        fine_col_rotate<<<n_blocks, block_dim>>>(fn, m, n, data);
    }
    coarse_col_rotate<<<n_blocks, dim3(32, 16)>>>(
        fn, m, n, data);
}

}
}
