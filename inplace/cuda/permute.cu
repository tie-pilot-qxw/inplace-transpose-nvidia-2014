#include "equations.cuh"
#include "permute.h"
#include "../common/index.cuh"
#include <vector>
#include <set>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include "util.cuh"


namespace inplace {
namespace detail {
    
template<typename Fn>
void scatter_cycles(Fn f, std::vector<int>& heads, std::vector<int>& lens) {
    int len = f.len();
    thrust::counting_iterator<int> i(0);
    std::set<int> unvisited(i, i+len);
    while(!unvisited.empty()) {
        int idx = *unvisited.begin();
        unvisited.erase(unvisited.begin());
        int dest = f(idx);
        if (idx != dest) {
            heads.push_back(idx);
            int start = idx;
            int len = 1;
            //std::cout << "Cycle: " << start << " " << dest << " ";
            while(dest != start) {
                idx = dest;
                unvisited.erase(idx);
                dest = f(idx);
                len++;
                //std::cout << dest << " ";
            }
            //std::cout << std::endl;
            lens.push_back(len);
        }
    }
}


template<typename T, typename F, int U>
__device__ __forceinline__ void unroll_cycle_row_permute(
    F f, row_major_index rm, T* data, int i, int j, int l) {
    
    T src = data[rm(i, j)];
    T loaded[U+1];
    loaded[0] = src;
    for(int k = 0; k < l / U; k++) {
        int rows[U];
#pragma unroll
        for(int x = 0; x < U; x++) {
            i = f(i);
            rows[x] = i;
        }
#pragma unroll
        for(int x = 0; x < U; x++) {
            loaded[x+1] = data[rm(rows[x], j)];
        }
#pragma unroll
        for(int x = 0; x < U; x++) {
            data[rm(rows[x], j)] = loaded[x];
        }
        loaded[0] = loaded[U];
    }
    T tmp = loaded[0];
    for(int k = 0; k < l % U; k++) {
        i = f(i);
        T new_tmp = data[rm(i, j)];
        data[rm(i, j)] = tmp;
        tmp = new_tmp;
    }
    
}

template<typename T, typename F, int U>
__global__ void cycle_row_permute(F f, T* data, int* heads,
                                  int* lens, int n_heads) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = f.n;
    row_major_index rm(f.m, f.n);


    if ((j < n) && (h < n_heads)) {
        int i = heads[h];
        int l = lens[h];
        unroll_cycle_row_permute<T, F, U>(f, rm, data, i, j, l);
    }
}

template<typename T, typename F>
void scatter_permute(F f, int m, int n, T* data, int* tmp) {
    std::vector<int> heads;
    std::vector<int> lens;
    scatter_cycles(f, heads, lens);
    int* d_heads = tmp;
    int* d_lens = tmp + m / 2;
    cudaMemcpy(d_heads, heads.data(), sizeof(int)*heads.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lens, lens.data(), sizeof(int)*lens.size(),
               cudaMemcpyHostToDevice);
    int n_threads_x = 256;
    int n_threads_y = 1024/n_threads_x;
    
    int n_blocks_x = div_up(n, n_threads_x);
    int n_blocks_y = div_up(heads.size(), n_threads_y);
    cycle_row_permute<T, F, 4>
        <<<dim3(n_blocks_x, n_blocks_y),
        dim3(n_threads_x, n_threads_y)>>>
        (f, data, d_heads, d_lens, heads.size());
}


template void scatter_permute(c2r::scatter_postpermuter, int, int, float*, int*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, double*, int*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, int*, int*);
template void scatter_permute(c2r::scatter_postpermuter, int, int, long long*, int*);

template void scatter_permute(r2c::scatter_prepermuter, int, int, float*, int*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, double*, int*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, int*, int*);
template void scatter_permute(r2c::scatter_prepermuter, int, int, long long*, int*);


}
}
