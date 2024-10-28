#include "equations.cuh"
#include "../common/index.cuh"
#include "smem.cuh"

namespace inplace {
namespace detail {

template<typename T, typename F>
__global__ void smem_row_shuffle(int m, int n, T* d, F s) {
    T* shared_row = shared_memory<T>();
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        row_major_index rm(m, n);
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            shared_row[j] = d[rm(i, j)];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[rm(i, j)] = shared_row[s(j)];
        }
        __syncthreads();
    }        
}

//Work around nvcc/clang bug on OS X
#ifndef __clang__

template __global__ void smem_row_shuffle(int m, int n, float* d, c2r::shuffle s);
template __global__ void smem_row_shuffle(int m, int n, double* d, c2r::shuffle s);

template __global__ void smem_row_shuffle(int m, int n, int* d, c2r::shuffle s);
template __global__ void smem_row_shuffle(int m, int n, long long* d, c2r::shuffle s);

template __global__ void smem_row_shuffle(int m, int n, float* d, r2c::shuffle s);
template __global__ void smem_row_shuffle(int m, int n, double* d, r2c::shuffle s);

template __global__ void smem_row_shuffle(int m, int n, int* d, r2c::shuffle s);
template __global__ void smem_row_shuffle(int m, int n, long long* d, r2c::shuffle s);

#else
namespace {

template<typename A, typename B>
void* magic() {
    return (void*)&smem_row_shuffle<A, B>;
}


template void* magic<float, c2r::shuffle>();
template void* magic<double, c2r::shuffle>();
template void* magic<int, c2r::shuffle>();
template void* magic<long long, c2r::shuffle>();

template void* magic<float, r2c::shuffle>();
template void* magic<double, r2c::shuffle>();
template void* magic<int, r2c::shuffle>();
template void* magic<long long, r2c::shuffle>();

}

#endif

}
}
