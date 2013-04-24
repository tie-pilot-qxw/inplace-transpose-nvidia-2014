#include "rotate.h"

#include <thrust/device_vector.h>
#include <iostream>
#include <cassert>

template<typename T, typename Fn>
void print_array(const thrust::device_vector<T>& d, Fn index) {
    int m = index.m_m;
    int n = index.m_n;
    thrust::host_vector<T> h = d;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            T x = h[index(i, j)];
            std::cout.width(5); std::cout << std::right;
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
}


struct fine_rotate_gold {
    typedef int result_type;
    int m, n;
    __host__ __device__ fine_rotate_gold(int _m, int _n) : m(_m), n(_n) {}
    __host__ __device__ int operator()(int idx) {
        int row = idx / n;
        int col = idx % n;
        int group_col = col & (~0x1f);
        int coarse_rotate = group_col % m;
        int col_rotate = col % m;
        int fine_rotate = col_rotate - coarse_rotate;
        if (fine_rotate < 0) fine_rotate += m;
        int src_row = row + fine_rotate;
        if (src_row >= m) src_row -= m;
        return (src_row * n) + col;
    }
};

struct overall_rotate_gold {
    typedef int result_type;
    int m, n;
    __host__ __device__ overall_rotate_gold(int _m, int _n) : m(_m), n(_n) {}
    __host__ __device__ int operator()(int idx) {
        int row = idx / n;
        int col = idx % n;
        int rotate = col % m;
        int src_row = row + rotate;
        if (src_row >= m) src_row -= m;
        return (src_row * n) + col;
    }
};

int main() {
    //int m = 6;
    //int n = 23;
    int m = 511;
    int n = 64000;
    // int m = 32;
    // int n = 64;
    // int m = 33;
    // int n = 16;
    thrust::device_vector<int> x(m * n);
    thrust::copy(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(0) + m * n,
                 x.begin());

    // print_array(x, inplace::row_major_index(m, n));
    std::cout << std::endl;

 
    
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    
    inplace::post_rotate(m, n, thrust::raw_pointer_cast(x.data()));
   
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(m * n * sizeof(float) * 2) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl;

    // thrust::device_vector<int> y(m*n);
    // thrust::counting_iterator<int> c(0);
    // thrust::transform(c, c+m*n, y.begin(), fine_rotate_gold(m, n));
    
    // print_array(y, inplace::row_major_index(m, n));

    //print_array(x, inplace::row_major_index(m, n));
    
    assert(thrust::equal(x.begin(), x.end(), thrust::make_transform_iterator(
                             thrust::counting_iterator<int>(0),
                             overall_rotate_gold(m, n))));

    
}
