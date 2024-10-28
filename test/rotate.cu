#include <cuda/rotate.h>
#include <cuda/equations.cuh>
#include <common/gcd.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cassert>
#include <common/index.cuh>

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

template<typename F>
struct overall_rotate_gold {
    typedef int result_type;
    int m, n;
    F fn;
    __host__ __device__ overall_rotate_gold(int _m, int _n, F _fn) : m(_m), n(_n), fn(_fn) {}
    __host__ __device__ int operator()(int idx) {
        int row = idx / n;
        int col = idx % n;
        int rotate = fn(col);
        int src_row = row + rotate;
        if (src_row >= m) src_row -= m;
        return (src_row * n) + col;
    }
};

template<typename C, typename I>
void print_column(const C& c, const I& idx, int col) {
    for(int i = 0; i < idx.m; i++) {
        typename C::value_type x = c[idx(i, col)];
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

void test_rotate(int m, int n) {

    int c, k;
    inplace::extended_gcd(m, n, c, k);
    int b = n / c;

    
    typedef long long T;
    thrust::device_vector<T> x(m * n);
    thrust::copy(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(0) + m * n,
                 x.begin());
    // print_column(x, inplace::row_major_index(m, n), n-1);
    //print_array(x, inplace::row_major_index(m, n));
    std::cout << "m: " << m << " n: " << n << std::endl;
    
    
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    inplace::detail::r2c::postrotator fn(b, m);
    
    inplace::detail::rotate(fn, m, n, thrust::raw_pointer_cast(x.data()));
    
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(m * n * sizeof(T) * 2) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl;
    std::cout << std::endl;
    // thrust::device_vector<int> y(m*n);
    // thrust::counting_iterator<int> c(0);
    // thrust::transform(c, c+m*n, y.begin(), fine_rotate_gold(m, n));
    // print_column(x, inplace::row_major_index(m, n), n-1);    
    //print_array(x, inplace::row_major_index(m, n));

    
    assert(thrust::equal(x.begin(), x.end(), thrust::make_transform_iterator(
                             thrust::counting_iterator<int>(0),
                             overall_rotate_gold
                             <inplace::detail::r2c::postrotator>(m, n, fn))));
    std::cout << "-----------------PASSES----------------------" << std::endl;
}


int main() {
    // int m = 33;
    // int n = 33;
    // test_rotate(m, n);
    // int m = 32;
    // int n = 64;
    // int m = 33;
    // int n = 16;
    for(int m = 32; m < 100; m++) {
        for(int n = 32; n < 100; n++) {
            test_rotate(m, n);
        }
    }
    // test_rotate(604,372);
}
