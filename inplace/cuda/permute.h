#pragma once

namespace inplace {
namespace detail {

template<typename T, typename F>
void scatter_permute(F f, int m, int n, T* data, int* tmp);

}
}
