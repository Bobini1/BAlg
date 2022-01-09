
#include "Algorithms/Algorithms.h"
#include <iostream>
#include "NdArrayVariadic.h"

using namespace BAlg::DataStructures;

template<typename T, typename F, typename R>
__global__ void map(const T* in, R* out, std::size_t size, F fun)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = fun(in[idx]);
}

int main()
{
    NdArray<int, 5, 5> b;
    for (int i = 0; i < b.elementCount(); ++i)
    {
        for(int j = 0; j < 5; ++j)
            b[i][j] = i + j;
    }

    NdArray<int, 5, 5> c = b.map([]__device__(int x) { return x * 2; });

    for (int i = 0; i < 5; ++i)
    {
        for(int j = 0; j < 5; ++j)
            std::cout << c[i][j] << " ";
    }

}