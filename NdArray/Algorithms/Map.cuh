//
// Created by bobini on 07.01.2022.
//

#ifndef BALG_MAP_CUH
#define BALG_MAP_CUH

#include "commonFunctionality.cuh"
#include <cuda/std/type_traits>

namespace BAlg::Algorithms::Implementations {

    template<typename T, typename F, typename R>
    __global__ void map(const T* in, R* out, std::size_t size, F fun)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            out[idx] = fun(in[idx]);
    }


    template<typename T, typename F>
    void mapDevice(const T *in, std::invoke_result_t<F,T>* out, std::size_t size, F fun)
    {
        constexpr int blockSize = 256;
        dim3 grid(ceil((double)size / (double)blockSize));
        dim3 block(blockSize);
        BAlg::Algorithms::Implementations::map<<<grid, block>>>(in, out, size, fun);
        auto zzz = cudaDeviceSynchronize();
        if (zzz != cudaSuccess)
            throw std::runtime_error("mapDevice: cudaDeviceSynchronize failed");
        checkErrors();
    }
}

#endif //BALG_MAP_CUH
