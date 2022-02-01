//
// Created by bobini on 30.01.2022.
//

#ifndef BALG_COMBINE_CUH
#define BALG_COMBINE_CUH

namespace BAlg::Algorithms::Implementations
{
    template<typename T, typename F>
    __global__ void combine(T *dst, const T *src, size_t count, size_t stride, const F *fun)
    {
        alignas(T) extern __shared__ unsigned char sdata_u[];
        T* sdata = reinterpret_cast<T*>(sdata_u);

        size_t tid = threadIdx.x;
        size_t bid = blockIdx.x;
        size_t bsize = blockDim.x;
        size_t gsize = gridDim.x;

        size_t i = bid * bsize + tid;

        size_t elemsPerBlock = 256;


        size_t elemsPerThread = ((count % gsize) ? ((count / gsize) + 1) : (count / gsize)) / bsize;

        if (tid < elemsPerBlock)
        {
            for (size_t j = 0; j < count; j++)
            {
                sdata[tid + j] = src[i * stride + j];
            }
        }

        __syncthreads();

        for (size_t it = i; i < count; i += stride)
        {

        }
    }

    template<typename T, typename F>
    void combineDevice(T *dst, const T *src, size_t count, size_t stride, F fun)
    {
        dim3 gridSize;
        dim3 blockSize;
        cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, combine<T, F>, 0, 0);

        combine<<<gridSize, blockSize>>>(dst, src, count, stride, fun);
    }
}

#endif //BALG_COMBINE_CUH
