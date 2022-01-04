//
// Created by bobini on 30.12.2021.
//

#ifndef BALG_ALGORITHMS_CUH
#define BALG_ALGORITHMS_CUH

#include <ctgmath>
#include <cstdio>
#include <algorithm>

namespace BAlg::Algorithms
{
    template <size_t blockSize, typename F, typename R>
    __device__ void warpReduce(volatile R* sdata, const size_t tid, F fun) {
        if (blockSize >= 64) sdata[tid] = fun(sdata[tid], sdata[tid + 32]);
        if (blockSize >= 32) sdata[tid] = fun(sdata[tid], sdata[tid + 16]);
        if (blockSize >= 16) sdata[tid] = fun(sdata[tid], sdata[tid + 8]);
        if (blockSize >= 8) sdata[tid] = fun(sdata[tid], sdata[tid + 4]);
        if (blockSize >= 4) sdata[tid] = fun(sdata[tid], sdata[tid + 2]);
        if (blockSize >= 2) sdata[tid] = fun(sdata[tid], sdata[tid + 1]);
    }

    // source: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    template <typename T, size_t blockSize, typename F, typename R>
    __global__ void reduce(const T* g_idata, R* g_odata, const size_t n, F fun, R identityElement) {
        alignas(sizeof(R)) extern __shared__ unsigned char sdata_u[];
        R* sdata = reinterpret_cast<R*>(sdata_u);

        size_t tid = threadIdx.x;
        size_t i = blockIdx.x * (blockSize * 2) + tid;
        size_t gridSize = blockSize * 2 * gridDim.x;
        sdata[tid] = identityElement;
        while (i < n) { sdata[tid] = fun(sdata[tid], fun((R)g_idata[i], (R)g_idata[i + blockSize])); i += gridSize; }
        __syncthreads();
        if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fun(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
        if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fun(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
        if (blockSize >= 128) { if (tid < 64) { sdata[tid] = fun(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
        if (tid < 32) warpReduce<blockSize, F, R> (sdata, tid, fun);
        if (tid == 0)
        {
            g_odata[blockIdx.x] = sdata[0];
        }
    }

    template <typename T, typename F, typename R>
    R reduceDevice(const T in[], const size_t count, F fun, R identityElement)
    {
        if (count == 0) return identityElement;

        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props{};
        cudaGetDeviceProperties(&props, device);

        // explanation: http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/2-Findings/Findings.html
        // (bottom of the page)
        static constexpr double multiplier = 10.0;

        auto threadsTheory = (double)count / std::max(1.0, log2((double)count));

        auto threadsPerBlockTheory = std::min((double)props.sharedMemPerBlock / sizeof(R), sqrt(threadsTheory));

        size_t threadsPerBlock = std::max(1l, std::min(lround(pow(2, ceil(log2(threadsPerBlockTheory)))), 512l));

        auto gridSize = count / threadsPerBlock + ((count % threadsPerBlock != 0) ? 1 : 0);

        if (gridSize > (size_t)props.maxGridSize[0]) throw std::runtime_error("Grid size too large");

        dim3 dimGrid(gridSize);
        dim3 dimBlock(threadsPerBlock);
        size_t smemSize = sizeof(R) * (count / gridSize + ((count % gridSize != 0) ? 1 : 0));

        if (smemSize > (size_t)props.sharedMemPerBlock) throw std::runtime_error("Shared memory too large");

        if (threadsPerBlock > (size_t)props.maxThreadsPerBlock) throw std::runtime_error("Too many threads per block");



        R* result;
        cudaMalloc(&result, dimGrid.x * sizeof(R));

        switch (threadsPerBlock)
        {
            case 512:
                reduce<T, 512, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 256:
                reduce<T, 256, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 128:
                reduce<T, 128, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 64:
                reduce<T, 64, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 32:
                reduce<T, 32, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 16:
                reduce<T, 16, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 8:
                reduce<T, 8, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 4:
                reduce<T, 4, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 2:
                reduce<T, 2, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
            case 1:
                reduce<T, 1, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun, identityElement); break;
        }

        cudaError_t cudaStatus;
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        }

        if (dimGrid.x != 1)
        {
            auto ret = reduceDevice<R, F, R>(result, dimGrid.x, fun, identityElement);
            cudaFree(result);
            return ret;
        }
        else
        {
            R returnVal;

            cudaMemcpy(&returnVal, result, sizeof(R), cudaMemcpyDeviceToHost);
            cudaFree(result);

            return returnVal;
        }
    }
}

#endif //BALG_ALGORITHMS_CUH
