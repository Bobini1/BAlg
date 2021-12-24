#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ctgmath>
#include <cstdio>
#include <algorithm>

namespace BAlg::Algorithms
	{
		enum class Operation
		{
			ADD, MUL
		};

		namespace {
			template <size_t blockSize, typename F, typename R>
			__device__ void warpReduce(volatile R* sdata, size_t tid, F fun) {
				if (blockSize >= 64) sdata[tid] = fun(sdata[tid], sdata[tid + 32]);
				if (blockSize >= 32) sdata[tid] = fun(sdata[tid], sdata[tid + 16]);
				if (blockSize >= 16) sdata[tid] = fun(sdata[tid], sdata[tid + 8]);
				if (blockSize >= 8) sdata[tid] = fun(sdata[tid], sdata[tid + 4]);
				if (blockSize >= 4) sdata[tid] = fun(sdata[tid], sdata[tid + 2]);
				if (blockSize >= 2) sdata[tid] = fun(sdata[tid], sdata[tid + 1]);
			}

			// source: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
			template <typename T, size_t blockSize, typename F, typename R>
			__global__ void reduce(T* g_idata, R* g_odata, size_t n, F fun) {
				extern __shared__ R sdata[];
				size_t tid = threadIdx.x;
				size_t i = blockIdx.x * (blockSize * 2) + tid;
				size_t gridSize = blockSize * 2 * gridDim.x;
				sdata[tid] = 0;
				while (i < n) { sdata[tid] = fun(sdata[tid], fun((R)g_idata[i], (R)g_idata[i + blockSize])); i += gridSize; }
				__syncthreads();
				if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fun(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
				if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fun(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
				if (blockSize >= 128) { if (tid < 64) { sdata[tid] = fun(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
				if (tid < 32) warpReduce<blockSize, F, R> (sdata, tid, fun);
				if (tid == 0)
                {
                    g_odata[blockIdx.x] = sdata[0];
                    //printf("%ld", fun(100, 200));
                }
			}

            template <typename T, typename F, typename R = T>
            R reduceDevice(T in[], size_t count, F fun)
            {
                int device;
                cudaGetDevice(&device);
                cudaDeviceProp props{};
                cudaGetDeviceProperties(&props, device);

                // explanation: http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/2-Findings/Findings.html
                // (bottom of the page)

                auto threadsTheory = (double)count / log2((double)count);

                auto threadsPerBlockTheory = sqrt(threadsTheory);

                size_t threadsPerBlock = std::max(1l, std::min(lround(pow(2, ceil(log2(threadsPerBlockTheory)))), 512l));

                auto blocksTheory = threadsPerBlockTheory;

                auto elemsPerThread = (size_t)ceil((double)count / ((double)threadsPerBlock * blocksTheory));

                size_t gridSize = count / (threadsPerBlock * elemsPerThread) + ((count % (threadsPerBlock * elemsPerThread) != 0) ? 1 : 0);

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
                        reduce<T, 512, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 256:
                        reduce<T, 256, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 128:
                        reduce<T, 128, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 64:
                        reduce<T, 64, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 32:
                        reduce<T, 32, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 16:
                        reduce<T, 16, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 8:
                        reduce<T, 8, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 4:
                        reduce<T, 4, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 2:
                        reduce<T, 2, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
                    case 1:
                        reduce<T, 1, F, R> <<< dimGrid, dimBlock, smemSize >>> (in, result, count, fun); break;
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
                    auto ret = reduceDevice(result, dimGrid.x, fun);
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

		template <typename T, typename F, typename R = T>
		R reduce(T arr[], size_t count, F fun)
		{
			R* result;
            T* input;
			cudaMalloc(&result, count * sizeof(R));
			cudaMalloc(&input, count * sizeof(T));

			cudaMemcpy(input, arr, count * sizeof(T), cudaMemcpyHostToDevice);

            auto returnVal = reduceDevice<T, F, R>(input, count, fun);

			cudaFree(result);
			cudaFree(input);

			return returnVal;
		}

		template <typename T, typename R = T>
		R reduce(T arr[], size_t count, Operation op)
		{
            auto addition = [=]__device__(R x, R y) { return x + y; };
            auto multiplication = [=]__device__(R x, R y) { return x * y; };

			switch (op)
			{
			case Operation::ADD:
                return reduce<T, decltype(addition), R>(arr, count, addition);
			case Operation::MUL:
                break;
			}
            return reduce<T, decltype(multiplication), R>(arr, count, multiplication);
		}
	}