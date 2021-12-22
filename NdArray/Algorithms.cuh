#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <ctgmath>
#include <cstdio>
#include <algorithm>

template<typename T>
using func_t = T(*) (T, T);

namespace BAlg
{
	namespace Algorithms
	{
		enum class Operation
		{
			ADD, MUL
		};

		template <typename T>
		__device__ T add(T x, T y)
		{
			return x + y;
		}

		template <typename T>
		__device__ T mul(T x, T y)
		{
			return x * y;
		}


		// explanation: https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/
		template <typename T>
		__device__ func_t<T> p_add = add<T>;
		template <typename T>
		__device__ func_t<T> p_mul = mul<T>;

		namespace {
			template <typename T, size_t blockSize>
			__device__ void warpReduce(volatile T* sdata, size_t tid, func_t<T> fun) {
				if (blockSize >= 64) sdata[tid] = (*fun)(sdata[tid], sdata[tid + 32]);
				if (blockSize >= 32) sdata[tid] = (*fun)(sdata[tid], sdata[tid + 16]);
				if (blockSize >= 16) sdata[tid] = (*fun)(sdata[tid], sdata[tid + 8]);
				if (blockSize >= 8) sdata[tid] = (*fun)(sdata[tid], sdata[tid + 4]);
				if (blockSize >= 4) sdata[tid] = (*fun)(sdata[tid], sdata[tid + 2]);
				if (blockSize >= 2) sdata[tid] = (*fun)(sdata[tid], sdata[tid + 1]);
			}

			// source: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
			template <typename T, size_t blockSize>
			__global__ void reduce(T* g_idata, T* g_odata, size_t n, func_t<T> fun) {
				extern __shared__ T sdata[];
				size_t tid = threadIdx.x;
				size_t i = blockIdx.x * (blockSize * 2) + tid;
				size_t gridSize = blockSize * 2 * gridDim.x;
				sdata[tid] = 0;
				while (i < n) { sdata[tid] = (*fun)(sdata[tid], g_idata[i] + g_idata[i + blockSize]); i += gridSize; }
				__syncthreads();
				if (blockSize >= 512) { if (tid < 256) { sdata[tid] = (*fun)(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
				if (blockSize >= 256) { if (tid < 128) { sdata[tid] = (*fun)(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
				if (blockSize >= 128) { if (tid < 64) { sdata[tid] = (*fun)(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
				if (tid < 32) warpReduce<T, blockSize> (sdata, tid, fun);
				if (tid == 0) g_odata[blockIdx.x] = sdata[0];
			}
		}

		template <typename T>
		T reduce(T arr[], size_t count, func_t<T> fun)
		{
			

			int device;
			cudaGetDevice(&device);
			cudaDeviceProp props;
			cudaGetDeviceProperties(&props, device);


			size_t warpSize = props.warpSize;

			// explanation: http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/2-Findings/Findings.html
			// (bottom of the page)
			
			auto threadsTheory = log2(count);

			auto blocksTheory = sqrt(threadsTheory);

			auto threadsPerBlockTheory = blocksTheory;

			size_t threadsPerBlock = std::max(1l, std::min((long)(pow(2, (size_t)ceil(log2(threadsPerBlockTheory))) + 0.5), 512l));

			size_t elemsPerThread = (size_t)ceil(count / (threadsPerBlock * blocksTheory));

			size_t gridSize = count / (threadsPerBlock * elemsPerThread) +((count % (threadsPerBlock * elemsPerThread) != 0) ? 1 : 0);

			if (gridSize > (size_t)props.maxGridSize[0]) throw std::runtime_error("Grid size too large");

			dim3 dimGrid(gridSize);
			dim3 dimBlock(threadsPerBlock);
			size_t smemSize = sizeof(T) * (count / gridSize + ((count % gridSize != 0) ? 1 : 0));

			if (smemSize > (size_t)props.sharedMemPerBlock) throw std::runtime_error("Shared memory too large");

			if (threadsPerBlock > (size_t)props.maxThreadsPerBlock) throw std::runtime_error("Too many threads per block");

			T* result, * input;
			cudaMalloc(&result, count * sizeof(T));
			cudaMalloc(&input, count * sizeof(T));

			cudaMemcpy(input, arr, count * sizeof(T), cudaMemcpyHostToDevice);

			switch (threadsPerBlock)
			{
			case 512:
				reduce<T, 512> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 256:
				reduce<T, 256> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 128:
				reduce<T, 128> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 64:
				reduce<T, 64> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 32:
				reduce<T, 32> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 16:
				reduce<T, 16> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 8:
				reduce<T, 8> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 4:
				reduce<T, 4> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 2:
				reduce<T, 2> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
			case 1:
				reduce<T, 1> <<< dimGrid, dimBlock, smemSize >>> (input, result, count, fun); break;
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

			T returnVal;

			cudaMemcpy(&returnVal, result, sizeof(T), cudaMemcpyDeviceToHost);

			cudaFree(result);
			cudaFree(input);

			return returnVal;
		}

		template <typename T>
		T reduce(T arr[], size_t count, Operation op)
		{
			func_t<T> fun;
			switch (op)
			{
			case Operation::ADD:
				cudaMemcpyFromSymbol(&fun, p_add<T>, sizeof(func_t<T>));
			case Operation::MUL:
				cudaMemcpyFromSymbol(&fun, p_mul<T>, sizeof(func_t<T>));
			}

			return reduce(arr, count, fun);
		}
	}
}