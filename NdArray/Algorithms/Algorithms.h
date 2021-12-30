#pragma once

#include <ctgmath>
#include <cstdio>
#include <algorithm>
#include "Algorithms.cuh"

namespace BAlg::Algorithms
	{
		enum class Operation
		{
			ADD, MUL
		};

		template <typename T, typename F, typename R = T>
		R reduce(const T arr[], const size_t count, F fun, R identityElement = 0)
		{
			R* result;
            T* input;
			cudaMalloc(&result, count * sizeof(R));
			cudaMalloc(&input, count * sizeof(T));

			cudaMemcpy(input, arr, count * sizeof(T), cudaMemcpyHostToDevice);

            auto returnVal = reduceDevice<T, F, R>(input, count, fun, identityElement);

			cudaFree(result);
			cudaFree(input);

			return returnVal;
		}

		template <typename T, typename R = T>
		R reduce(const T arr[], const size_t count, Operation op)
		{
            auto addition = [=]__device__(R x, R y) { return x + y; };
            auto multiplication = [=]__device__(R x, R y) { return x * y; };

			switch (op)
			{
			case Operation::ADD:
                return reduce<T, decltype(addition), R>(arr, count, addition, 0);
			case Operation::MUL:
                break;
			}
            return reduce<T, decltype(multiplication), R>(arr, count, multiplication, 1);
		}
	}