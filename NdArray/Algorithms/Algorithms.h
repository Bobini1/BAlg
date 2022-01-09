#pragma once

#include <ctgmath>
#include <cstdio>
#include <algorithm>
#include "Reduce.cuh"
#include "Map.cuh"

namespace BAlg::Algorithms
	{
        enum class Operation
        {
            ADD, MUL
        };

		template <typename T, typename F, typename R = T>
		R reduce(const T arr[], const size_t count, F fun, R identityElement = 0)
		{
            T* input;
			cudaMalloc(&input, count * sizeof(T));

			cudaMemcpy(input, arr, count * sizeof(T), cudaMemcpyHostToDevice);

            auto returnVal = Implementations::reduceDevice<T, F, R>(input, count, fun, identityElement);

			cudaFree(input);

			return returnVal;
		}

		template <typename T, Operation op, typename R = T>
		R reduce(const T arr[], size_t count)
		{
            if constexpr (op == Operation::ADD)
            {
                auto addition = [=]__device__(R x, R y) { return x + y; };
                return reduce<T, decltype(addition), R>(arr, count, addition, (R)0);
            }
            else if constexpr (op == Operation::MUL)
            {
                auto multiplication = [=]__device__(R x, R y) { return x * y; };
                return reduce<T, decltype(multiplication), R>(arr, count, multiplication, (R)1);
            }
            else
            {
                return (R)0;
            }
		}



    template <typename T, typename F>
    void map(const T* arr, std::invoke_result_t<F,T>* out, std::size_t count, F fun)
    {
        Implementations::mapDevice(arr, out, count, fun);
    }

    }