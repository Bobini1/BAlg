#pragma once

#include <ctgmath>
#include <cstdio>
#include <algorithm>
#include "Reduce.cuh"
#include "Map.cuh"
#include "Combine.cuh"

namespace BAlg::Algorithms {
    enum class Operation {
        ADD, MUL
    };

    template<typename T, typename F, typename R = T>
    R reduce(const T arr[], const size_t count, F fun, R identityElement = 0) {

        auto returnVal = Implementations::reduceDevice<T, F, R>(arr, count, fun, identityElement);

        return returnVal;
    }

    template<typename T, Operation op, typename R = T>
    R reduce(const T arr[], size_t count) {
        if constexpr (op == Operation::ADD) {
            auto addition = [=]__host__ __device__(R x, R y) -> R { return x + y; };
            return reduce<T, decltype(addition), R>(arr, count, addition, (R) 0);
        } else if constexpr (op == Operation::MUL) {
            auto multiplication = [=]__host__ __device__(R x, R y) -> R { return x * y; };
            return reduce<T, decltype(multiplication), R>(arr, count, multiplication, (R) 1);
        } else {
            return (R) 0;
        }
    }


    template<typename T, typename F>
    void map(const T *arr, cuda::std::invoke_result_t<F, T> *out, std::size_t count, F fun) {
        Implementations::mapDevice(arr, out, count, fun);
    }


    template<typename T, typename F>
    void combine(const T *arr1, const T *arr2, std::size_t count, std::size_t stride, F fun) {
        Implementations::combineDevice(arr1, arr2, count, stride, fun);
    }
}