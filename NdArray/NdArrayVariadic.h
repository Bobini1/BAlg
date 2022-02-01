//
// Created by bobini on 29.12.2021.
//

#ifndef BALG_NDARRAYVARIADIC_H
#define BALG_NDARRAYVARIADIC_H

#include <cstddef>
#include <array>
#include <memory>
#include <cuda/std/type_traits>
#include "Algorithms/Algorithms.h"

namespace BAlg::DataStructures {
    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArray;

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayRef;

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayCommonBase {
        friend class NdArrayRef<T, firstDim, dims...>;
        friend class NdArray<T, firstDim, dims...>;

    protected:
        static constexpr size_t elements = firstDim * (dims * ... * 1);

        virtual T* getMemoryStart() const = 0;

        static constexpr size_t numDims = sizeof...(dims);

    public:
        [[nodiscard]] constexpr std::size_t elementCount() const {
            return elements;
        }

        [[nodiscard]] constexpr std::size_t dimensionCount() const {
            return sizeof...(dims) + 1;
        }

        [[nodiscard]] constexpr std::size_t size() const {
            return firstDim;
        }

        template<typename R = T>
        R sum() const {
            return Algorithms::reduce<T, Algorithms::Operation::ADD, R>(getMemoryStart(), elements);
        }

        template<typename R = T>
        R product() const {
            return Algorithms::reduce<T, Algorithms::Operation::MUL, R>(getMemoryStart(), elements);
        }

        template<typename F>
        NdArray<cuda::std::invoke_result_t<F, T>, firstDim, dims...> map(F fun)
        {
            NdArray<cuda::std::invoke_result_t<F, T>, firstDim, dims...> result;
            Algorithms::map(getMemoryStart(), result.getMemoryStart(), elements, fun);
            return result;
        }

        decltype(auto) operator[](size_t index)
        {
            if constexpr(sizeof...(dims) == 0) {
                return getMemoryStart()[index];
            }
            else {
                return NdArrayRef<T, dims...>(getMemoryStart() + index * firstDim);
            }
        }
    };

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArray : public NdArrayCommonBase<T, firstDim, dims...> {
        using Base = NdArrayCommonBase<T, firstDim, dims...>;
        using Base::elements;
        using Base::dimensionCount;

        T* data;

        template<typename, std::size_t, std::size_t...>
        friend class NdArrayCommonBase;

        explicit NdArray(T* memoryStart)
        {
            auto error = cudaMallocManaged(&data, sizeof(T) * elements);
            if (error != cudaSuccess) {
                if (error == cudaErrorMemoryAllocation) {
                    throw std::bad_alloc();
                }
                else if (error == cudaErrorInvalidValue) {
                    throw std::invalid_argument("Invalid value");
                }

                else std::cout << (int)error << std::endl;
            }
            for (size_t i = 0; i < elements; ++i)
                data[i] = memoryStart[i];
        }
    protected:
        T* getMemoryStart() const override {
            return data;
        }

    public:

        NdArray()
        {
            auto size = sizeof(T) * elements;
            auto error = cudaMallocManaged(&data, size);
            if (error != cudaSuccess) {
                if (error == cudaErrorMemoryAllocation) {
                    throw std::bad_alloc();
                }
                else if (error == cudaErrorInvalidValue) {
                    throw std::invalid_argument("Invalid value");
                }

                else throw std::runtime_error("Unknown allocation error");
            }
        }

        ~NdArray()
        {
            cudaFree(data);
        }

        NdArray(const NdArrayCommonBase<T, firstDim, dims...>& array)
        {
            cudaMallocManaged(&data, sizeof(T) * elements);
            for (size_t i = 0; i < elements; ++i)
                data[i] = array.data[i];
        }

        NdArray(NdArray&& array) noexcept
        {
            data = array.data;
            array.data = nullptr;
        }

        NdArray& operator=(const NdArrayCommonBase<T, firstDim, dims...>& array) {
            if (this->getMemoryStart() != array.getMemoryStart()) {
                for (size_t i = 0; i < elements; ++i)
                {
                    auto memory = array.getMemoryStart();
                    data[i] = memory[i];
                }
            }
            return *this;
        }
    };

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayRef : public NdArrayCommonBase<T, firstDim, dims...> {
        using Base = NdArrayCommonBase<T, firstDim, dims...>;
        using Base::elements;
        using Base::dimensionCount;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArrayCommonBase;

        T* memoryStart;

        explicit NdArrayRef(T* memoryStart)
        {
            this->memoryStart = memoryStart;
        }

    protected:
        T* getMemoryStart() const override {
            return memoryStart;
        }
    public:
        ~NdArrayRef() = default;

        NdArrayRef(const NdArrayRef& array)
        {
            this->memoryStart = array.memoryStart;
        }

        NdArrayRef& operator=(const NdArrayRef& array) {
            if (this->getMemoryStart() != array.getMemoryStart()) {
                memoryStart = array.memoryStart;
            }
            return *this;
        }

        NdArrayRef &operator=(const NdArray<T, firstDim, dims...>& array)
        {
            if (array.data != memoryStart) {
                for (size_t i = 0; i < elements; ++i) {
                    memoryStart[i] = array.data[i];
                }
            }
            return *this;
        }
    };

}
#endif //BALG_NDARRAYVARIADIC_H
