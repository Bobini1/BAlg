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

    /**
     * @brief NdArrayCommonBase is the base class of NdArray and NdArrayRef.
     * @tparam T the underlying type of the elements of the array.
     * @tparam firstDim the size of the first dimension of the array.
     * @tparam dims the sizes of the remaining dimensions of the array.
     */
    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayCommonBase {
        friend class NdArrayRef<T, firstDim, dims...>;
        friend class NdArray<T, firstDim, dims...>;

    protected:
        static constexpr size_t elements = firstDim * (dims * ... * 1);

        virtual T* getMemoryStart() const = 0;

        static constexpr size_t numDims = 1 + sizeof...(dims);

    public:
        /**
         * @brief elementCount returns the total number of elements of type T in the array.
         * @return the total number of elements in the array.
         */
        [[nodiscard]] static constexpr std::size_t elementCount() {
            return elements;
        }

        /**
         * @brief dimensionCount returns the number of dimensions of the array.
         * @return the number of dimensions of the NdArray.
         */
        [[nodiscard]] static constexpr std::size_t dimensionCount() {
            return sizeof...(dims) + 1;
        }

        /**
         * @brief getDimension returns the size of the specified dimension.
         * @param dim the dimension to get the size of.
         * @return the size of the specified dimension.
         */
        [[nodiscard]] static constexpr std::size_t getDimension(std::size_t dim) {
            static_assert(dim < numDims, "Dimension out of bounds");
            return dim == 0 ? firstDim : NdArrayCommonBase<T, dims...>::getDimension(dim - 1);
        }

        /**
         * @brief getDimensions returns an <a href="https://en.cppreference.com/w/cpp/utility/integer_sequence">std::index_sequence</a>
         * representing the sizes of the dimensions of the array.
         * @return an index sequence representing the sizes of the dimensions of the NdArray.
         */
        [[nodiscard]] static constexpr std::index_sequence<firstDim, dims...> getDimensions() {
            return {firstDim, dims...};
        }


        /**
         * @brief size returns the size of the first dimension of the NdArray (the number of top-level sub-arrays).
         * @return the size of the first dimension of the NdArray.
         */
        [[nodiscard]] static constexpr std::size_t size() {
            return firstDim;
        }

        /**
         * @brief sum calculates the sum of all underlying elements of the array.
         * @tparam R The return type of the function. This can be used for optimization purposes to create the array of
         * the target type directly on the device. Example: summing the array of 1_000_000 chars to a long long. If we
         * convert the chars to long longs on the host, we need to copy much more data.
         * T must be convertible to R.
         * @return the sum of the elements of the entire NdArray.
         */
        template<typename R = T>
        R sum() const {
//            if constexpr(!std::is_same_v<R, T>)
//            {
//                static_assert(std::is_convertible_v<T, R>, "T must be convertible to R");
//            }
            return Algorithms::reduce<T, Algorithms::Operation::ADD, R>(getMemoryStart(), elements);
        }


        /**
         * @brief product calculates the product of all underlying elements of the array.
         * @tparam R The return type of the function. This can be used for optimization purposes to create the array of
         * the target type directly on the device. Example: calculating the product the array of 1_000_000 chars to a
         * long long. If we convert the chars to long longs on the host, we need to copy much more data.
         * T must be convertible to R.
         * @return the sum of the elements of the entire NdArray.
         */
        template<typename R = T>
        R product() const {
//            if constexpr(!std::is_same_v<R, T>)
//            {
//                static_assert(std::is_convertible_v<T, R>, "T must be convertible to R");
//            }
            return Algorithms::reduce<T, Algorithms::Operation::MUL, R>(getMemoryStart(), elements);
        }

        /**
         * @brief map applies a function to each underlying element of the array.
         * @tparam F the type of the function to apply.
         * @param fun the function to apply.
         * @return a new NdArray with the results of the function applied to the elements of the source NdArray.
         */
        template<typename F>
        NdArray<cuda::std::invoke_result_t<F, T>, firstDim, dims...> map(F fun)
        {
            NdArray<cuda::std::invoke_result_t<F, T>, firstDim, dims...> result;
            Algorithms::map(getMemoryStart(), result.getMemoryStart(), elements, fun);
            return result;
        }

        /**
         * @brief operator[] can be used to access the elements of the NdArray.
         * @param index the index of the element to return.
         * @return either a reference to the element at the specified index (in the case of a 1D
         * NdArray) or an NdArrayRef representing the sub-array at the specified index.
         */
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

    /**
     * @brief NdArray is a multidimensional array.
     * @tparam T the underlying type of the elements of the array.
     * @tparam firstDim the size of the first dimension of the array.
     * @tparam dims the sizes of the remaining dimensions of the array.
     */
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
            if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
            for (size_t i = 0; i < elements; ++i)
                data[i] = memoryStart[i];
        }
    protected:
        T* getMemoryStart() const override {
            return data;
        }

    public:

        /**
         * @brief NdArray is a multidimensional array.
         * @tparam T the underlying type of the elements of the array.
         * @tparam firstDim the size of the first dimension of the array.
         * @tparam dims the sizes of the remaining dimensions of the array.
         */
        explicit NdArray(bool initializeElements = true)
        {
            static constexpr auto size = sizeof(T) * elements;
            auto error = cudaMallocManaged(&data, size);
            if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
            if (initializeElements) data = new (data) T[elements]();
        }

        ~NdArray()
        {
            for (size_t i = 0; i < elements; ++i)
                data[i].~T();
            cudaFree(data);
        }

        NdArray(const NdArrayCommonBase<T, firstDim, dims...>& array)
        {
            cudaMallocManaged(&data, sizeof(T) * elements);
            for (size_t i = 0; i < elements; ++i)
                data[i] = array.data[i];
        }

        /**
         * Constructs the NdArray from an rvalue reference to another NdArray.
         * @param array
         */
        NdArray(NdArray&& array) noexcept
        {
            data = array.data;
            array.data = nullptr;
        }

        /**
         * @brief operator= assigns the contents of the source NdArray to the target NdArray.
         * @param array the source NdArray.
         * @return a reference to the target NdArray.
         */
        NdArray& operator=(const NdArrayCommonBase<T, firstDim, dims...>& array) {
            if (this->getMemoryStart() != array.getMemoryStart()) {
                auto memory = array.getMemoryStart();
                for (size_t i = 0; i < elements; ++i)
                {
                    data[i] = memory[i];
                }
            }
            return *this;
        }
    };

    /**
     * @brief NdArrayRef is a reference to a multidimensional array or any of its sub-arrays.
     * @tparam T the underlying type of the elements of the array.
     * @tparam firstDim the size of the first dimension of the array.
     * @tparam dims the sizes of the remaining dimensions of the array.
     */
    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayRef : public NdArrayCommonBase<T, firstDim, dims...> {
        using Base = NdArrayCommonBase<T, firstDim, dims...>;
        using Base::elements;
        using Base::dimensionCount;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArrayCommonBase;

        T* memoryStart;

        /**
         * Constructs an NdArrayRef from a pointer to the first element of the array. The ref does not own the memory.
         * @param memoryStart the pointer to the first element of the array.
         */
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

        /**
         * @brief Constructs the NdArrayRef from another NdArrayRef. Does not copy the memory, but instead references
         * the same memory.
         * @param array the source NdArrayRef.
         */
        NdArrayRef(const NdArrayRef& array)
        {
            this->memoryStart = array.memoryStart;
        }

        /**
         * @brief assigns the data of the source NdArray to the target NdArrayRef. A copy of the elements is performed.
         * The changes will be visible in the the original NdArray of the ref.
         * @param array
         */
        NdArrayRef& operator=(const NdArrayCommonBase<T, firstDim, dims...>& array) {
            if (this->getMemoryStart() != array.getMemoryStart()) {
                auto memory = array.getMemoryStart();
                for (size_t i = 0; i < elements; ++i)
                {
                    memoryStart[i] = memory[i];
                }
            }
            return *this;
        }
    };

}
#endif //BALG_NDARRAYVARIADIC_H
