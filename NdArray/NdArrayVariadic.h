//
// Created by bobini on 29.12.2021.
//

#ifndef BALG_NDARRAYVARIADIC_H
#define BALG_NDARRAYVARIADIC_H

#include <cstddef>
#include <array>
#include <memory>

namespace BAlg::DataStructures {
    template<typename T, std::size_t firstDim, std::size_t... dims>
    struct NdArrayCommonBase {
        static constexpr size_t elements = firstDim * (dims * ... * 1);

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
    };

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArray;

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayRef;

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayRefBase : public NdArrayCommonBase<T, firstDim, dims...>
    {
        using Base = NdArrayCommonBase<T, firstDim, dims...>;
    protected:
        T* memoryStart;
        using Base::elements;
        using Base::dimensionCount;
        using Base::size;
        NdArrayRefBase(const NdArrayRefBase & array) = default;
        NdArrayRefBase(NdArrayRefBase && array) noexcept = default;

        explicit NdArrayRefBase(T* memoryStart)
        {
            this->memoryStart = memoryStart;
        }

    public:
        NdArrayRefBase& operator=(const NdArrayRefBase & array) {
            if (this != &array) {
                memoryStart = array.memoryStart;
            }
            return *this;
        }

        virtual NdArrayRefBase &operator=(const NdArray<T, firstDim, dims...>& array)
        {
            if (array.data.get() != memoryStart) {
                for (size_t i = 0; i < elements; ++i) {
                    memoryStart[i] = array.data[i];
                }
            }
            return *this;
        }
    };

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayBase : public NdArrayCommonBase<T, firstDim, dims...>
    {
        using Base = NdArrayCommonBase<T, firstDim, dims...>;
    protected:
        std::shared_ptr<T[]> data;
        explicit NdArrayBase(T* memoryStart) : data(std::make_unique<T[]>(elements))
        {
            for (size_t i = 0; i < elements; ++i)
                data[i] = memoryStart[i];
        }
        using Base::elements;
        using Base::dimensionCount;
        using Base::size;
        NdArrayBase()
        {
            data = std::make_unique<T[]>(elements);
        }
        NdArrayBase(const NdArrayBase & array)
        {
            data = std::make_unique<T[]>(elements);
            for (size_t i = 0; i < elements; ++i)
                data[i] = array.data[i];
        }
        NdArrayBase(NdArrayBase && array) noexcept
        {
            data = std::move(array.data);
        }

    public:
        NdArrayBase& operator=(const NdArrayBase & array) {
            if (this != &array) {
                data = std::make_unique<T[]>(elements);
                for (size_t i = 0; i < elements; ++i)
                    data[i] = array.data[i];
            }
            return *this;
        }

        virtual NdArrayBase& operator=(const NdArrayRef<T, firstDim, dims...>& array)
        {
            if (data.get() != array.memoryStart) {
                for (size_t i = 0; i < elements; ++i) {
                    data[i] = array.memoryStart[i];
                }
            }
            return *this;
        }
    };



    template<typename, std::size_t, std::size_t...>
    class NdArrayRef;

    template<typename T, std::size_t firstDim, std::size_t... dims>
    struct NdArray : public NdArrayBase<T, firstDim, dims...> {
        using Base = NdArrayBase<T, firstDim, dims...>;
        using Base::elements;
        using Base::dimensionCount;
        using Base::size;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArrayRefBase;

    public:
        NdArray() : Base() {}

        ~NdArray() = default;

        NdArray(const NdArray& array) : Base(array) {}

        NdArray(NdArray&& array) noexcept : Base(array) {}

        NdArrayRef<T, dims...> operator[](std::size_t index) {
            return NdArrayRef<T, dims...>(this->data.get() + index * (dims * ... * 1));
        }

        NdArray& operator=(const NdArrayRef<T, firstDim, dims...>& array)
        {
            Base::operator=(array);
            return *this;
        }
    };

    template<typename T, std::size_t firstDim, std::size_t... dims>
    class NdArrayRef : public NdArrayRefBase<T, firstDim, dims...> {
        using Base = NdArrayRefBase<T, firstDim, dims...>;
        using Base::elements;
        using Base::dimensionCount;
        using Base::size;
        using Base::memoryStart;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArrayBase;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArray;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArrayRef;

        explicit NdArrayRef(T* memoryStart) : Base(memoryStart) {}
    public:
        ~NdArrayRef() = default;

        NdArrayRef(const NdArrayRef& array) : Base(array) {}

        NdArrayRef(NdArrayRef&& array) noexcept : Base(std::move(array)) {}

        NdArrayRef<T, dims...>& operator[](std::size_t index) {
            if (index >= firstDim) {
                throw std::out_of_range("Index out of range");
            }
            return NdArrayRef<T, dims...>(memoryStart + index * (dims * ... * 1));
        }


        NdArrayRef& operator=(const NdArray<T, firstDim, dims...>& array)
        {
            Base::operator=(array);
            return *this;
        }
    };

    template<typename T, std::size_t firstDim>
    class NdArray<T, firstDim> : public NdArrayBase<T, firstDim> {
        using Base = NdArrayBase<T, firstDim>;
        using Base::elements;
        using Base::dimensionCount;
        using Base::size;
        using Base::data;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArrayRefBase;


        explicit NdArray(T* memoryStart) : Base(memoryStart) {}
    public:
        NdArray() : Base() {}

        ~NdArray() = default;

        NdArray(const NdArray& array) : Base(array) {}

        NdArray(NdArray&& array) noexcept : Base(array) {}

        T& operator[](std::size_t index) {
            if (index >= firstDim) {
                throw std::out_of_range("Index out of range");
            }
            return data[index];
        }

        NdArray& operator=(const NdArrayRef<T, firstDim>& array)
        {
            Base::operator=(array);
            return *this;
        }
    };

    template<typename T, std::size_t firstDim>
    class NdArrayRef<T, firstDim> : public NdArrayRefBase<T, firstDim> {
        using Base = NdArrayRefBase<T, firstDim>;
        using Base::elements;
        using Base::dimensionCount;
        using Base::size;
        using Base::memoryStart;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArray;

        template<typename, std::size_t, std::size_t...>
        friend
        class NdArrayBase;

        explicit NdArrayRef(T *memoryStart) : Base(memoryStart) {}
    public:
        NdArrayRef(const NdArrayRef & array) : Base(array) {}

        NdArrayRef(NdArrayRef && array) noexcept : Base(std::move(array)) {}

        T& operator[](std::size_t index) const {
            if (index >= firstDim) {
                throw std::out_of_range("Index out of range");
            }
            return *(memoryStart + index);
        }

        NdArrayRef& operator=(const NdArray<T, firstDim>& array)
        {
            Base::operator=(array);
            return *this;
        }
    };


}
#endif //BALG_NDARRAYVARIADIC_H
