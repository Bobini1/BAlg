#pragma once

#include <vector>
#include <array>
#include <memory>
#include <type_traits>
#include "Shape.h"

namespace BAlg::DataStructures {
    namespace {
        template<typename T, std::size_t dims>
        class NdArrayBase
        {
            static_assert(dims >= 1);
            friend class NdArrayBase<T, dims+1>;
            friend class NdArrayBase<T, dims-1>;
        protected:
            Shape<dims> shape;
            std::shared_ptr<T[]> data;
            std::size_t offset;

            NdArrayBase(std::initializer_list<std::size_t> shape)
            {
                this->shape = Shape<dims>(shape);
                data = std::shared_ptr<T[]>(new T[this->shape.elementCount()]);
                offset = 0;
            }

            NdArrayBase(const NdArrayBase& otherArray)
            {
                auto newData = std::make_shared<T[]>(new T[otherArray.shape.elementCount()]);
                std::copy(otherArray.data + otherArray.offset,
                          otherArray.data + otherArray.offset + otherArray.shape.elementCount(),
                          newData);
                data = newData;
                shape = otherArray.shape;
                offset = 0;
            }

            ~NdArrayBase() = default;
            NdArrayBase(NdArrayBase&& otherArray) noexcept
            :shape(otherArray.shape)
            {
                auto newData = std::make_shared<T[]>(new T[otherArray.shape.elementCount()]);
                std::copy(otherArray.data + otherArray.offset,
                          otherArray.data + otherArray.offset + otherArray.shape.elementCount(),
                          newData);
                data = newData;
                offset = otherArray.offset;
            }

            explicit NdArrayBase(const NdArrayBase<T, dims+1>& otherArray, std::size_t index)
            :shape(otherArray.shape.subShape())
            {
                data = otherArray.data;
                offset = otherArray.offset + shape.strideAt(0) * index;
            }
        public:
            NdArrayBase& operator=(const NdArrayBase& otherArray)
            {
                auto newData = std::make_shared<T[]>(new T[otherArray.shape.elementCount()]);
                std::copy(otherArray.data + otherArray.offset,
                          otherArray.data + otherArray.offset + otherArray.shape.elementCount(),
                          newData);
                data = newData;
                shape = otherArray.shape;
                offset = 0;
            }

            NdArrayBase& operator=(NdArrayBase&& otherArray) noexcept
            {
                shape = otherArray.shape;
                auto newData = std::make_shared<T[]>(new T[otherArray.shape.elementCount()]);
                std::copy(otherArray.data + otherArray.offset,
                          otherArray.data + otherArray.offset + otherArray.shape.elementCount(),
                          newData);
                data = newData;
                offset = 0;
            }
        };
    }

    template<typename T, std::size_t dims>
    class NdArray : public NdArrayBase<T, dims>
    {
        using base = NdArrayBase<T, dims>;
        friend class NdArray<T, dims+1>;
        friend class NdArray<T, dims-1>;
        explicit NdArray(const NdArray<T, dims+1>& otherArray, std::size_t index) : base(otherArray, index) {}
    public:
        NdArray<T, dims-1> operator[](std::size_t index)
        {
            return NdArray<T, dims-1>(*this, index);
        }
        NdArray(std::initializer_list<std::size_t> shape) : base(shape) {}
        NdArray(const NdArray& otherArray) : base(otherArray) {}
        ~NdArray() = default;
        NdArray(NdArray&& otherArray) noexcept : base(otherArray) {}
    };


    template<typename T>
    class NdArray<T, 1> : public NdArrayBase<T, 1>
    {
        using base = NdArrayBase<T, 1>;
        friend class NdArray<T, 2>;
    public:
        explicit NdArray(const NdArray<T, 2>& otherArray, std::size_t index) : base(otherArray, index) {}
        NdArray(std::initializer_list<std::size_t> shape) : base(shape) {}
        NdArray(const NdArray& otherArray) : base(otherArray) {};
        ~NdArray() = default;
        NdArray(NdArray&& otherArray) noexcept : base(otherArray) {}
        T& operator[](std::size_t index)
        {
            return base::data[index];
        }
    };
}

