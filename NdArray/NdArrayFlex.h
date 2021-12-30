#pragma once

/**
 Failed concept, left only for reference
**/

#include <vector>
#include <array>
#include <memory>
#include <type_traits>
#include <ostream>
#include "Shape.h"

namespace BAlg::DataStructures::Deprecated {
    template<typename T, std::size_t dims>
    class NdArrayFlex;

    namespace {
        template<typename T, std::size_t dims>
        class NdArrayFlexBase
        {
            static_assert(dims >= 1);
            friend class NdArrayFlexBase<T, dims+1>;
            friend class NdArrayFlexBase<T, dims-1>;
        protected:
            Shape<dims> shape;
            mutable std::shared_ptr<T[]> data;
            mutable std::size_t offset;
            mutable bool isOwner = true;

            void detach() const {
                if (!isOwner) {
                    isOwner = true;
                    auto newData = std::make_unique<T[]>(this->shape.elementCount());
                    std::copy(data.get() + offset, data.get() + offset + this->shape.elementCount(), newData.get());
                    data = std::move(newData);
                    offset = 0;
                }
            }

            NdArrayFlexBase(std::initializer_list<std::size_t> shape)
            {
                this->shape = Shape<dims>(shape);
                data = std::make_unique<T[]>(this->shape.elementCount());
                offset = 0;
            }

            NdArrayFlexBase(const NdArrayFlexBase& otherArray)
            :shape(otherArray.shape)
            {
                otherArray.detach();
                isOwner = false;
                data = otherArray.data;
                offset = 0;
            }

            ~NdArrayFlexBase() = default;
            NdArrayFlexBase(NdArrayFlexBase&& otherArray) noexcept
            :shape(otherArray.shape)
            {
                otherArray.detach();
                data = std::move(otherArray.data);
                offset = otherArray.offset;
                isOwner = false;
            }

            explicit NdArrayFlexBase(const NdArrayFlexBase<T, dims + 1>& otherArray, std::size_t index)
            :shape(otherArray.shape.subShape())
            {
                data = otherArray.data;
                offset = otherArray.offset + shape.strideAt(0) * index;
                isOwner = true;
            }

        public:
            NdArrayFlexBase& operator=(const NdArrayFlexBase& otherArray)
            {
                if (this != &otherArray) {
                    otherArray.detach();
                    isOwner = false;
                    data = otherArray.data;
                    shape = otherArray.shape;
                    offset = 0;
                }
            }

            NdArrayFlexBase& operator=(NdArrayFlexBase&& otherArray) noexcept
            {
                if (this != &otherArray) {
                    otherArray.detach();
                    shape = otherArray.shape;
                    data = std::move(otherArray.data);
                    isOwner = data.unique();
                    offset = otherArray.offset;
                }
            }

            [[nodiscard]] std::size_t size() const
            {
                return shape.firstDimSize();
            }

            [[nodiscard]] std::size_t elementCount() const { return shape.elementCount(); }
        };
    }

    template<typename T, std::size_t dims>
    class NdArrayFlex : public NdArrayFlexBase<T, dims>
    {
        using base = NdArrayFlexBase<T, dims>;
        friend class NdArrayFlex<T, dims+1>;
        friend class NdArrayFlex<T, dims-1>;
        explicit NdArrayFlex(const NdArrayFlex<T, dims + 1>& otherArray, std::size_t index) : base(otherArray, index) {}

        std::ostream& print(std::ostream& os, std::size_t pad, bool initPad) const
        {
            if (initPad)
            {
                for (size_t i = 0; i < pad; i++)
                {
                    os << ' ';
                }
            }
            os << "[";

            if constexpr (dims == 2)
            {
                if (base::size() != 0)
                {
                    operator[](0).print(os, pad, false) << (base::size() > 1 ? "\n" : "");
                }
            }

            pad++;
            for (std::size_t i = 1; i < base::size(); ++i) {
                operator[](i).print(os, pad, true) << (i < base::size() - 1 ? "\n" : "");
            }
            os << "]";
            return os;
        }
    public:
        NdArrayFlex<T, dims - 1> operator[](std::size_t index)
        {
            base::detach();
            auto ret = NdArrayFlex<T, dims - 1>(*this, index);
            return std::move(ret);
        }

        NdArrayFlex<T, dims - 1> operator[](std::size_t index) const
        {
            auto ret = NdArrayFlex<T, dims - 1>(*this, index);
            return std::move(ret);
        }

        NdArrayFlex(std::initializer_list<std::size_t> shape) : base(shape) {}
        NdArrayFlex(const NdArrayFlex& otherArray) : base(otherArray) {}
        ~NdArrayFlex() = default;
        NdArrayFlex(NdArrayFlex&& otherArray) noexcept : base(std::move(otherArray)) {}


        friend std::ostream& operator<<(std::ostream& os, const NdArrayFlex<T, dims + 1>& array);

        friend std::ostream& operator<<(std::ostream& os, const NdArrayFlex& array)
        {
            os << "[";

            if (array.size() != 0)
            {
                array.operator[](0).print(os, 1, false) << (array.size() > 1 ? "\n" : "");
            }

            for (std::size_t i = 1; i < array.size(); ++i)
            {
                array[i].print(os, 1, true) << (i < array.size() - 1 ? "\n" : "");
            }
            os << "]";
            return os;
        }
    };


    template<typename T>
    class NdArrayFlex<T, 1> : public NdArrayFlexBase<T, 1>
    {
        using base = NdArrayFlexBase<T, 1>;
        friend class NdArrayFlex<T, 2>;

        std::ostream& print(std::ostream& os, std::size_t pad, bool initPad) const
        {
            if (initPad)
            {
                for (size_t i = 0; i < pad; i++)
                {
                    os << ' ';
                }
            }
            os << "[";
            for (std::size_t i = 0; i < base::size(); ++i) {
                os << base::data[base::offset + i] << (i < base::size() - 1 ? ", " : "");
            }
            os << "]";
            return os;
        }
        explicit NdArrayFlex(const NdArrayFlex<T, 2>& otherArray, std::size_t index) : base(otherArray, index) {}
    public:
        NdArrayFlex(std::initializer_list<std::size_t> shape) : base(shape) {}
        NdArrayFlex(const NdArrayFlex& otherArray) : base(otherArray) {};
        ~NdArrayFlex() = default;
        NdArrayFlex(NdArrayFlex&& otherArray) noexcept : base(std::move(otherArray)) {}
        T& operator[](std::size_t index)
        {
            base::detach();
            return base::data[index];
        }

        T& operator[](std::size_t index) const
        {
            return base::data[index];
        }

        friend std::ostream& operator<<(std::ostream& os, const NdArrayFlex<T, 1>& array)
        {

            os << "[";
            for (std::size_t i = 0; i < array.size(); ++i)
            {
                os << array.data[array.offset + i] << (i < array.size() - 1 ? ", " : "");
            }
            os << "]";
            return os;
        }


        friend std::ostream& operator<<(std::ostream& os, const NdArrayFlex<T, 2>& array);
    };
}

