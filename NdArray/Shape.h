//
// Created by bobini on 26.12.2021.
//

#ifndef BALG_SHAPE_H
#define BALG_SHAPE_H

#include <array>
#include <cstdint>

template<std::size_t shapeDims>
class Shape
{
    static_assert(shapeDims >= 1);
    friend class Shape<shapeDims+1>;
    friend class Shape<shapeDims-1>;
    std::array<std::size_t, shapeDims> shape;
    std::array<std::size_t, shapeDims> strides;
    std::size_t totalSize;

    explicit Shape(const Shape<shapeDims+1>& biggerShape)
    {
        std::copy(biggerShape.shape.begin() + 1, biggerShape.shape.end(), shape.begin());
        std::copy(biggerShape.strides.begin() + 1, biggerShape.strides.end(), strides.begin());
        totalSize = biggerShape.totalSize / biggerShape[0];
    }
public:
    Shape(std::initializer_list<std::size_t> shape)
    {
        std::copy(shape.begin(), shape.end(), this->shape.begin());
        this->shape[shapeDims-1] = 1;

        strides[shapeDims-1] = 1;
        for (std::size_t i = shapeDims-2; i != SIZE_MAX; i--)
        {
            strides[i] = strides[i+1] * this->shape[i];
        }
        totalSize = strides[0] * this->shape[0];
    }
    std::size_t operator[](std::size_t index) const
    {
        return shape[index];
    }
    std::size_t strideAt(std::size_t index) const
    {
        return strides[index];
    };
    std::size_t size() const
    {
        return shape.size();
    }
    std::size_t elementCount() const
    {
        return totalSize;
    }
    Shape<shapeDims-1> subShape() const
    {
        return Shape<shapeDims-1>(*this);
    }

    Shape() = default;
};
#endif //BALG_SHAPE_H
