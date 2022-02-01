//
// Created by bobini on 30.12.2021.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN  // in only one cpp file
#include <boost/test/unit_test.hpp>
#include "NdArrayVariadic.h"
#include <boost/multiprecision/cpp_int.hpp>

BOOST_AUTO_TEST_CASE(trueCheck)
{
    cudaFree(nullptr);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE(NdArray_basic)

BOOST_AUTO_TEST_CASE(writingToNdArrayElementsOneDimension)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 20> testArray;
    for (size_t i = 0; i < 20; i++)
    {
        testArray[i] = i;
    }
    for (size_t i = 0; i < 20; i++)
    {
        BOOST_CHECK_EQUAL(testArray[i], i);
    }
}

BOOST_AUTO_TEST_CASE(writingToNdArrayElementsMoreDimensions)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 20, 20> testArray;
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i * j;
        }
    }
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            BOOST_CHECK_EQUAL(testArray[i][j], i * j);
        }
    }
}

BOOST_AUTO_TEST_CASE(CopyingSubArrays)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 20, 20> testArray;
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i * j;
        }
    }
    NdArray<size_t, 20> subArrays[20];

    for (size_t i = 0; i < 20; i++)
    {
        subArrays[i] = testArray[i];
    }
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            BOOST_CHECK_EQUAL(subArrays[i][j], i * j);
        }
    }
}

BOOST_AUTO_TEST_CASE(assigningSubArrays)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 20, 20> testArray;
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i * j;
        }
    }
    NdArray<size_t, 20> subArrays[20];

    for (size_t i = 0; i < 20; i++)
    {
        subArrays[i] = testArray[i];
    }
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            BOOST_CHECK_EQUAL(subArrays[i][j], i * j);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(NdArray_reduce)

BOOST_AUTO_TEST_CASE(SumOfArrayElements)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 20, 20> testArray;
    size_t actualSum = 0;
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i * j;
            actualSum += i * j;
        }
    }
    auto sum = testArray.sum();
    BOOST_CHECK_EQUAL(sum, actualSum);
}

BOOST_AUTO_TEST_CASE(ProductOfArrayElements)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 20, 20> testArray;
    size_t actualProduct = 1;
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i * j;
            actualProduct *= i * j;
        }
    }
    auto product = testArray.product();
    BOOST_CHECK_EQUAL(product, actualProduct);
}

BOOST_AUTO_TEST_CASE(sumOfElementsStressTest)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 200, 200> testArray;
    size_t actualSum = 0;
    for (size_t i = 0; i < 200; i++)
    {
        for (size_t j = 0; j < 200; j++)
        {
            testArray[i][j] = i * j;
            actualSum += testArray[i][j];
        }
    }
    auto sum = testArray.sum();
    BOOST_CHECK_EQUAL(sum, actualSum);
}

BOOST_AUTO_TEST_CASE(CustomReturnType)
{
    using namespace BAlg::DataStructures;
    NdArray<unsigned char, 20, 20> testArray;
    size_t actualSum = 0;
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i * j;
            actualSum += testArray[i][j];
        }
    }
    auto sum = testArray.sum<size_t>();
    BOOST_CHECK_EQUAL(sum, actualSum);
}

struct B
{
    constexpr explicit B(size_t i) : i(i) {}
    size_t i;
    constexpr B operator+(const B& other) const
    {
        return B(i + other.i);
    }
    constexpr B& operator+=(const B& other)
    {
        i += other.i;
        return *this;
    }
    constexpr explicit operator size_t() const
    {
        return i;
    }
    constexpr bool operator== (const B& other) const
    {
        return i == other.i;
    }
    constexpr B(const B&) = default;
    constexpr B& operator=(const B& other) = default;

    friend std::ostream& operator<<(std::ostream& os, const B& b)
    {
        os << b.i;
        return os;
    }
};

BOOST_AUTO_TEST_CASE(CustomReturnTypeUserType)
{
    using namespace BAlg::DataStructures;
    NdArray<unsigned char, 20, 20> testArray;
    B actualSum = B{0};
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i * j;
            actualSum += B(testArray[i][j]);
        }
    }
    auto sum = testArray.sum<B>();
    BOOST_CHECK_EQUAL(sum, actualSum);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(NdArray_map)

BOOST_AUTO_TEST_CASE(NdArrayMap)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 20, 20> testArray;
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i + j;
        }
    }
    auto mappedArray = testArray.map([]__device__ __host__(size_t i)-> size_t { return i * 2; });
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            auto res1 = mappedArray[i][j];
            auto res2 = testArray[i][j] * 2;
            BOOST_CHECK_EQUAL(res1, res2);
        }
    }
}


BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_CASE(CustomReturnTypeMultiprecision)
{
    using namespace BAlg::DataStructures;
    NdArray<unsigned char, 20, 20> testArray;
    boost::multiprecision::uint128_t actualSum = 0;
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
        {
            testArray[i][j] = i * j;
            actualSum += testArray[i][j];
        }
    }
    auto sum = testArray.sum<boost::multiprecision::uint128_t>();
    BOOST_CHECK_EQUAL(sum, actualSum);
}


BOOST_AUTO_TEST_CASE(sumOfElementsStressTest2)
{
    using namespace BAlg::DataStructures;
    NdArray<size_t, 2000, 2000> testArray;
    size_t actualSum = 0;
    for (size_t i = 0; i < 2000; i++)
    {
        for (size_t j = 0; j < 2000; j++)
        {
            testArray[i][j] = i * j;
            actualSum += testArray[i][j];
        }
    }
    auto sum = testArray.sum();
    BOOST_CHECK_EQUAL(sum, actualSum);
}
