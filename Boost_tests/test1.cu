//
// Created by bobini on 30.12.2021.
//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN  // in only one cpp file
#include <boost/test/unit_test.hpp>
#include "NdArrayVariadic.h"

BOOST_AUTO_TEST_CASE(trueCheck)
{
    BOOST_CHECK(true);
}

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