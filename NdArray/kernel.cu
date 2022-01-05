
#include "Algorithms/Algorithms.h"
#include "NdArrayFlex.h"
#include <iostream>
#include "NdArrayVariadic.h"

using namespace BAlg::DataStructures;

int main()
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

    return 0;
}