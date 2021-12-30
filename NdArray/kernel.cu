
#include "Algorithms/Algorithms.h"
#include "NdArrayFlex.h"
#include <iostream>
#include "NdArrayVariadic.h"

using namespace BAlg::DataStructures;

int main()
{
    NdArray<float, 5> zzz;

    zzz[0] = 1;

    NdArray<float, 5, 10> zzz2;

    zzz2[0][0] = 1;

    auto copying = zzz2[0];

    auto x = zzz2[0];

    auto y = zzz2[0][0];

    x[0] = 2;

    std::cout << zzz2[0][0] << std::endl;

    return 0;
}