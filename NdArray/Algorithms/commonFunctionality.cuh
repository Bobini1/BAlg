//
// Created by bobini on 07.01.2022.
//

#ifndef BALG_COMMONFUNCTIONALITY_CUH
#define BALG_COMMONFUNCTIONALITY_CUH

#include <cstdio>
#include <stdexcept>

namespace BAlg::Algorithms::Implementations
{
    void checkErrors();

    cudaDeviceProp getDeviceProperties();
}

#endif //BALG_COMMONFUNCTIONALITY_CUH
