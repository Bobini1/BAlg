//
// Created by bobini on 07.01.2022.
//

#include "commonFunctionality.cuh"

void BAlg::Algorithms::Implementations::checkErrors()
{
    cudaError_t cudaStatus;
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
}

cudaDeviceProp BAlg::Algorithms::Implementations::getDeviceProperties() {
    static cudaDeviceProp props;
    static bool initialized = false;
    if (initialized) {
        return props;
    }
    int device;
    cudaGetDevice(&device);
    auto error  = cudaGetDeviceProperties(&props, device);
    if (error != cudaSuccess) {
        throw std::runtime_error("Error getting device properties");
    }
    return props;
}