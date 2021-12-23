cmake_minimum_required(VERSION 3.21)
project(BAlg CUDA)

set(CMAKE_CUDA_STANDARD 20)

include_directories(NdArray)

add_executable(BAlg
        NdArray/Algorithms.cu
        NdArray/Algorithms.cuh
        NdArray/kernel.cu
        NdArray/NdArray.cuh)

set_target_properties(BAlg PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)